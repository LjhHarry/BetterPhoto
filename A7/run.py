"""
A7: Method Comparison -- PSR-Net vs PartialConv vs GatedConv vs LaMa

Compares PSR-Net against three established inpainting paradigms.  All baselines
receive the GT mask concatenated to the dirty image (4-channel input); PSR-Net
learns the mask automatically from 3-channel input.

Degradation types tested: block occlusion, gaussian blur, JPEG artifacts.

Outputs (A7/outputs/):
  - results.json           Full per-config + per-method aggregates
  - bar_comparison.png     Bar chart across key metrics
  - method_radar.png       Radar chart of normalized metrics
  - visual_grid_*.png      One visual comparison grid per degradation type
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model_factory import create_model, count_parameters
from common.data_utils import generate_synthetic_pair, SyntheticDataset, SDImageDataset
from common.training import CheckpointManager, _format_time
from common.evaluation import (
    compute_psnr, compute_ssim, compute_iou,
    compute_mask_contrast_ratio, compute_l1_improvement,
    measure_inference_performance, sanitize_metric_array,
)
from common.visualization import plot_method_comparison

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================================================================
# Baseline Models  (all tuned to ~1M parameters with base=32)
# ===================================================================

class PartialConvBaseline(nn.Module):
    """
    Simple UNet taking 4ch (dirty+mask) input, predicting 3ch repaired image.
    Mimics the idea of PartialConv where the mask guides inpainting.
    """
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        c = base
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c*2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.dec3 = nn.ConvTranspose2d(c*4, c*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c*4, c*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c*2, c, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c*2, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = F.relu(self.dec3(e3))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


class GatedConvBaseline(nn.Module):
    """
    UNet where every conv layer has a parallel gating branch (sigmoid) that
    learns a soft mask, multiplied element-wise with the main branch features.
    """
    def __init__(self, in_ch=4, base=24):
        super().__init__()
        self.in_gate = self._make_gated(in_ch, base)
        self.enc1 = self._make_gated(base, base)
        self.down1 = nn.Conv2d(base, base*2, 3, stride=2, padding=1)
        self.enc2 = self._make_gated(base*2, base*2)
        self.down2 = nn.Conv2d(base*2, base*4, 3, stride=2, padding=1)
        self.bn = self._make_gated(base*4, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1)
        self.dec2 = self._make_gated(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1)
        self.dec1 = self._make_gated(base*2, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    @staticmethod
    def _make_gated(in_c, out_c):
        return nn.ModuleDict({
            "feat": nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True)),
            "gate": nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.Sigmoid()),
        })

    @staticmethod
    def _apply_gate(m, x):
        return m["feat"](x) * m["gate"](x)

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        e0 = F.relu(self._apply_gate(self.in_gate, inp))
        e1 = F.relu(self._apply_gate(self.enc1, e0))
        d1 = F.relu(self.down1(e1))
        e2 = F.relu(self._apply_gate(self.enc2, d1))
        d2 = F.relu(self.down2(e2))
        b = F.relu(self._apply_gate(self.bn, d2))
        u2 = F.relu(self.up2(b))
        d2u = F.relu(self._apply_gate(self.dec2, torch.cat([u2, e2], dim=1)))
        u1 = F.relu(self.up1(d2u))
        d1u = F.relu(self._apply_gate(self.dec1, torch.cat([u1, e1], dim=1)))
        return self.out(d1u)


class LaMaLikeBaseline(nn.Module):
    """
    UNet with 7x7/5x5 kernels and dilated bottleneck convolutions (dilation=2,4,2)
    for a larger effective receptive field, inspired by LaMa.
    """
    def __init__(self, in_ch=4, base=24):
        super().__init__()
        c = base
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 7, padding=3), nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c*2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c*2, c*2, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*2, 5, padding=2), nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c*4, c*4, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(c*4, c*4, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
            nn.Conv2d(c*4, c*4, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c*4, c*2, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*2, 5, padding=2), nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(c*2, c, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c*2, c, 7, padding=3), nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, 7, padding=3),
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        e1 = self.enc1(inp)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        b = self.bottleneck(d2)
        u2 = F.relu(self.up2(b))
        d2u = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.relu(self.up1(d2u))
        return self.dec1(torch.cat([u1, e1], dim=1))


# ===================================================================
# Degradation helpers
# ===================================================================

def _make_degradation_fn(degrade_type):
    if degrade_type == "block":
        def _fn(size, seed):
            return generate_synthetic_pair(size, num_defects=3, defect_size=8, seed=seed)
        return _fn
    elif degrade_type == "blur":
        from common.data_utils import apply_gaussian_blur
        def _fn(size, seed):
            np.random.seed(seed)
            base = np.random.rand() * 0.7 + 0.3
            var = np.random.rand() * 0.3
            clean = np.clip(np.random.rand(size, size, 3)*var+base, 0, 1).astype(np.float32)
            d, m = apply_gaussian_blur(clean, num_blobs=3, max_sigma=5.0)
            return d, clean, m
        return _fn
    elif degrade_type == "jpeg":
        from common.data_utils import apply_jpeg_artifact
        def _fn(size, seed):
            np.random.seed(seed)
            base = np.random.rand() * 0.7 + 0.3
            var = np.random.rand() * 0.3
            clean = np.clip(np.random.rand(size, size, 3)*var+base, 0, 1).astype(np.float32)
            d, m = apply_jpeg_artifact(clean, num_regions=3, quality=10)
            return d, clean, m
        return _fn
    return _make_degradation_fn("block")


def _make_sd_degradation_fn(degrade_type):
    """SD 模式适配器：接受 clean image，返回 (dirty, gt_mask)。

    与 _make_degradation_fn 的区别：_fn(size, seed) 自行生成 clean 并返回 3 元组
    (供 SyntheticDataset 使用)；本函数返回的 _fn(clean) 接受外部 clean 并返回 2 元组
    (供 SDImageDataset 使用，签名匹配 common/data_utils.py:446)。
    """
    from common.data_utils import (apply_random_degradation,
                                   apply_gaussian_blur, apply_jpeg_artifact)

    if degrade_type == "block":
        def _fn(clean):
            dirty, gt_mask, _ = apply_random_degradation(clean, "block")
            return dirty, gt_mask
        return _fn
    elif degrade_type == "blur":
        def _fn(clean):
            dirty, gt_mask = apply_gaussian_blur(clean, num_blobs=3, max_sigma=5.0)
            return dirty, gt_mask
        return _fn
    elif degrade_type == "jpeg":
        def _fn(clean):
            dirty, gt_mask = apply_jpeg_artifact(clean, num_regions=3, quality=10)
            return dirty, gt_mask
        return _fn
    return _make_sd_degradation_fn("block")


# Degradation-type-specific lambda_sparse map
# block:  sparse defects → strong sparsity is fine
# blur/jpeg: diffuse degradation → need weaker sparsity to avoid mask collapse
# Paper Table 1: λ_s=0.1 is optimal (32.9 dB PSNR, 382,527x mask contrast).
# λ_s=0.03 is the "unstable intermediate state" (23.1 dB).  Use 0.1 for all.
LAMBDA_SPARSE_MAP = {"block": 0.1, "blur": 0.1, "jpeg": 0.1, "noise": 0.05, "mixed": 0.1}




class BaselineTrainer:
    def __init__(self, model, device, lr=1e-3, epochs=80, save_dir=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.ckpt_mgr = CheckpointManager(save_dir, keep_last=3) if save_dir else None

    def train(self, train_ldr, val_ldr, verbose=True, print_every=10):
        import time
        history = defaultdict(list)
        best_psnr = -float("inf")
        best_state = None
        start_epoch = 0

        # ── 断点续训 ──
        if self.ckpt_mgr:
            start_epoch, loaded_history = self.ckpt_mgr.load_latest(
                self.model, self.optimizer, self.scheduler)
            if start_epoch >= self.epochs:
                print(f"  [Skip] Already completed {self.epochs} epochs, loading best model")
                best_path = os.path.join(self.ckpt_mgr.save_dir, "best_model.pt")
                if os.path.exists(best_path):
                    self.model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"])
                return loaded_history if loaded_history else {}
            if loaded_history:
                history.update(loaded_history)
                # Recover best PSNR from history
                if "val_psnr" in history and history["val_psnr"]:
                    best_psnr = max(history["val_psnr"])
                print(f"  [Resume] Continuing from epoch {start_epoch} | best PSNR so far: {best_psnr:.1f}")

        train_start = time.time()
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            loss_sum = 0.0
            for dirty, clean, gt_mask in train_ldr:
                dirty, clean, gt_mask = dirty.to(self.device), clean.to(self.device), gt_mask.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(dirty, gt_mask)
                loss = F.l1_loss(pred, clean)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
            self.scheduler.step()
            history["train_loss"].append(loss_sum / len(train_ldr))

            val_metrics = {}
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                val_metrics = self._validate(val_ldr)
                for k, v in val_metrics.items():
                    history.setdefault(f"val_{k}", []).append(v)
                    history.setdefault("val_epoch", []).append(epoch)
                if val_metrics.get("psnr", 0) > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 定期保存
            if self.ckpt_mgr and (epoch % 10 == 0 or epoch == self.epochs - 1):
                self.ckpt_mgr.save(self.model, self.optimizer, epoch, dict(history), val_metrics, self.scheduler)

            if verbose and (epoch % print_every == 0 or epoch == self.epochs - 1):
                elapsed = time.time() - train_start
                done = epoch + 1 - start_epoch
                total = self.epochs - start_epoch
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"  Epoch {epoch:3d}/{self.epochs} [{100*(epoch+1)//self.epochs}%] | "
                      f"Loss={history['train_loss'][-1]:.4f} | "
                      f"PSNR={history.get('val_psnr', [0])[-1]:.1f} | ETA={_format_time(eta)}")
        if best_state:
            self.model.load_state_dict(best_state)
        return dict(history)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        psnr_l, ssim_l, l1_l, l1i_l = [], [], [], []
        for dirty, clean, gt_mask in loader:
            dirty, clean, gt_mask = dirty.to(self.device), clean.to(self.device), gt_mask.to(self.device)
            pred = self.model(dirty, gt_mask)
            psnr_l.append(compute_psnr(pred, clean))
            ssim_l.append(compute_ssim(pred, clean))
            l1_l.append(F.l1_loss(pred, clean).item())
            l1i_l.append(compute_l1_improvement(dirty, pred, clean))
        return {"psnr": float(np.mean(psnr_l)), "ssim": float(np.mean(ssim_l)),
                "l1_loss": float(np.mean(l1_l)), "l1_improvement": float(np.mean(l1i_l))}


class PSRNetTrainer:
    def __init__(self, model, device, lr=1e-3, epochs=80, lambda_sparse=0.1,
                 warmup_epochs=40, save_dir=None, degradation_type="block",
                 use_distill=False):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.degradation_type = degradation_type
        self.use_distill = use_distill
        self.ckpt_mgr = CheckpointManager(save_dir, keep_last=3) if save_dir else None

    def _lamb(self, ep):
        lambda_target = LAMBDA_SPARSE_MAP.get(self.degradation_type, 0.1)
        if ep < self.warmup_epochs:
            return lambda_target * ep / max(self.warmup_epochs, 1)
        return lambda_target

    def train(self, train_ldr, val_ldr, verbose=True, print_every=10):
        import time
        history = defaultdict(list)
        best_psnr = -float("inf")
        best_state = None
        start_epoch = 0

        # ── 断点续训 ──
        if self.ckpt_mgr:
            start_epoch, loaded_history = self.ckpt_mgr.load_latest(
                self.model, self.optimizer, self.scheduler)
            if start_epoch >= self.epochs:
                print(f"  [Skip] Already completed {self.epochs} epochs, loading best model")
                best_path = os.path.join(self.ckpt_mgr.save_dir, "best_model.pt")
                if os.path.exists(best_path):
                    self.model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"])
                return loaded_history if loaded_history else {}
            if loaded_history:
                history.update(loaded_history)
                if "val_psnr" in history and history["val_psnr"]:
                    best_psnr = max(history["val_psnr"])
                print(f"  [Resume] Continuing from epoch {start_epoch} | best PSNR so far: {best_psnr:.1f}")

        train_start = time.time()
        lambda_target = LAMBDA_SPARSE_MAP.get(self.degradation_type, 0.1)
        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            loss_sum = 0.0
            lamb = self._lamb(epoch)
            for dirty, clean, gt_mask in train_ldr:
                dirty, clean, gt_mask = dirty.to(self.device), clean.to(self.device), gt_mask.to(self.device)
                self.optimizer.zero_grad()
                residual, mask = self.model(dirty)
                refined = dirty + residual * mask
                # Paper loss: L = L1 + λ_s * mean(M).  No GT mask in loss.
                loss = F.l1_loss(refined, clean) + lamb * mask.mean()
                if self.use_distill:
                    # Optional BCE distillation (off by default — not in paper)
                    loss_distill = F.binary_cross_entropy(
                        mask.clamp(1e-6, 1-1e-6), gt_mask)
                    loss = loss + loss_distill
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
            self.scheduler.step()
            history["train_loss"].append(loss_sum / len(train_ldr))

            val_metrics = {}
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                val_metrics = self._validate(val_ldr)
                for k, v in val_metrics.items():
                    history.setdefault(f"val_{k}", []).append(v)
                    history.setdefault("val_epoch", []).append(epoch)
                if val_metrics.get("psnr", 0) > best_psnr:
                    best_psnr = val_metrics["psnr"]
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # 定期保存
            if self.ckpt_mgr and (epoch % 10 == 0 or epoch == self.epochs - 1):
                self.ckpt_mgr.save(self.model, self.optimizer, epoch, dict(history), val_metrics, self.scheduler)

            if verbose and (epoch % print_every == 0 or epoch == self.epochs - 1):
                elapsed = time.time() - train_start
                done = epoch + 1 - start_epoch
                total = self.epochs - start_epoch
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"  Epoch {epoch:3d}/{self.epochs} [{100*done//total}%] | "
                      f"Loss={history['train_loss'][-1]:.4f} | λ_s={lamb:.4f} | "
                      f"PSNR={history.get('val_psnr', [0])[-1]:.1f} | ETA={_format_time(eta)}")
        if best_state:
            self.model.load_state_dict(best_state)
        return dict(history)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        psnr_l, ssim_l, l1_l, l1i_l, iou_l, mc_l, mm_l = [], [], [], [], [], [], []
        for dirty, clean, gt_mask in loader:
            dirty, clean, gt_mask = dirty.to(self.device), clean.to(self.device), gt_mask.to(self.device)
            refined, _res, mask = self.model.refine(dirty)
            psnr_l.append(compute_psnr(refined, clean))
            ssim_l.append(compute_ssim(refined, clean))
            l1_l.append(F.l1_loss(refined, clean).item())
            l1i_l.append(compute_l1_improvement(dirty, refined, clean))
            iou_l.append(compute_iou(mask, gt_mask))
            mc_l.append(compute_mask_contrast_ratio(mask, gt_mask))
            mm_l.append(mask.mean().item())
        return {"psnr": float(np.mean(psnr_l)), "ssim": float(np.mean(ssim_l)),
                "l1_loss": float(np.mean(l1_l)), "l1_improvement": float(np.mean(l1i_l)),
                "iou": float(np.mean(sanitize_metric_array(iou_l))) if sanitize_metric_array(iou_l) else 0.0,
                "mask_contrast": float(np.mean(sanitize_metric_array(mc_l))) if sanitize_metric_array(mc_l) else 0.0,
                "mask_mean": float(np.mean(mm_l))}


# ===================================================================
# Final evaluation (with saved models)
# ===================================================================

@torch.no_grad()
def evaluate_final(model, loader, device, model_type):
    model.eval()
    all_m = defaultdict(list)
    for dirty, clean, gt_mask in loader:
        dirty, clean, gt_mask = dirty.to(device), clean.to(device), gt_mask.to(device)
        if model_type == "psrnet":
            refined, _res, mask = model.refine(dirty)
            all_m["mask_mean"].append(mask.mean().item())
            all_m["iou"].append(compute_iou(mask, gt_mask))
            all_m["mask_contrast"].append(compute_mask_contrast_ratio(mask, gt_mask))
        else:
            refined = model(dirty, gt_mask)
            # 基线方法不产生掩膜，跳过 iou 和 mask_contrast（不注入 NaN）
        all_m["psnr"].append(compute_psnr(refined, clean))
        all_m["ssim"].append(compute_ssim(refined, clean))
        all_m["l1_improvement"].append(compute_l1_improvement(dirty, refined, clean))
    # Inference perf
    sample = next(iter(loader))
    d_sample = sample[0][:1].to(device)
    m_sample = sample[2][:1].to(device) if model_type != "psrnet" else None
    if model_type == "psrnet":
        perf = measure_inference_performance(model, d_sample, num_warmup=30, num_runs=100)
    else:
        # Wrap baseline model in nn.Module so measure_inference_performance works
        class _BaselineWrapper(nn.Module):
            def __init__(self, m, mask):
                super().__init__()
                self.m = m
                self.mask = mask
            def forward(self, x):
                return self.m(x, self.mask.expand(x.size(0), -1, -1, -1))
        wrapped = _BaselineWrapper(model, m_sample).to(device)
        perf = measure_inference_performance(wrapped, d_sample, num_warmup=30, num_runs=100)
    all_m["inference_time_ms"] = [perf["inference_time_ms"]]
    all_m["gpu_memory_mb"] = [perf["gpu_memory_mb"]]
    all_m["params"] = [count_parameters(model)]
    # 使用 sanitize_metric_array 过滤 NaN/Inf 后再求均值
    summary = {}
    for k, v in all_m.items():
        clean = sanitize_metric_array(v)
        if clean:
            summary[k] = float(np.mean(clean))
    return summary


# ===================================================================
# Visualisation
# ===================================================================

def _to_np(t):
    img = t[0].detach().cpu().numpy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    elif img.shape[0] == 1:
        img = img.squeeze(0)
    return np.clip(img, 0, 1)


def plot_visual_grid(models_dict, dataloader, deg_name, device, out_dir):
    """models_dict: {name: (model, model_type)}"""
    dirty, clean, gt_mask = next(iter(dataloader))
    dirty, clean, gt_mask = dirty.to(device), clean.to(device), gt_mask.to(device)

    methods = list(models_dict.keys())
    n_cols = 5  # Dirty | Refined | Mask | GT | GT-Mask
    n_rows = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.5, n_rows*2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_names = ["Dirty", "Refined", "Mask", "GT", "GT-Mask"]
    for ax, name in zip(axes[0], col_names):
        ax.set_title(name, fontsize=10)

    for i, (mname, (model, mtype)) in enumerate(models_dict.items()):
        model.eval()
        with torch.no_grad():
            if mtype == "psrnet":
                refined, _res, mask = model.refine(dirty[:1])
            else:
                refined = model(dirty[:1], gt_mask[:1])
                mask = gt_mask[:1]  # baselines don't produce a mask

        axes[i, 0].imshow(_to_np(dirty))
        axes[i, 1].imshow(_to_np(refined))
        axes[i, 2].imshow(_to_np(mask), cmap="hot")
        axes[i, 3].imshow(_to_np(clean))
        axes[i, 4].imshow(_to_np(gt_mask), cmap="hot")
        for j in range(n_cols):
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(mname, fontsize=9, rotation=0, labelpad=40, va="center")

    plt.suptitle(f"Method Comparison — Degradation: {deg_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, f"visual_grid_{deg_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_radar(agg_results, out_dir):
    categories = ["PSNR", "SSIM", "IoU", "Mask Contrast", "L1 Impr."]
    n = len(categories)
    angles = [a / float(n) * 2 * pi for a in range(n)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    colors = {"PartialConv": "#1f77b4", "GatedConv": "#ff7f0e",
              "LaMa": "#2ca02c", "PSR-Net": "#d62728"}

    for method in ["PartialConv", "GatedConv", "LaMa", "PSR-Net"]:
        m = agg_results[method]
        mc = m.get("mask_contrast", 0)
        if mc is None or (isinstance(mc, float) and np.isnan(mc)):
            mc = 0
        vals = [
            m.get("psnr", 0),
            m.get("ssim", 0),
            m.get("iou", 0) if m.get("iou", 0) is not None and not (isinstance(m.get("iou", 0), float) and np.isnan(m.get("iou", 0))) else 0,
            min(mc, 10),
            m.get("l1_improvement", 0),
        ]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.1, color=colors[method])
        ax.plot(angles, vals, "o-", linewidth=2, label=method, color=colors[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Method Comparison Radar", fontsize=14, pad=25)
    path = os.path.join(out_dir, "method_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="A7: Method Comparison")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_sd", action="store_true",
                        help="Use SD v1.5 generated images instead of random noise")
    parser.add_argument("--use_distill", action="store_true", default=False,
                        help="Enable BCE mask distillation (NOT in paper; paper uses "
                             "self-supervision without GT mask in loss)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    degradation_types = ["block", "blur", "jpeg"]
    all_results = {}

    # ---- Build shared datasets ----
    dataloaders = {}
    DatasetClass = SDImageDataset if args.use_sd else SyntheticDataset
    for deg in degradation_types:
        fn = _make_sd_degradation_fn(deg) if args.use_sd else _make_degradation_fn(deg)
        tr_ds = DatasetClass(args.train_samples, args.image_size,
                             degradation_fn=fn, seed=42,
                             device=str(device) if args.use_sd else None)
        te_ds = DatasetClass(args.test_samples, args.image_size,
                             degradation_fn=fn, seed=1042,
                             device=str(device) if args.use_sd else None)
        dataloaders[deg] = (
            DataLoader(tr_ds, args.batch_size, shuffle=True),
            DataLoader(te_ds, args.batch_size, shuffle=False),
        )

    # ---- Train & evaluate each method on each degradation ----
    method_specs = {
        "PartialConv": ("baseline", lambda: PartialConvBaseline(in_ch=4, base=32)),
        "GatedConv":   ("baseline", lambda: GatedConvBaseline(in_ch=4, base=24)),
        "LaMa":        ("baseline", lambda: LaMaLikeBaseline(in_ch=4, base=24)),
        "PSR-Net":     ("psrnet",   lambda: create_model("standard", base_channels=32, device=str(device))),
    }

    # Store trained models for visual grids
    trained_models = {deg: {} for deg in degradation_types}

    for mname, (mtype, factory) in method_specs.items():
        for deg in degradation_types:
            cfg_key = f"{mname}_{deg}"
            print(f"\n{'='*60}")
            print(f"  Training: {cfg_key}")
            print(f"{'='*60}")

            train_ldr, test_ldr = dataloaders[deg]
            model = factory()

            if mtype == "psrnet":
                trainer = PSRNetTrainer(model, device, lr=args.lr, epochs=args.epochs,
                                        warmup_epochs=args.epochs//2,
                                        degradation_type=deg,
                                        use_distill=args.use_distill,
                                        save_dir=os.path.join(out_dir, f"{cfg_key}_ckpt"))
            else:
                trainer = BaselineTrainer(model, device, lr=args.lr, epochs=args.epochs,
                                          save_dir=os.path.join(out_dir, f"{cfg_key}_ckpt"))
            trainer.train(train_ldr, test_ldr, print_every=10)

            metrics = evaluate_final(model, test_ldr, device, mtype)
            all_results[cfg_key] = metrics
            trained_models[deg][mname] = (model, mtype)

            print(f"  => PSNR={metrics.get('psnr',0):.2f}  SSIM={metrics.get('ssim',0):.4f}  "
                  f"IoU={metrics.get('iou',0):.4f}  Params={metrics.get('params',0)/1e3:.1f}K")

    # ---- Aggregate per method (mean across degradations) ----
    agg = {}
    for mname in ["PartialConv", "GatedConv", "LaMa", "PSR-Net"]:
        vals = defaultdict(list)
        for deg in degradation_types:
            for k, v in all_results[f"{mname}_{deg}"].items():
                vals[k].append(v)
        # 使用 sanitize_metric_array 过滤 NaN 后再求均值
        agg[mname] = {}
        for k, v in vals.items():
            clean = sanitize_metric_array(v)
            if clean:
                agg[mname][k] = float(np.mean(clean))

    # ---- Save JSON ----
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({"per_config": all_results, "per_method": agg,
                    "degradation_types": degradation_types}, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Bar chart ----
    bar_data = {}
    for mname in ["PartialConv", "GatedConv", "LaMa", "PSR-Net"]:
        bar_data[mname] = {
            "psnr": agg[mname].get("psnr", 0),
            "ssim": agg[mname].get("ssim", 0),
            "mask_contrast_ratio": agg[mname].get("mask_contrast", 0),
            "iou": agg[mname].get("iou", 0),
            "l1_improvement_pct": agg[mname].get("l1_improvement", 0),
        }
    bar_path = os.path.join(out_dir, "bar_comparison.png")
    plot_method_comparison(bar_data,
                           metrics=["psnr", "ssim", "mask_contrast_ratio", "iou", "l1_improvement_pct"],
                           save_path=bar_path, figsize=(16, 5))
    print(f"Bar chart saved to {bar_path}")

    # ---- Radar chart ----
    radar_path = plot_radar(agg, out_dir)
    print(f"Radar chart saved to {radar_path}")

    # ---- Visual grids ----
    for deg in degradation_types:
        grid_path = plot_visual_grid(trained_models[deg], dataloaders[deg][1],
                                      deg, device, out_dir)
        print(f"Visual grid ({deg}) saved to {grid_path}")

    # ---- Summary table ----
    print("\n" + "=" * 92)
    print("  METHOD COMPARISON SUMMARY  (mean across block/blur/jpeg)")
    print("=" * 92)
    header = (f"{'Method':<14} {'PSNR':>7} {'SSIM':>7} {'IoU':>7} {'M.Contr':>8} "
              f"{'L1Impr%':>8} {'Time_ms':>8} {'ParamK':>8}")
    print(header)
    print("-" * 92)
    for m in ["PartialConv", "GatedConv", "LaMa", "PSR-Net"]:
        a = agg[m]
        print(f"{m:<14} {a.get('psnr',0):7.2f} {a.get('ssim',0):7.4f} {a.get('iou',0):7.4f} "
              f"{a.get('mask_contrast',0):8.2f} {a.get('l1_improvement',0):8.1f} "
              f"{a.get('inference_time_ms',0):8.2f} {a.get('params',0)/1e3:8.1f}")
    print("=" * 92)

    print("\nA7 experiment complete!")


if __name__ == "__main__":
    main()
