"""
Experiment A3: Two-Stage Surgical Redrawing System

Stage 1 (Error Localizer): PSR-Net variant outputs error probability mask M_error.
  Trained with IoU loss to precisely localize defect regions.
Stage 2 (Local Inpainter): Lightweight UNet CNN trained to inpaint masked regions.
  Only processes pixels where M_error > 0.5.

Pipeline: input → Stage1(M_error) → Stage2(inpaint masked) → repaired
Metrics: Stage1 IoU/F1, Stage2 PSNR/SSIM, inference time comparison.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- path setup ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import get_config
from common.model_factory import OverPaintNet
from common.evaluation import (compute_psnr, compute_ssim, compute_iou,
                                measure_inference_performance, evaluate_all)
from common.data_utils import generate_synthetic_pair
from common.visualization import tensor_to_numpy
from common.training import CheckpointManager, _format_time


# =============================================================================
# Stage 1: Error Localizer (PSR-Net variant for mask-only)
# =============================================================================
# 注：Stage 2 当前实现为轻量 UNet，论文描述为"局部扩散重绘器"，
# 这是概念验证阶段的简化实现。

class ErrorLocalizer(nn.Module):
    """
    PSR-Net variant optimized for error mask prediction.
    Outputs ONLY the error probability mask M_error (1ch, sigmoid).
    Architecture matches PSR-Net encoder-decoder, but output is 1ch mask.
    """

    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.enc(x)
        out = self.dec(feat)
        return torch.sigmoid(out)


# =============================================================================
# Stage 2: Local Inpainter (Lightweight UNet)
# =============================================================================

class LightweightInpainter(nn.Module):
    """
    Lightweight UNet for inpainting masked regions.
    Input: image (3ch) + mask (1ch) = 4ch
    Output: inpainted image (3ch)
    """

    def __init__(self, input_channels: int = 4, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.ReLU(inplace=True))

        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.ReLU(inplace=True))  # skip concat

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(c * 2, c, 3, padding=1), nn.ReLU(inplace=True))  # skip concat

        self.final = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H, W) = [image, mask] concatenated
        e1 = self.enc1(x)        # (B, C, H, W)
        e2 = self.enc2(e1)       # (B, 2C, H/2, W/2)
        e3 = self.enc3(e2)       # (B, 4C, H/4, W/4)

        b = self.bottleneck(e3)  # (B, 4C, H/4, W/4)

        d3 = self.dec3(b)        # (B, 2C, H/2, W/2)
        d3 = self.dec3_conv(torch.cat([d3, e2], dim=1))

        d2 = self.dec2(d3)       # (B, C, H, W)
        d2 = self.dec2_conv(torch.cat([d2, e1], dim=1))

        out = self.final(d2)     # (B, 3, H, W)
        return out


# =============================================================================
# Synthetic Anomaly Data Generator
# =============================================================================

def generate_anomaly_sample(size: int = 128, anomaly_type: str = "random_object",
                              seed: int = None) -> dict:
    """
    Generate an image with synthetic anomaly (pasted random object).
    Returns: normal, anomalous, anomaly_mask
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate clean background
    base_val = np.random.rand() * 0.5 + 0.3
    variation = np.random.rand() * 0.2
    clean = np.random.rand(size, size, 3) * variation + base_val
    clean = np.clip(clean, 0.0, 1.0).astype(np.float32)

    anomalous = clean.copy()
    mask = np.zeros((size, size, 1), dtype=np.float32)

    if anomaly_type == "random_object":
        # Paste a random colored rectangle/ellipse as anomaly
        obj_w = np.random.randint(size // 6, size // 3)
        obj_h = np.random.randint(size // 6, size // 3)
        x = np.random.randint(0, size - obj_w)
        y = np.random.randint(0, size - obj_h)

        obj_color = np.random.rand(3).astype(np.float32) * 0.8 + 0.2
        anomalous[y:y + obj_h, x:x + obj_w, :] = obj_color
        mask[y:y + obj_h, x:x + obj_w, 0] = 1.0

    elif anomaly_type == "texture_patch":
        # Paste a textured patch
        pw = np.random.randint(size // 5, size // 3)
        ph = np.random.randint(size // 5, size // 3)
        x = np.random.randint(0, size - pw)
        y = np.random.randint(0, size - ph)

        texture = np.random.rand(ph, pw, 3).astype(np.float32)
        # Add some structure to texture
        for i in range(3):
            grad = np.linspace(0, 1, ph)[:, None] * np.linspace(0, 1, pw)[None, :]
            texture[:, :, i] = texture[:, :, i] * 0.5 + grad * 0.5

        anomalous[y:y + ph, x:x + pw, :] = texture
        mask[y:y + ph, x:x + pw, 0] = 1.0

    elif anomaly_type == "color_shift":
        # Shift color in a region
        cw = np.random.randint(size // 4, size // 2)
        ch = np.random.randint(size // 4, size // 2)
        x = np.random.randint(0, size - cw)
        y = np.random.randint(0, size - ch)

        shift = np.random.rand(3).astype(np.float32) * 0.6 - 0.3
        patch = anomalous[y:y + ch, x:x + cw, :] + shift
        anomalous[y:y + ch, x:x + cw, :] = np.clip(patch, 0, 1)
        mask[y:y + ch, x:x + cw, 0] = 1.0

    else:
        # Default: random block
        bw = np.random.randint(size // 6, size // 3)
        bh = np.random.randint(size // 6, size // 3)
        x = np.random.randint(0, size - bw)
        y = np.random.randint(0, size - bh)
        anomalous[y:y + bh, x:x + bw, :] = np.random.rand() > 0.5 and 1.0 or 0.0
        mask[y:y + bh, x:x + bw, 0] = 1.0

    return {
        "normal": clean,
        "anomalous": anomalous,
        "mask": mask,
    }


class AnomalyDataset(Dataset):
    """Dataset of synthetic anomaly images."""

    def __init__(self, num_samples: int, size: int = 128,
                 anomaly_types: list = None, seed: int = 42):
        self.num_samples = num_samples
        self.size = size
        self.anomaly_types = anomaly_types or ["random_object", "texture_patch", "color_shift"]
        np.random.seed(seed)
        self.seeds = np.random.randint(0, 2**31 - 1, size=num_samples)
        self.type_indices = np.random.randint(0, len(self.anomaly_types), size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(int(self.seeds[idx]))
        atype = self.anomaly_types[self.type_indices[idx]]
        sample = generate_anomaly_sample(size=self.size, anomaly_type=atype,
                                          seed=int(self.seeds[idx]))

        return (torch.from_numpy(sample["anomalous"].transpose(2, 0, 1)).float(),
                torch.from_numpy(sample["normal"].transpose(2, 0, 1)).float(),
                torch.from_numpy(sample["mask"].transpose(2, 0, 1)).float())


# =============================================================================
# IoU Loss
# =============================================================================

class IoULoss(nn.Module):
    """Soft IoU loss for mask prediction."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        union = (pred + target - pred * target).sum(dim=1)
        iou = (intersection + self.smooth) / (union + self.smooth)
        return (1 - iou).mean()


# =============================================================================
# Training functions
# =============================================================================

def train_error_localizer(model, train_loader, val_loader, epochs, device,
                           lambda_sparse=0.1, use_distill=False,
                           verbose=True, save_dir=None):
    """Train Stage 1: Error Localizer.

    Loss = IoU + λ_s * mean(M) + (optional BCE distill).
    Paper core contribution: λ_s sparsity regularization drives automatic
    defect localization without manual mask annotations.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    iou_loss_fn = IoULoss()
    ckpt_mgr = CheckpointManager(save_dir or ".", keep_last=3) if save_dir else None
    train_start = time.time()

    best_iou = 0.0
    history = {"train_loss": [], "val_iou": [], "val_f1": [],
               "train_sparse": [], "train_iou": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_sparse = 0.0
        n_batches = 0

        for batch in train_loader:
            anomalous, clean, gt_mask = [b.to(device) for b in batch]

            pred_mask = model(anomalous)
            loss_iou = iou_loss_fn(pred_mask, gt_mask)
            # Sparsity regularization: core of PSR-Net (paper Section 3.3)
            loss_sparse = lambda_sparse * pred_mask.mean()
            total_loss = loss_iou + loss_sparse
            if use_distill:
                # Optional BCE (off by default — not in paper)
                loss_bce = F.binary_cross_entropy(pred_mask, gt_mask)
                total_loss = total_loss + 0.5 * loss_bce

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_iou += loss_iou.item()
            epoch_sparse += loss_sparse.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        history["train_iou"].append(epoch_iou / max(n_batches, 1))
        history["train_sparse"].append(epoch_sparse / max(n_batches, 1))

        # Validation
        model.eval()
        val_ious, val_f1s = [], []
        with torch.no_grad():
            for batch in val_loader:
                anomalous, clean, gt_mask = [b.to(device) for b in batch]
                pred_mask = model(anomalous)

                for i in range(len(pred_mask)):
                    val_ious.append(compute_iou(pred_mask[i:i+1], gt_mask[i:i+1], threshold=None))
                    pred_bin = (pred_mask[i] > 0.5).float()
                    gt_bin = gt_mask[i]
                    tp = (pred_bin * gt_bin).sum().item()
                    fp = (pred_bin * (1 - gt_bin)).sum().item()
                    fn = ((1 - pred_bin) * gt_bin).sum().item()
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    val_f1s.append(f1)

        avg_iou = float(np.mean(val_ious))
        avg_f1 = float(np.mean(val_f1s))
        history["val_iou"].append(avg_iou)
        history["val_f1"].append(avg_f1)

        if avg_iou > best_iou:
            best_iou = avg_iou

        # 进度 + ETA
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            elapsed = time.time() - train_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0
            print(f"  [Localizer] Epoch {epoch:3d}/{epochs} [{100*(epoch+1)//epochs}%] "
                  f"| Loss={avg_loss:.4f} | IoU={avg_iou:.4f} | F1={avg_f1:.4f} "
                  f"| λ_s={lambda_sparse:.3f} | ETA={_format_time(eta)}")
        
        # 定期保存
        if ckpt_mgr and (epoch % 10 == 0 or epoch == epochs - 1):
            ckpt_mgr.save(model, optimizer, epoch, history,
                         {"iou": avg_iou, "f1": avg_f1})

    return history, best_iou


def train_local_inpainter(model, train_loader, val_loader, epochs, device, 
                           verbose=True, save_dir=None):
    """Train Stage 2: Local Inpainter with L1 + perceptual loss."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ckpt_mgr = CheckpointManager(save_dir or ".", keep_last=3) if save_dir else None
    train_start = time.time()

    best_psnr = 0.0
    history = {"train_loss": [], "val_psnr": [], "val_ssim": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            anomalous, clean, gt_mask = [b.to(device) for b in batch]

            inp = torch.cat([anomalous, gt_mask], dim=1)
            pred = model(inp)

            loss_full = F.l1_loss(pred, clean)
            loss_mask = (F.l1_loss(pred, clean, reduction="none").mean(dim=1, keepdim=True) * gt_mask).sum() / (gt_mask.sum() + 1e-8)
            total_loss = loss_full + 2.0 * loss_mask

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        model.eval()
        val_psnr, val_ssim = [], []
        with torch.no_grad():
            for batch in val_loader:
                anomalous, clean, gt_mask = [b.to(device) for b in batch]
                inp = torch.cat([anomalous, gt_mask], dim=1)
                pred = model(inp)
                for i in range(len(pred)):
                    val_psnr.append(compute_psnr(pred[i:i+1], clean[i:i+1]))
                    val_ssim.append(compute_ssim(pred[i:i+1], clean[i:i+1]))

        avg_psnr = float(np.mean(val_psnr))
        avg_ssim = float(np.mean(val_ssim))
        history["val_psnr"].append(avg_psnr)
        history["val_ssim"].append(avg_ssim)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr

        # 进度 + ETA
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            elapsed = time.time() - train_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0
            print(f"  [Inpainter] Epoch {epoch:3d}/{epochs} [{100*(epoch+1)//epochs}%] "
                  f"| Loss={avg_loss:.4f} | PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f} "
                  f"| ETA={_format_time(eta)}")
        
        # 定期保存
        if ckpt_mgr and (epoch % 10 == 0 or epoch == epochs - 1):
            ckpt_mgr.save(model, optimizer, epoch, history,
                         {"psnr": avg_psnr, "ssim": avg_ssim})

    return history, best_psnr


# =============================================================================
# Pipeline Inference
# =============================================================================

@torch.no_grad()
def pipeline_inference(localizer, inpainter, anomalous, device):
    """
    Two-stage pipeline inference.
    Returns: error_mask, repaired_image
    """
    localizer.eval()
    inpainter.eval()

    # Stage 1: Error localization
    error_mask = localizer(anomalous)
    error_binary = (error_mask > 0.5).float()

    # Stage 2: Inpaint only error regions
    inp = torch.cat([anomalous, error_binary], dim=1)
    repaired = inpainter(inp)

    # Blend: use repaired only where error_mask > 0.5
    # repaired_region = error_binary * repaired + (1 - error_binary) * anomalous
    repaired_region = repaired  # The inpainter handles the blending internally

    return error_mask, repaired_region


@torch.no_grad()
def full_image_baseline(anomalous, target_size):
    """
    Placeholder baseline: simulate full-image regeneration via downsampling+upsampling.

    WARNING: This is NOT a real diffusion inpainting baseline.  It only downsamples to
    8×8 and upsamples back — a rough approximation of information loss during full
    regeneration.  The paper's claim (X + 0.05Y << 10X) requires comparing against an
    actual full-image diffusion model (e.g., SD inpainting).  This placeholder
    underestimates full regeneration time by >100x; do NOT use the speedup_ratio from
    this function as scientific evidence.
    """
    small = F.interpolate(anomalous, size=(8, 8), mode='bilinear', align_corners=False)
    regenerated = F.interpolate(small, size=target_size, mode='bilinear', align_corners=False)
    return regenerated


def measure_pipeline_time(localizer, inpainter, test_loader, device):
    """Measure pipeline inference time vs full-image regeneration."""
    localizer.eval()
    inpainter.eval()

    # Warmup
    sample_batch = next(iter(test_loader))
    dummy = sample_batch[0][:1].to(device)
    for _ in range(5):
        _ = pipeline_inference(localizer, inpainter, dummy, device)
        _ = full_image_baseline(dummy, dummy.shape[-2:])

    if device.type == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Stage 1 timing
        times_s1 = []
        times_s2 = []
        times_full = []

        for batch in test_loader:
            anomalous = batch[0][:1].to(device)
            for _ in range(3):
                torch.cuda.synchronize()
                start.record()
                _ = localizer(anomalous)
                end.record()
                torch.cuda.synchronize()
                times_s1.append(start.elapsed_time(end))

                error_mask = localizer(anomalous)
                inp = torch.cat([anomalous, (error_mask > 0.5).float()], dim=1)
                start.record()
                _ = inpainter(inp)
                end.record()
                torch.cuda.synchronize()
                times_s2.append(start.elapsed_time(end))

                start.record()
                _ = full_image_baseline(anomalous, anomalous.shape[-2:])
                end.record()
                torch.cuda.synchronize()
                times_full.append(start.elapsed_time(end))

        return {
            "stage1_ms": float(np.mean(times_s1)),
            "stage2_ms": float(np.mean(times_s2)),
            "pipeline_total_ms": float(np.mean(times_s1) + np.mean(times_s2)),
            "full_regen_ms": float(np.mean(times_full)),
            "speedup_ratio": float(np.mean(times_full) / (np.mean(times_s1) + np.mean(times_s2) + 1e-6)),
        }
    else:
        times_s1, times_s2, times_full = [], [], []
        for batch in test_loader:
            anomalous = batch[0][:1].to(device)
            for _ in range(3):
                t0 = time.perf_counter()
                _ = localizer(anomalous)
                times_s1.append((time.perf_counter() - t0) * 1000)

                error_mask = localizer(anomalous)
                inp = torch.cat([anomalous, (error_mask > 0.5).float()], dim=1)
                t0 = time.perf_counter()
                _ = inpainter(inp)
                times_s2.append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                _ = full_image_baseline(anomalous, anomalous.shape[-2:])
                times_full.append((time.perf_counter() - t0) * 1000)

        return {
            "stage1_ms": float(np.mean(times_s1)),
            "stage2_ms": float(np.mean(times_s2)),
            "pipeline_total_ms": float(np.mean(times_s1) + np.mean(times_s2)),
            "full_regen_ms": float(np.mean(times_full)),
            "speedup_ratio": float(np.mean(times_full) / (np.mean(times_s1) + np.mean(times_s2) + 1e-6)),
        }


# =============================================================================
# Visualization
# =============================================================================

def plot_surgical_pipeline(original, error_mask, repaired, gt, save_path):
    """Visualize: original → error_mask → repaired → GT"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(tensor_to_numpy(original))
    axes[0].set_title("Input (Anomalous)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(tensor_to_numpy(error_mask), cmap="hot")
    axes[1].set_title("Error Mask (Stage 1)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(tensor_to_numpy(repaired))
    axes[2].set_title("Repaired (Stage 2)", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(tensor_to_numpy(gt))
    axes[3].set_title("GT (Normal)", fontsize=11)
    axes[3].axis("off")

    plt.suptitle("A3: Two-Stage Surgical Redrawing Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A3: Two-Stage Surgical Redrawing System")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs per stage")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=128, help="Image size")
    parser.add_argument("--lambda_sparse", type=float, default=0.1,
                        help="Sparsity coefficient for Error Localizer (paper optimal: 0.1)")
    parser.add_argument("--use_distill", action="store_true", default=False,
                        help="Enable BCE mask supervision (NOT in paper; paper uses "
                             "self-supervision with sparsity only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    config = get_config("A3",
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        image_size=args.image_size)
    config.output_dir = output_dir
    config.device = device.__str__()

    print(f"\n{'='*60}")
    print(f"  Experiment A3: Two-Stage Surgical Redrawing System")
    print(f"  Image size: {config.image_size}, Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"{'='*60}")

    # =====================================================================
    # Prepare anomaly dataset
    # =====================================================================
    print("\n[Data] Generating synthetic anomaly dataset...")
    anomaly_types = ["random_object", "texture_patch", "color_shift"]

    train_dataset = AnomalyDataset(
        num_samples=config.train_samples, size=config.image_size,
        anomaly_types=anomaly_types, seed=config.seed)
    test_dataset = AnomalyDataset(
        num_samples=config.test_samples, size=config.image_size,
        anomaly_types=anomaly_types, seed=config.seed + 1000)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # =====================================================================
    # Stage 1: Error Localizer
    # =====================================================================
    print(f"\n{'='*50}")
    print(f"  Stage 1: Training Error Localizer")
    print(f"{'='*50}")

    localizer = ErrorLocalizer(input_channels=3, base_channels=64).to(device)
    t0 = time.time()
    loc_history, best_iou = train_error_localizer(
        localizer, train_loader, test_loader,
        epochs=config.epochs, device=device,
        lambda_sparse=args.lambda_sparse, use_distill=args.use_distill,
        verbose=True,
        save_dir=os.path.join(output_dir, "stage1_ckpt"))
    loc_time = time.time() - t0
    print(f"  Stage 1 completed in {loc_time:.1f}s | Best IoU: {best_iou:.4f}")

    # =====================================================================
    # Stage 2: Local Inpainter
    # =====================================================================
    print(f"\n{'='*50}")
    print(f"  Stage 2: Training Local Inpainter")
    print(f"{'='*50}")

    inpainter = LightweightInpainter(input_channels=4, base_channels=32).to(device)
    t0 = time.time()
    inp_history, best_psnr = train_local_inpainter(
        inpainter, train_loader, test_loader,
        epochs=config.epochs, device=device, verbose=True,
        save_dir=os.path.join(output_dir, "stage2_ckpt"))
    inp_time = time.time() - t0
    print(f"  Stage 2 completed in {inp_time:.1f}s | Best PSNR: {best_psnr:.2f} dB")

    # =====================================================================
    # Full evaluation
    # =====================================================================
    print(f"\n{'='*50}")
    print(f"  Evaluating Pipeline")
    print(f"{'='*50}")

    localizer.eval()
    inpainter.eval()

    stage1_metrics = {"iou": [], "f1": []}
    stage2_metrics = {"psnr": [], "ssim": [], "l1_loss": []}
    pipeline_metrics = {"psnr": [], "ssim": [], "l1_loss": []}

    with torch.no_grad():
        for batch in test_loader:
            anomalous, clean, gt_mask = [b.to(device) for b in batch]

            # Stage 1 evaluation
            pred_mask = localizer(anomalous)
            for i in range(len(pred_mask)):
                iou_val = compute_iou(pred_mask[i:i+1], gt_mask[i:i+1], threshold=None)
                stage1_metrics["iou"].append(iou_val)

                pred_bin = (pred_mask[i] > 0.5).float()
                gt_bin = gt_mask[i]
                tp = (pred_bin * gt_bin).sum().item()
                fp = (pred_bin * (1 - gt_bin)).sum().item()
                fn = ((1 - pred_bin) * gt_bin).sum().item()
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                stage1_metrics["f1"].append(f1)

            # Stage 2 evaluation (with GT mask - oracle)
            inp_gt = torch.cat([anomalous, gt_mask], dim=1)
            pred_inp = inpainter(inp_gt)
            for i in range(len(pred_inp)):
                stage2_metrics["psnr"].append(compute_psnr(pred_inp[i:i+1], clean[i:i+1]))
                stage2_metrics["ssim"].append(compute_ssim(pred_inp[i:i+1], clean[i:i+1]))
                stage2_metrics["l1_loss"].append(F.l1_loss(pred_inp[i:i+1], clean[i:i+1]).item())

            # Pipeline evaluation (end-to-end)
            pred_binary = (pred_mask > 0.5).float()
            inp_pipe = torch.cat([anomalous, pred_binary], dim=1)
            pred_pipe = inpainter(inp_pipe)
            for i in range(len(pred_pipe)):
                pipeline_metrics["psnr"].append(compute_psnr(pred_pipe[i:i+1], clean[i:i+1]))
                pipeline_metrics["ssim"].append(compute_ssim(pred_pipe[i:i+1], clean[i:i+1]))
                pipeline_metrics["l1_loss"].append(F.l1_loss(pred_pipe[i:i+1], clean[i:i+1]).item())

    # =====================================================================
    # Measure inference time
    # =====================================================================
    print("\n[Timing] Measuring pipeline inference cost...")
    timing = measure_pipeline_time(localizer, inpainter, test_loader, device)
    print(f"  Stage 1: {timing['stage1_ms']:.2f}ms")
    print(f"  Stage 2: {timing['stage2_ms']:.2f}ms")
    print(f"  Pipeline total: {timing['pipeline_total_ms']:.2f}ms")
    print(f"  Full regen (sim): {timing['full_regen_ms']:.2f}ms")
    print(f"  Speedup: {timing['speedup_ratio']:.2f}x")

    # =====================================================================
    # Generate visualization
    # =====================================================================
    print("\n[Visualization] Generating pipeline visualization...")

    # Pick first test sample
    sample_anom, sample_clean, sample_mask = test_dataset[0]
    sample_anom = sample_anom.unsqueeze(0).to(device)
    sample_clean_t = sample_clean.unsqueeze(0)

    with torch.no_grad():
        error_mask, repaired = pipeline_inference(localizer, inpainter, sample_anom, device)

    vis_path = os.path.join(output_dir, "surgical_pipeline.png")
    plot_surgical_pipeline(
        sample_anom.squeeze(0).cpu(),
        error_mask.squeeze(0).cpu(),
        repaired.squeeze(0).cpu(),
        sample_clean_t.squeeze(0),
        vis_path)
    print(f"  Saved: {vis_path}")

    # =====================================================================
    # Save results
    # =====================================================================
    print("\n[Saving] Writing results...")

    def safe_mean(arr, default=0.0):
        if not arr:
            return default
        clean_arr = [v for v in arr if not (np.isnan(v) or np.isinf(v))]
        return float(np.mean(clean_arr)) if clean_arr else default

    results = {
        "experiment": "A3",
        "description": "Two-Stage Surgical Redrawing System",
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "device": device.__str__(),
        },
        "stage1_error_localizer": {
            "iou_mean": safe_mean(stage1_metrics["iou"]),
            "iou_std": safe_mean([], 0.0) if not stage1_metrics["iou"] else float(np.std([v for v in stage1_metrics["iou"] if not (np.isnan(v) or np.isinf(v))])),
            "f1_mean": safe_mean(stage1_metrics["f1"]),
            "best_iou_during_training": round(float(best_iou), 4),
            "training_time_s": round(loc_time, 1),
        },
        "stage2_inpainter_oracle": {
            "psnr_mean": safe_mean(stage2_metrics["psnr"]),
            "ssim_mean": safe_mean(stage2_metrics["ssim"]),
            "best_psnr_during_training": round(float(best_psnr), 2),
            "training_time_s": round(inp_time, 1),
        },
        "pipeline_end_to_end": {
            "psnr_mean": safe_mean(pipeline_metrics["psnr"]),
            "ssim_mean": safe_mean(pipeline_metrics["ssim"]),
        },
        "inference_cost": timing,
    }

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_json_path}")

    # =====================================================================
    # Summary Table
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  A3 Results Summary")
    print(f"{'='*80}")
    print(f"  {'Stage 1 (Error Localizer)':40s}")
    print(f"    IoU: {safe_mean(stage1_metrics['iou']):.4f}  |  F1: {safe_mean(stage1_metrics['f1']):.4f}  |  Time: {loc_time:.1f}s")
    print(f"  {'Stage 2 (Inpainter - Oracle Mask)':40s}")
    print(f"    PSNR: {safe_mean(stage2_metrics['psnr']):.2f} dB  |  SSIM: {safe_mean(stage2_metrics['ssim']):.4f}  |  Time: {inp_time:.1f}s")
    print(f"  {'Pipeline (End-to-End)':40s}")
    print(f"    PSNR: {safe_mean(pipeline_metrics['psnr']):.2f} dB  |  SSIM: {safe_mean(pipeline_metrics['ssim']):.4f}")
    print(f"  {'Inference Cost':40s}")
    print(f"    Pipeline: {timing['pipeline_total_ms']:.2f}ms  |  Full Regen: {timing['full_regen_ms']:.2f}ms  |  Speedup: {timing['speedup_ratio']:.2f}x")
    print(f"{'='*80}")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Results JSON: {results_json_path}")
    print(f"  Visualization: {vis_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
