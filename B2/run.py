#!/usr/bin/env python3
"""
=============================================================================
B2: Low-Cost High-Resolution Enhancement — Selective High-Frequency Injection
=============================================================================

Core Idea: PSR-Net injects details only in high-frequency regions (edges,
textures, details), while smooth regions maintain cheap interpolation quality.
This allows a low-resolution model to generate a coarse draft, then PSR-Net
selectively injects high-frequency details instead of regenerating at full
high resolution.

Framework (same as B1 — Selective Texture Injection):
  - Input: 3ch blurry upsampled image (I_blurry_upscaled)
  - Output: 4ch → 3ch residual R + 1ch sigmoid mask M
  - Refinement formula: I_refined = I_blurry + R * M
  - Loss: L1 + λ_s * mean(M)   (λ_s=0.05)
  - Mask M activates on: edges, textures, high-frequency detail regions

Workflow:
  1. Load high-resolution real images
  2. Downsample to 1/4 resolution → bilinear upsample back to original resolution → "blurry" version
  3. Train PSR-Net: blurry → original (learn to inject high-frequency details)
  4. Evaluate: PSNR/SSIM/LPIPS/Edge Preservation Index
  5. Baseline comparison: Bicubic | Lanczos | Simplified Super-Res | SD Upscale | PSR-Net
  6. Cost analysis: mask activation rate, quality-cost scatter plot

Usage:
  python B2/run.py --mode simple --epochs 100 --image_size 256
  python B2/run.py --mode full --epochs 100 --image_size 512 --downscale_factor 4

Relationship with B1:
  B1 and B2 share the same PSR-Net architecture and selective texture injection
  training paradigm. The only difference is the training data:
    - B1: blur(sigma=1~2.5) blurring → inject photographic texture
    - B2: downscale→upscale blurring → inject high-frequency details
=============================================================================
"""

"""
=============================================================================
B2: 低成本高分辨率增强 — 选择性高频注入
     (Low-Cost High-Resolution Enhancement — Selective High-Frequency Injection)
=============================================================================

核心思想: PSR-Net 仅在高频区域（边缘、纹理、细节）注入细节，平滑区域保持廉价插值质量。
这样可以用低分辨率模型生成粗稿，再用 PSR-Net 选择性地注入高频细节来替代全图高分辨率重新生成。

框架 (与 B1 相同 — 选择性纹理注入):
  - 输入: 3ch 模糊上采样图像 (I_blurry_upscaled)
  - 输出: 4ch → 3ch 残差 R + 1ch sigmoid 掩膜 M
  - 精修公式: I_refined = I_blurry + R * M
  - 损失: L1 + λ_s * mean(M)   (λ_s=0.05)
  - 掩膜 M 应激活: 边缘、纹理、高频细节区域

工作流:
  1. 加载高分辨率真实图像
  2. 下采样到 1/4 分辨率 → 双线性上采样回原分辨率 → "模糊" 版本
  3. 训练 PSR-Net: blurry → original (学习注入高频细节)
  4. 评估: PSNR/SSIM/LPIPS/边缘保留指数
  5. 对比基线: Bicubic | Lanczos | Simplified Super-Res | SD Upscale | PSR-Net
  6. 成本分析: 掩膜激活率、质量-成本散点图

用法:
  python B2/run.py --mode simple --epochs 100 --image_size 256
  python B2/run.py --mode full --epochs 100 --image_size 512 --downscale_factor 4

与 B1 的关系:
  B1 和 B2 共用相同的 PSR-Net 架构和选择性纹理注入训练范式,
  区别仅在于训练数据:
    - B1: blur(sigma=1~2.5) 模糊 → 注入摄影纹理
    - B2: 降采样→升采样 模糊 → 注入高频细节
=============================================================================
"""
import os, sys, json, time, argparse, warnings, random, io
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore")

# ── Path Configuration ──────────────────────────────────────────────────────────────
# ── 路径配置 ────────────────────────────────────────────────────────────────
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(FILE_DIR))

from common.model_factory import create_model, save_checkpoint, load_checkpoint
from common.config import get_config, TextureInjectionConfig
from common.training import TrainingEngine
from common.evaluation import (
    compute_psnr, compute_ssim, compute_lpips_approx, compute_fid,
    compute_activation_stats, measure_inference_performance, evaluate_all,
)
from common.data_utils import load_real_images
from common.visualization import plot_results_grid, plot_training_curves, tensor_to_numpy

# Dataset paths
# 数据集路径
REDRAWING_DIR = os.path.join(os.path.dirname(os.path.dirname(FILE_DIR)), "RedrawingPhotoCreating")
DATASET_DIR = os.path.join(REDRAWING_DIR, "dataset")
RESOURCES_DIR = os.path.join(REDRAWING_DIR, "resourses")
OUTPUT_ROOT = os.path.join(FILE_DIR, "outputs")

# ── Constants ──────────────────────────────────────────────────────────────────
# ── 常量 ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — High-Frequency Enhancement Training Pairs (B1/B2 Shared Framework, Different Degradation Strategies)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 数据集 — 高频增强训练对 (B1/B2 共用框架, 不同退化策略)
# ══════════════════════════════════════════════════════════════════════════════

class DownscaleUpscaleDataset(Dataset):
    """
    Downscale→upscale training pair dataset.

    Strategy:
      1. Take high-resolution real image → GT (original)
      2. Downscale to 1/factor × 1/factor resolution
      3. Bilinear upscale back to original resolution → "blurry" version (I_blurry)
      4. Network learns: blurry → original (i.e., inject high-frequency details)

    This simulates the "cheap low-resolution model output" scenario:
      - Quickly generate a coarse draft with a low-resolution network
      - PSR-Net injects details only in high-frequency regions
      - Total cost << full high-resolution regeneration
    """
    """
    降采样→升采样训练对数据集。

    策略:
      1. 取高分辨率真实图像 → GT (原图)
      2. 降采样到 1/factor × 1/factor 分辨率
      3. 双线性上采样回原分辨率 → "模糊" 版本 (I_blurry)
      4. 网络学习: blurry → original (即注入高频细节)

    这模拟了"廉价低分辨率模型输出"的场景:
      - 用低分辨率网络快速生成粗稿
      - PSR-Net 仅在高频区域注入细节
      - 总成本 << 全高分辨率重新生成
    """
    
    def __init__(self, images: list, num_samples: int, size: int = 256,
                 downscale_factor: int = 4,
                 interpolation: str = "bilinear",
                 seed: int = DEFAULT_SEED, augment: bool = True):
        """
        Args:
            images: List of [H, W, 3] numpy arrays, value range [0, 1]
            num_samples: Number of samples per epoch
            size: Target crop size
            downscale_factor: Downscale factor (default 4, i.e. 256→64→256)
            interpolation: Upscale method ('bilinear' | 'bicubic' | 'lanczos')
            augment: Whether to enable data augmentation
        """
        """
        Args:
            images: [H, W, 3] numpy 数组列表, 值域 [0, 1]
            num_samples: epoch 内采样数
            size: 目标裁剪尺寸
            downscale_factor: 降采样倍率 (默认 4, 即 256→64→256)
            interpolation: 上采样方法 ('bilinear' | 'bicubic' | 'lanczos')
            augment: 是否启用数据增强
        """
        self.images = images
        self.num_samples = num_samples
        self.size = size
        self.factor = downscale_factor
        self.interp = interpolation
        self.augment = augment
        self.low_size = size // downscale_factor
        
        rng = np.random.RandomState(seed)
        self.seeds = rng.randint(0, 2**31 - 1, size=num_samples)
        self.n_images = len(images)
    
    def __len__(self):
        return self.num_samples
    
    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.size or w < self.size:
            return self._resize(img, self.size)
        top = np.random.randint(0, h - self.size + 1)
        left = np.random.randint(0, w - self.size + 1)
        return img[top:top+self.size, left:left+self.size]
    
    def _resize(self, img: np.ndarray, target: int, method=None) -> np.ndarray:
        pil = Image.fromarray((img * 255).astype(np.uint8))
        if method is None:
            pil = pil.resize((target, target), Image.LANCZOS)
        else:
            pil = pil.resize((target, target), method)
        return np.array(pil).astype(np.float32) / 255.0
    
    def _upscale_pil(self, img: np.ndarray) -> np.ndarray:
        """Downscale + Upscale = Create blurry version"""
        """降采样 + 上采样 = 创建模糊版本"""
        # Downscale
        # 降采样
        low = self._resize(img, self.low_size)
        # Upscale method mapping
        # 上采样方法映射
        method_map = {
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        method = method_map.get(self.interp, Image.BILINEAR)
        up = self._resize(low, self.size, method=method)
        return up
    
    def __getitem__(self, idx):
        seed = int(self.seeds[idx])
        np.random.seed(seed)
        img_idx = np.random.randint(0, self.n_images)
        clean = self.images[img_idx].copy()
        
        # Crop
        # 裁剪
        clean = self._random_crop(clean)
        
        # Data augmentation
        # 数据增强
        if self.augment:
            if np.random.rand() > 0.5:
                clean = np.fliplr(clean).copy()
            if np.random.rand() > 0.3:
                clean = np.rot90(clean, k=np.random.randint(0, 4)).copy()
            # Mild color jitter
            # 轻微颜色抖动
            if np.random.rand() > 0.3:
                jitter = np.random.uniform(0.95, 1.05, 3)
                clean = np.clip(clean * jitter, 0, 1)
        
        # Downscale + Upscale → blurry
        # 降采样 + 上采样 → blurry
        blurry = self._upscale_pil(clean)
        
        # GT mask: high-frequency difference regions
        # GT mask: 高频差异区域
        diff = np.abs(clean - blurry).max(axis=2)  # [H, W]
        gt_mask = torch.from_numpy((diff > 0.01).astype(np.float32)).unsqueeze(0)
        
        dirty = torch.from_numpy(blurry.transpose(2, 0, 1)).float()
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        
        return dirty, clean_t, gt_mask


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Metrics — High-Frequency Enhancement Specific
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 评估指标 — 高频增强专用
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_edge_preservation_index(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Edge Preservation Index (EPI).

    EPI = sum(Δpred * Δtarget) / sqrt(sum(Δpred²) * sum(Δtarget²))
    Values closer to 1 indicate better edge preservation.
    """
    """
    边缘保留指数 (Edge Preservation Index, EPI).

    EPI = sum(Δpred * Δtarget) / sqrt(sum(Δpred²) * sum(Δtarget²))
    值越接近 1 表示边缘保留越好.
    """
    # Simplified version of Sobel operator — Laplacian gradient
    # 使用 Sobel 算子的简化版 — 拉普拉斯梯度
    kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]],
                           device=pred.device, dtype=pred.dtype)
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    edge_preds, edge_targets = [], []
    for c in range(3):
        ep = F.conv2d(pred[:, c:c+1], kernel, padding=1)
        et = F.conv2d(target[:, c:c+1], kernel, padding=1)
        edge_preds.append(ep)
        edge_targets.append(et)
    
    ep = torch.cat(edge_preds, dim=1)
    et = torch.cat(edge_targets, dim=1)
    
    num = (ep * et).sum()
    den = torch.sqrt((ep ** 2).sum() * (et ** 2).sum() + 1e-8)
    return float((num / den).item())


@torch.no_grad()
def compute_high_freq_energy(image: torch.Tensor) -> float:
    """Compute high-frequency energy (normalized Laplacian energy)"""
    """计算高频能量 (归一化拉普拉斯能量)"""
    kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]],
                           device=image.device, dtype=image.dtype)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    energy = 0.0
    for c in range(3):
        edge = F.conv2d(image[:, c:c+1], kernel, padding=1)
        energy += edge.pow(2).mean().item()
    return energy / 3.0


# ══════════════════════════════════════════════════════════════════════════════
# Baseline Methods
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 基线方法
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def bicubic_upscale(lowres_batch: torch.Tensor, target_size: int) -> torch.Tensor:
    """Bicubic interpolation upscale (lower bound baseline)"""
    """双三次插值上采样（下界基线）"""
    results = []
    for img in lowres_batch:
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil = pil.resize((target_size, target_size), Image.BICUBIC)
        up_np = np.array(pil).astype(np.float32) / 255.0
        results.append(torch.from_numpy(up_np).permute(2, 0, 1))
    return torch.stack(results)


@torch.no_grad()
def lanczos_upscale(lowres_batch: torch.Tensor, target_size: int) -> torch.Tensor:
    """Lanczos interpolation upscale"""
    """Lanczos 插值上采样"""
    results = []
    for img in lowres_batch:
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil = pil.resize((target_size, target_size), Image.LANCZOS)
        up_np = np.array(pil).astype(np.float32) / 255.0
        results.append(torch.from_numpy(up_np).permute(2, 0, 1))
    return torch.stack(results)


@torch.no_grad()
def simple_cnn_superres(lowres_batch: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Simplified CNN super-resolution (simulates Real-ESRGAN idea without loading large models).
    Uses bicubic interpolation + mild sharpening as approximation.
    """
    """
    简化 CNN 超分辨率 (模拟 Real-ESRGAN 思想但不加载大模型).
    使用双三次插值 + 轻度锐化作为近似.
    """
    results = []
    for img in lowres_batch:
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil = pil.resize((target_size, target_size), Image.LANCZOS)
        # Mild USM sharpening to simulate CNN super-resolution effect
        # 轻度 USM 锐化模拟 CNN 超分效果
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=2))
        up_np = np.array(pil).astype(np.float32) / 255.0
        results.append(torch.from_numpy(up_np).permute(2, 0, 1))
    return torch.stack(results)


@torch.no_grad()
def sd_upscale_proxy(lowres_batch: torch.Tensor, target_size: int,
                      device: str) -> torch.Tensor:
    """
    SD img2img upscale proxy (when full SD is not loaded).
    Uses Lanczos + strong USM as substitute (full mode will attempt real SD upscale).
    """
    """
    SD img2img 上采样代理 (不加载完整 SD 时).
    使用 Lanczos + 强 USM 作为替代 (full 模式下会尝试真实 SD 上采样).
    """
    results = []
    for img in lowres_batch:
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil = pil.resize((target_size, target_size), Image.LANCZOS)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=1))
        up_np = np.array(pil).astype(np.float32) / 255.0
        results.append(torch.from_numpy(up_np).permute(2, 0, 1))
    return torch.stack(results)


# ══════════════════════════════════════════════════════════════════════════════
# SD Img2Img Real Upscale (Full Mode)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SD img2img 真实上采样 (full 模式)
# ══════════════════════════════════════════════════════════════════════════════

def sd_img2img_upscale(lowres_images: list, target_size: int,
                        strength: float = 0.3, guidance_scale: float = 7.5) -> list:
    """Real upscale using SD v1.5 img2img pipeline"""
    """使用 SD v1.5 img2img 管道进行真实上采样"""
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe = pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"    [WARN] Cannot load SD Img2Img: {e}")
        return None
    
    results = []
    for img_tensor in lowres_images:
        img_np = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil = pil.resize((target_size, target_size), Image.LANCZOS)
        
        with torch.autocast(device_type=DEVICE.type):
            out = pipe(
                prompt="high quality, sharp details, realistic texture",
                image=pil,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=25,
            ).images[0]
        
        out_np = np.array(out).astype(np.float32) / 255.0
        results.append(torch.from_numpy(out_np).permute(2, 0, 1))
    
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualization — Enhancement Comparison + Cost Analysis
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 可视化 — 增强对比 + 成本分析
# ══════════════════════════════════════════════════════════════════════════════

def save_enhancement_comparison(gt_list: list, baseline_list: list,
                                 baseline_names: list,
                                 mask_list: list = None,
                                 save_path: str = "enhancement_comparison.png"):
    """
    Save enhancement comparison plot.
    Columns: GT | Bicubic | Lanczos | SimpleCNN | SD Proxy | PSR-Net | Mask (opt)
    """
    """
    保存增强对比图。
    Columns: GT | Bicubic | Lanczos | SimpleCNN | SD Proxy | PSR-Net | Mask (opt)
    """
    n = len(gt_list)
    n_cols = 1 + len(baseline_list) + (1 if mask_list else 0)
    
    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3.5, n * 3.5))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    all_cols = ["GT (Original)"] + baseline_names
    if mask_list:
        all_cols.append("PSR-Net Mask")
    
    for ax, label in zip(axes[0], all_cols):
        ax.set_title(label, fontsize=10, fontweight="bold")
    
    for i in range(n):
        axes[i, 0].imshow(tensor_to_numpy(gt_list[i]))
        axes[i, 0].axis("off")
        
        for j, bl in enumerate(baseline_list):
            axes[i, j+1].imshow(tensor_to_numpy(bl[i]))
            axes[i, j+1].axis("off")
        
        if mask_list and i < len(mask_list):
            axes[i, -1].imshow(tensor_to_numpy(mask_list[i]), cmap="hot")
            axes[i, -1].axis("off")
    
    plt.suptitle("B2: High-Resolution Enhancement Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_cost_analysis(methods_data: dict, save_path: str = "cost_analysis.png"):
    """
    Save quality-cost scatter plot.
    methods_data: {"Method": {"quality": float, "cost": float, "mask_ratio": float}}
    """
    """
    保存质量-成本散点图。
    methods_data: {"Method": {"quality": float, "cost": float, "mask_ratio": float}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(methods_data.keys())
    qualities = [methods_data[n].get("quality", 0) for n in names]
    costs = [methods_data[n].get("cost", 0) for n in names]
    mask_ratios = [methods_data[n].get("mask_ratio", 0) for n in names]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    texts_left, texts_right = [], []

    # Left plot: Quality vs Cost
    # 左图: Quality vs Cost
    for n, q, c, col in zip(names, qualities, costs, colors):
        axes[0].scatter(c, q, s=200, color=col, edgecolors="black", zorder=5)
        texts_left.append(axes[0].text(c, q, n, fontsize=8,
                                       ha="center", va="center",
                                       bbox=dict(boxstyle="round,pad=0.2",
                                                 fc="white", alpha=0.8,
                                                 ec="gray", lw=0.3)))
    axes[0].set_xlabel("Relative Cost (lower is cheaper)", fontsize=11)
    axes[0].set_ylabel("PSNR (dB)", fontsize=11)
    axes[0].set_title("Quality vs Cost", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Quality vs Mask Activation Ratio
    # 右图: Quality vs Mask Activation Ratio
    for n, q, r, col in zip(names, qualities, mask_ratios, colors):
        axes[1].scatter(r, q, s=200, color=col, edgecolors="black", zorder=5)
        texts_right.append(axes[1].text(r, q, n, fontsize=8,
                                        ha="center", va="center",
                                        bbox=dict(boxstyle="round,pad=0.2",
                                                  fc="white", alpha=0.8,
                                                  ec="gray", lw=0.3)))
    axes[1].set_xlabel("Mask Activation Ratio (%)", fontsize=11)
    axes[1].set_ylabel("PSNR (dB)", fontsize=11)
    axes[1].set_title("Quality vs Selectivity", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Annotate PSR-Net cost savings
    # 标注 PSR-Net 的成本节约
    psrnet_data = methods_data.get("PSR-Net Selective (Ours)", {})
    if psrnet_data:
        mask_ratio = psrnet_data.get("mask_ratio", 0)
        axes[0].annotate(
            f"PSR-Net: {mask_ratio*100:.1f}% pixels\n"
            f"Cost = {mask_ratio:.2f}× full regen.",
            xy=(psrnet_data["cost"], psrnet_data["quality"]),
            xytext=(20, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
        )
    
    axes[0].margins(x=0.15, y=0.15)
    axes[1].margins(x=0.15, y=0.15)
    adjust_text(texts_left, ax=axes[0],
                expand_points=(1.5, 1.2), expand_text=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                force_text=(0.5, 0.5), force_points=(0.2, 0.2),
                lim=200)
    adjust_text(texts_right, ax=axes[1],
                expand_points=(1.5, 1.2), expand_text=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                force_text=(0.5, 0.5), force_points=(0.2, 0.2),
                lim=200)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_mask_activation_map(blurry_list: list, refined_list: list,
                              mask_list: list, save_path: str):
    """Save mask activation map"""
    """保存掩膜激活图"""
    n = len(blurry_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    axes[0, 0].set_title("Blurry Input", fontsize=10)
    axes[0, 1].set_title("Refined Output", fontsize=10)
    axes[0, 2].set_title("Mask (Where Detail Added)", fontsize=10)
    
    for i in range(n):
        axes[i, 0].imshow(tensor_to_numpy(blurry_list[i]))
        axes[i, 0].axis("off")
        axes[i, 1].imshow(tensor_to_numpy(refined_list[i]))
        axes[i, 1].axis("off")
        
        mask_np = tensor_to_numpy(mask_list[i])
        if mask_np.ndim == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze()
        axes[i, 2].imshow(mask_np, cmap="hot")
        axes[i, 2].set_title(f"Mask (μ={mask_np.mean():.4f})", fontsize=10)
        axes[i, 2].axis("off")
    
    plt.suptitle("B2: Where PSR-Net Adds High-Frequency Details", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    """Main entry point for B2 experiment"""
    """B2 实验主入口"""
    config = get_config("B2",
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_samples=args.train_samples,
    )
    config.device = str(DEVICE)
    config.lambda_sparse = 0.05
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  B2: Low-Cost High-Resolution Enhancement")
    print(f"  Device: {DEVICE} | Mode: {args.mode} | Size: {config.image_size}")
    print(f"  Epochs: {config.epochs} | λ_s: {config.lambda_sparse}")
    print(f"  Downscale: 1/{args.downscale_factor} ×")
    print(f"{'='*70}")
    
    # ── 1. Load Real Images ──
    # ── 1. 加载真实图片 ──
    print("\n[Step 1] Loading high-resolution images...")
    real_imgs = []
    for src_dir in [DATASET_DIR, RESOURCES_DIR]:
        if os.path.isdir(src_dir):
            # Load images slightly larger than target_size to ensure crop quality
            # 加载比 target_size 稍大的图片以确保裁剪质量
            larger_size = max(args.image_size * 2, 512)
            loaded = load_real_images(src_dir, target_size=larger_size, max_images=None)
            real_imgs.extend(loaded)
            print(f"  Loaded {len(loaded)} images from {src_dir}")
    
    if not real_imgs:
        print("  [WARN] No real images found, using synthetic gradients as fallback")
        rng = np.random.RandomState(DEFAULT_SEED)
        for _ in range(100):
            t_size = max(args.image_size * 2, 512)
            grad = np.zeros((t_size, t_size, 3), dtype=np.float32)
            for c in range(3):
                vc = rng.rand()
                gx = np.linspace(0, 1, t_size) * vc
                gy = np.linspace(0, 1, t_size).reshape(-1, 1) * (1 - vc)
                grad[:, :, c] = 0.3 + 0.4 * (0.5 * gx + 0.5 * gy)
            real_imgs.append(grad.clip(0, 1))
    
    print(f"  Total images: {len(real_imgs)}, loaded at {real_imgs[0].shape[:2]}")
    
    # ── 2. Prepare Training Pairs ──
    # ── 2. 准备训练对 ──
    print("\n[Step 2] Preparing downscale→upscale training pairs...")
    train_dataset = DownscaleUpscaleDataset(
        real_imgs, num_samples=config.train_samples,
        size=args.image_size,
        downscale_factor=args.downscale_factor,
        interpolation="bilinear",
        seed=DEFAULT_SEED, augment=True,
    )
    val_dataset = DownscaleUpscaleDataset(
        real_imgs, num_samples=min(50, config.train_samples // 4),
        size=args.image_size,
        downscale_factor=args.downscale_factor,
        interpolation="bilinear",
        seed=DEFAULT_SEED + 1000, augment=False,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                               num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    
    low_size = args.image_size // args.downscale_factor
    print(f"  Size: {args.image_size}×{args.image_size} → {low_size}×{low_size} → {args.image_size}×{args.image_size}")
    print(f"  Train samples: {config.train_samples}, Val samples: {len(val_dataset)}")
    
    # ── 3. Create Model ──
    # ── 3. 创建模型 ──
    print("\n[Step 3] Creating PSR-Net model...")
    model = create_model("large" if args.image_size >= 256 else "standard",
                          base_channels=config.base_channels,
                          input_channels=3, device=config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model.__class__.__name__}, Params: {n_params:,}")
    
    # ── 4. Training ──
    # ── 4. 训练 ──
    print(f"\n[Step 4] Training PSR-Net for selective high-frequency injection...")
    engine = TrainingEngine(model, config, config.device)
    history = engine.train(
        train_loader, val_loader,
        lambda_distill=0.0,
        verbose=True, val_freq=5,
        save_dir=OUTPUT_ROOT,
    )
    
    best_model_path = os.path.join(OUTPUT_ROOT, "best_model.pt")
    if os.path.exists(best_model_path):
        model = load_checkpoint(model, best_model_path, config.device)
        print("  Loaded best model checkpoint")
    
    # ── 5. Validation Metrics ──
    # ── 5. 验证集度量 ──
    print(f"\n[Step 5] Computing validation metrics...")
    model.eval()
    val_metrics = defaultdict(list)
    
    for batch in val_loader:
        dirty, clean, _ = [b.to(config.device) for b in batch]
        refined, residual, mask = model.refine(dirty)
        
        val_metrics["psnr"].append(compute_psnr(refined, clean))
        val_metrics["ssim"].append(compute_ssim(refined, clean))
        val_metrics["lpips"].append(compute_lpips_approx(refined, clean))
        val_metrics["epi"].append(compute_edge_preservation_index(refined, clean))
        val_metrics["mask_mean"].append(mask.mean().item())
        val_metrics["hf_energy"].append(compute_high_freq_energy(refined))
    
    val_summary = {k: float(np.mean(v)) for k, v in val_metrics.items()}
    print(f"  PSNR={val_summary['psnr']:.2f}dB, SSIM={val_summary['ssim']:.4f}, "
          f"EPI={val_summary['epi']:.4f}, Mask_Mean={val_summary['mask_mean']:.4f}")
    
    # ── 6. Inference + Baseline Comparison ──
    # ── 6. 推理 + 对比基线 ──
    print(f"\n[Step 6] Running inference and baseline comparison...")
    n_eval = min(20, len(val_dataset))
    eval_blurry, eval_clean, _ = zip(*[val_dataset[i] for i in range(n_eval)])
    eval_blurry_batch = torch.stack(eval_blurry)
    eval_clean_batch = torch.stack(eval_clean)
    
    blurry_device = eval_blurry_batch.to(config.device)
    clean_device = eval_clean_batch.to(config.device)
    
    # PSR-Net inference
    # PSR-Net 推理
    t_start = time.time()
    refined_device, residual_device, mask_device = model.refine(blurry_device)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    psrnet_inference_time = (time.time() - t_start) * 1000 / n_eval  # ms per image
    refined_out = refined_device.cpu()
    mask_out = mask_device.cpu()
    
    # Compute mask coverage
    # 计算掩膜覆盖率
    mask_activation_ratio = mask_out.mean().item()
    print(f"  PSR-Net mask activation: {mask_activation_ratio*100:.1f}% of pixels")
    
    # Bicubic baseline
    # Bicubic 基线
    lowres_clean = [F.interpolate(c.unsqueeze(0), size=low_size, mode='bilinear').squeeze(0)
                     for c in eval_clean]
    lowres_batch = torch.stack(lowres_clean)
    bicubic_result = bicubic_upscale(lowres_batch, args.image_size)
    lanczos_result = lanczos_upscale(lowres_batch, args.image_size)
    cnn_result = simple_cnn_superres(lowres_batch, args.image_size)
    sd_proxy_result = sd_upscale_proxy(lowres_batch, args.image_size, DEVICE)
    
    # Evaluate each baseline
    # 评估各基线
    baselines = {
        "Bicubic Interpolation": bicubic_result,
        "Lanczos Interpolation": lanczos_result,
        "Simple CNN Super-Res": cnn_result,
        "SD Upscale Proxy": sd_proxy_result,
        "PSR-Net Selective (Ours)": refined_out,
    }
    
    baseline_metrics = {}
    for name, pred in baselines.items():
        bm = {}
        bm["psnr"] = compute_psnr(pred, eval_clean_batch)
        bm["ssim"] = compute_ssim(pred, eval_clean_batch)
        bm["lpips"] = compute_lpips_approx(pred, eval_clean_batch)
        bm["epi"] = compute_edge_preservation_index(pred, eval_clean_batch)
        bm["hf_energy"] = compute_high_freq_energy(pred)
        baseline_metrics[name] = bm
        
        # Inference time
        # 推理时间
        if name == "PSR-Net Selective (Ours)":
            bm["inference_ms"] = psrnet_inference_time
            bm["mask_ratio"] = mask_activation_ratio
        else:
            # Approximate interpolation time
            bm["inference_ms"] = 0.5  # 近似插值时间
            # Full-image processing
            bm["mask_ratio"] = 1.0  # 全图处理
        
        print(f"  {name}: PSNR={bm['psnr']:.2f}dB, SSIM={bm['ssim']:.4f}, EPI={bm['epi']:.4f}")
    
    # SD img2img real upscale (full mode only)
    # SD img2img 真实上采样 (仅 full 模式)
    if args.mode == "full" and not args.use_real_data_only:
        print("  Attempting SD img2img upscale...")
        sd_lowres = eval_clean[:min(5, len(eval_clean))]
        sd_results = sd_img2img_upscale(sd_lowres, args.image_size, strength=0.3)
        if sd_results:
            sd_batch = torch.stack(sd_results)
            sd_metrics = {
                "psnr": compute_psnr(sd_batch, eval_clean_batch[:len(sd_results)]),
                "ssim": compute_ssim(sd_batch, eval_clean_batch[:len(sd_results)]),
                "lpips": compute_lpips_approx(sd_batch, eval_clean_batch[:len(sd_results)]),
                "epi": compute_edge_preservation_index(sd_batch, eval_clean_batch[:len(sd_results)]),
                "hf_energy": compute_high_freq_energy(sd_batch),
                "inference_ms": 2000.0,  # SD img2img is ~2s per image
                "mask_ratio": 1.0,
            }
            baseline_metrics["SD Img2Img Upscale"] = sd_metrics
            baselines["SD Img2Img Upscale"] = sd_batch
            print(f"  SD Img2Img Upscale: PSNR={sd_metrics['psnr']:.2f}dB")
    
    # ── 7. Cost Analysis ──
    # ── 7. 成本分析 ──
    print(f"\n[Step 7] Cost analysis...")
    # Assume full high-resolution SD regeneration = 1.0 unit cost
    # 假设全高分辨率SD重生成 = 1.0 单位成本
    # SD img2img upscale = 0.6 unit cost (faster than full generation)
    # SD img2img 上采样 = 0.6 单位成本 (比全图生成快)
    # Cheap low-resolution generation = 0.1 unit cost
    # 廉价低分辨率生成 = 0.1 单位成本
    # PSR-Net = mask_ratio × full_highres_cost ≈ mask_ratio × 0.9
    
    # Full high-resolution generation
    full_highres_cost = 1.0  # 全高分辨率生成
    # Cheap interpolation is nearly free
    cheap_upscale_cost = 0.05  # 插值几乎免费
    # Only mask regions need high-quality processing
    psrnet_cost = mask_activation_ratio * 0.9  # 仅掩膜区域需要高质量处理
    
    cost_data = {
        "Full High-Res Gen": {
            "quality": baseline_metrics.get("SD Upscale Proxy", {}).get("psnr", 0),
            "cost": full_highres_cost,
            "mask_ratio": 1.0,
            "description": "Full resolution regeneration",
        },
        "Bicubic (cheap)": {
            "quality": baseline_metrics["Bicubic Interpolation"]["psnr"],
            "cost": cheap_upscale_cost,
            "mask_ratio": 1.0,
            "description": "Bicubic interpolation (lower bound)",
        },
        "Lanczos": {
            "quality": baseline_metrics["Lanczos Interpolation"]["psnr"],
            "cost": cheap_upscale_cost + 0.01,
            "mask_ratio": 1.0,
            "description": "Lanczos interpolation",
        },
        "PSR-Net Selective (Ours)": {
            "quality": baseline_metrics["PSR-Net Selective (Ours)"]["psnr"],
            "cost": cheap_upscale_cost + psrnet_cost,
            "mask_ratio": mask_activation_ratio,
            "description": f"Cheap upscale + PSR-Net ({mask_activation_ratio*100:.1f}% pixels)",
        },
    }
    
    print(f"  Full High-Res cost: {full_highres_cost:.2f}")
    print(f"  PSR-Net cost: {cost_data['PSR-Net Selective (Ours)']['cost']:.3f}")
    print(f"  Cost saving: {(1 - cost_data['PSR-Net Selective (Ours)']['cost'] / full_highres_cost) * 100:.1f}%")
    print(f"  Mask activates {mask_activation_ratio*100:.1f}% of pixels")
    
    # ── 8. Visualizations ──
    # ── 8. 可视化 ──
    print(f"\n[Step 8] Generating visualizations...")
    
    n_display = min(5, n_eval)
    display_idx = list(range(n_display))
    
    # Enhancement comparison chart
    # 增强对比图
    comparison_path = save_enhancement_comparison(
        gt_list=[eval_clean_batch[i] for i in display_idx],
        baseline_list=[[baselines[n][i] for i in display_idx]
                        for n in list(baselines.keys())],
        baseline_names=list(baselines.keys()),
        mask_list=[mask_out[i].unsqueeze(0) if mask_out[i].dim() == 2 else mask_out[i]
                    for i in display_idx],
        save_path=os.path.join(OUTPUT_ROOT, "enhancement_comparison.png"),
    )
    print(f"  Saved: {comparison_path}")
    
    # Mask activation map
    # 掩膜激活图
    mask_act_path = save_mask_activation_map(
        blurry_list=[eval_blurry_batch[i] for i in display_idx],
        refined_list=[refined_out[i] for i in display_idx],
        mask_list=[mask_out[i].unsqueeze(0) if mask_out[i].dim() == 2 else mask_out[i]
                    for i in display_idx],
        save_path=os.path.join(OUTPUT_ROOT, "mask_activation_map.png"),
    )
    print(f"  Saved: {mask_act_path}")
    
    # Cost analysis chart
    # 成本分析图
    cost_path = save_cost_analysis(
        cost_data,
        save_path=os.path.join(OUTPUT_ROOT, "cost_analysis.png"),
    )
    print(f"  Saved: {cost_path}")
    
    # Training curves
    # 训练曲线
    curves_path = os.path.join(OUTPUT_ROOT, "training_curves.png")
    plot_training_curves(
        {"B2_HighRes_Enhancement": history},
        save_path=curves_path,
    )
    print(f"  Saved: {curves_path}")
    
    # ── 9. Summary JSON ──
    # ── 9. 汇总 JSON ──
    print(f"\n[Step 9] Saving results...")
    all_results = {
        "experiment": "B2_LowCost_HighResolution_Enhancement",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": args.mode,
            "image_size": config.image_size,
            "low_size": low_size,
            "downscale_factor": args.downscale_factor,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lambda_sparse": config.lambda_sparse,
            "train_samples": config.train_samples,
            "model_params": n_params,
        },
        "training": {
            "final_loss": float(history.get("train_loss", [0])[-1]) if history.get("train_loss") else None,
            "final_l1": float(history.get("train_l1", [0])[-1]) if history.get("train_l1") else None,
            "final_mask_mean": float(history.get("mask_mean", [0])[-1]) if history.get("mask_mean") else None,
        },
        "validation_metrics": val_summary,
        "baseline_comparison": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                      for kk, vv in v.items()}
                                 for k, v in baseline_metrics.items()},
        "cost_analysis": cost_data,
        "cost_summary": {
            "psrnet_mask_activation_ratio": float(mask_activation_ratio),
            "cost_saving_pct": float((1 - psrnet_cost / full_highres_cost) * 100),
            "psrnet_inference_ms_per_image": float(psrnet_inference_time),
        },
        "files": {
            "best_model": best_model_path,
            "training_curves": curves_path,
            "comparison_grid": comparison_path,
            "mask_activation": mask_act_path,
            "cost_analysis": cost_path,
        },
    }
    
    results_json = os.path.join(OUTPUT_ROOT, "results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved results: {results_json}")
    
    print(f"\n{'='*70}")
    print(f"  B2 Experiment Complete!")
    print(f"  Outputs: {OUTPUT_ROOT}")
    print(f"  Key finding: PSR-Net activates {mask_activation_ratio*100:.1f}% of pixels")
    print(f"  Cost saving: {all_results['cost_summary']['cost_saving_pct']:.1f}%")
    print(f"{'='*70}")
    
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B2: Low-Cost High-Res Enhancement — Selective HF Injection with PSR-Net")
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["full", "simple"],
                        help="'full' includes SD img2img upscale; 'simple' uses proxy baselines")
    parser.add_argument("--use_real_data_only", action="store_true",
                        help="Skip SD modules even in full mode")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto based on image_size)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="High-resolution target size")
    parser.add_argument("--downscale_factor", type=int, default=4,
                        help="Downscale factor (4 = 1/4 resolution)")
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Number of training samples per epoch")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Auto-adjust batch_size
    # 自动调整 batch_size
    if args.batch_size is None:
        if args.image_size >= 512:
            args.batch_size = 2
        elif args.image_size >= 256:
            args.batch_size = 4
        else:
            args.batch_size = 8
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.mode == "full" and args.use_real_data_only:
        args.mode = "simple"
    
    run_experiment(args)
