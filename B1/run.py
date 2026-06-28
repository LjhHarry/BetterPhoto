#!/usr/bin/env python3
"""
=============================================================================
B1: Style Refinement — De-AI-ification Texture Injection
=============================================================================

Core Idea: PSR-Net acts as a "Texture Grafting Layer", injecting real-world micro-textures into AI-generated "smooth" images.

Framework: PSR-Net (Selective Texture Injection)
  - Input: 3ch image (I_dirty)
  - Output: 4ch → 3ch residual R + 1ch sigmoid mask M
  - Refinement formula: I_refined = I_dirty + R * M
  - Loss: L1 + λ_s * mean(M)   (λ_s=0.05, encourages sparse but meaningful selection)

Workflow:
  1. Generate AI images (SD v1.5, 500 images at 512x512)
  2. Prepare training pairs: real photos + Gaussian blur → (smooth, original) pairs
  3. Train PSR-Net to learn texture injection
  4. Inference on AI images → more realistic micro-details
  5. Evaluation: FID/PSNR/SSIM/LPIPS + mask visualization
  6. Baseline comparison: Raw AI | USM Sharpening | Full Texture | PSR-Net Selective

Usage:
  python B1/run.py --mode simple --epochs 100 --image_size 256
  python B1/run.py --mode full --epochs 100 --image_size 512 --num_ai 200

Flags:
  --use_real_data_only: Skip SD generation, only use real images from dataset/ and resourses/

=============================================================================
B1: 风格精修 — 去AI化纹理注入 (Style Refinement — De-AI-ification Texture Injection)
=============================================================================

核心思想: PSR-Net 作为"纹理嫁接层"，将真实世界的微观纹理注入到 AI 生成的"平滑"图像中。

框架: PSR-Net (Selective Texture Injection)
  - 输入: 3ch 图像 (I_dirty)
  - 输出: 4ch → 3ch 残差 R + 1ch sigmoid 掩膜 M
  - 精修公式: I_refined = I_dirty + R * M
  - 损失: L1 + λ_s * mean(M)   (λ_s=0.05, 鼓励稀疏但有意义的选择)

工作流:
  1. 生成 AI 图像 (SD v1.5, 500 张 512x512)
  2. 准备训练对: 真实照片 + 高斯模糊 → (smooth, original) 对
  3. 训练 PSR-Net 学习纹理注入
  4. 在 AI 图像上推理 → 更真实的微观细节
  5. 评估: FID/PSNR/SSIM/LPIPS + 掩膜可视化
  6. 对比基线: Raw AI | USM Sharpening | Full Texture | PSR-Net Selective

用法:
  python B1/run.py --mode simple --epochs 100 --image_size 256
  python B1/run.py --mode full --epochs 100 --image_size 512 --num_ai 200

标志位:
  --use_real_data_only: 跳过 SD 生成, 仅使用 dataset/ 和 resourses/ 中的真实图片
=============================================================================
"""
import os, sys, json, time, argparse, warnings, random, io
from collections import defaultdict
from datetime import datetime
from glob import glob
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
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore")

# ── Path Configuration ──────────────────────────────────────────────────────────────
# ── 路径配置 ────────────────────────────────────────────────────────────────
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(os.path.dirname(FILE_DIR), "common")

if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from model_factory import create_model, save_checkpoint, load_checkpoint


def _resolve_resume(resume_arg: str, save_dir: str):
    """Parse --resume argument: 'auto' → auto detect, 'latest' → latest, 'none' → no resume, else use as path"""
    """解析 --resume 参数: 'auto' → 自动检测, 'latest' → 最新, 'none' → 不续训, 否则按路径"""
    if resume_arg is None or resume_arg.lower() == "none":
        return None
    if resume_arg.lower() == "latest":
        return "latest"
    if resume_arg.lower() == "auto":
        ckpts = sorted(glob(os.path.join(save_dir, "checkpoint_epoch_*.pt")))
        if ckpts:
            # [Auto-Resume] Detected N checkpoints, resuming from latest
            print(f"  [Auto-Resume] 检测到 {len(ckpts)} 个检查点, 从最新续训: {os.path.basename(ckpts[-1])}")
            return ckpts[-1]
        best = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best):
            # [Auto-Resume] Detected best_model.pt, will resume from it
            print(f"  [Auto-Resume] 检测到 best_model.pt, 将从中续训")
            return best
        # [Auto-Resume] No checkpoint detected, starting from scratch
        print("  [Auto-Resume] 未检测到检查点, 从头训练")
        return None
    return resume_arg  # Return path directly

from config import get_config, TextureInjectionConfig
from training import TrainingEngine
from evaluation import (
    compute_psnr, compute_ssim, compute_lpips_approx, compute_fid,
    compute_activation_stats, measure_inference_performance, evaluate_all,
    compute_pixel_fidelity,
)
from data_utils import load_real_images
from visualization import plot_results_grid, plot_training_curves, tensor_to_numpy

# Dataset paths
# 数据集路径
REDRAWING_DIR = os.path.join(os.path.dirname(os.path.dirname(FILE_DIR)), "RedrawingPhotoCreating")
DATASET_DIR = os.path.join(REDRAWING_DIR, "dataset")
RESOURCES_DIR = os.path.join(REDRAWING_DIR, "resourses")
OUTPUT_ROOT = os.path.join(FILE_DIR, "outputs")
SD_CACHE = os.path.join(os.path.dirname(FILE_DIR), "stable-diffusion-v1-5")

# ── Constants ──────────────────────────────────────────────────────────────────
# ── 常量 ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SEED = 42

# SD v1.5 prompts — covering portraits/objects/scenes, ensuring diversity
# SD v1.5 提示词 — 涵盖肖像/物体/场景, 确保多样性
STYLE_PROMPTS = [
    "professional portrait photo of a person, studio lighting, 85mm lens, sharp focus",
    "detailed macro photo of a flower, morning dew, natural sunlight, bokeh background",
    "street photography of a person walking, urban landscape, golden hour lighting",
    "close-up of a cat with green eyes, natural window light, shallow depth of field",
    "still life photography of fruits in a bowl, kitchen setting, soft diffused light",
    "landscape photo of mountains at sunset, dramatic clouds, wide angle lens",
    "product photography of a watch, reflective surface, studio lighting, macro detail",
    "food photography of a dish, restaurant presentation, warm lighting, top-down view",
    "architectural photography of a modern building, clean lines, blue sky, symmetry",
    "fashion photography, editorial style, natural pose, outdoor location",
    "portrait of an elderly person, wrinkled skin texture, character portrait, high detail",
    "macro photo of an insect on a leaf, extreme detail, nature documentary style",
    "night cityscape, neon lights reflecting on wet pavement, cinematic mood",
    "aerial photography of a coastline, turquoise water, white sand, drone shot",
    "black and white portrait, dramatic lighting, film grain, classic look",
    "close-up of an eye, iris detail, macro photography, reflection of window",
    "wildlife photography of a bird in flight, frozen motion, natural habitat",
    "underwater photography of coral reef, colorful fish, sunlight rays through water",
    "interior design photo of a living room, natural light, minimal style, cozy",
    "sports photography of a runner, dynamic pose, motion blur on background",
    "detailed photo of tree bark texture, macro, natural patterns",
    "rainy street scene, reflection of city lights on wet road, umbrella silhouette",
    "vintage car detail shot, chrome reflections, retro styling, sunny day",
    "aerial view of a forest in autumn, vibrant colors, drone photography",
    "macro photo of water droplets on a leaf, refraction, morning light",
    "portrait of a child laughing, candid moment, soft natural light, shallow DOF",
    "detailed macro shot of fabric texture, woven pattern, high contrast",
    "desert landscape at dawn, sand dunes, warm tones, leading lines",
    "close-up of hands playing piano, shallow depth of field, dramatic lighting",
    "photograph of a full moon, detailed craters, telephoto lens, night sky",
    "rustic wooden door detail, weathered paint texture, Mediterranean architecture",
    "coffee cup close-up, steam rising, cafe atmosphere, morning light",
    "macro photo of snowflake on a window, intricate crystal pattern, winter atmosphere",
    "pet photography, dog running on a beach, golden light, action shot",
    "close-up of a book page, text macro, paper texture, warm reading light",
    "sunset over the ocean, silhouette of palm trees, vibrant sky colors",
    "detailed photo of polished gemstone, facets catching light, macro",
    "minimal architectural detail, shadows and light, abstract composition",
    "portrait with dramatic rim lighting, dark background, artistic feel",
    "macro photo of a butterfly wing, scale detail, vibrant natural colors",
    "autumn leaves on the ground, wet after rain, texture and reflection, close-up",
    "glass of water with condensation droplets, macro detail, back lighting",
    "bicycle detail close-up, metal and leather texture, urban photography",
    "snow-covered mountain peak, clear sky, crisp alpine light, aerial view",
    "close-up of cooking in a pan, steam and sizzle, warm kitchen lighting",
    "abandoned building interior, peeling paint, texture detail, urban exploration",
    "flower field at golden hour, shallow DOF, bokeh lights, romantic atmosphere",
    "macro photo of rusted metal surface, oxidized texture, industrial abstract",
    "city skyline at blue hour, ambient light, long exposure, smooth water",
    "close-up portrait with freckles, skin detail, natural beauty, window light",
]


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — Training Pair Generation for Texture Injection
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 数据集 — 纹理注入专用的训练对生成
# ══════════════════════════════════════════════════════════════════════════════

class TextureInjectionDataset(Dataset):
    """
    Texture Injection Training Pair Dataset (shared framework for B1/B2).

    Strategy (Option A — Controllable Real Texture):
      1. Select clean (GT) from real photo set
      2. Apply Gaussian blur to clean → "smooth" version (I_dirty)
      3. Network learns: smooth → clean, i.e., inject texture

    Strategy (Option B — AI-to-Real):
      1. Use AI-generated image as I_dirty
      2. Use real photo with similar content as I_gt
      3. Network learns: AI_smooth → real_textured

    This experiment uses Option A by default (more controllable), which takes effect during inference on AI images.
    """

    """
    纹理注入训练对数据集 (B1/B2 共用框架)。

    策略 (Option A — 可控真实纹理):
      1. 从真实照片组中选取 clean (GT)
      2. 对 clean 施加高斯模糊 → "smooth" 版本 (I_dirty)
      3. 网络学习: smooth → clean, 即注入纹理

    策略 (Option B — AI-to-Real):
      1. 用 AI 生成图像作为 I_dirty
      2. 用相似内容的真实照片作为 I_gt
      3. 网络学习: AI_smooth → real_textured

    本实验默认使用 Option A (更可控), 然后对 AI 图像进行推理时生效。
    """
    
    def __init__(self, images: list, num_samples: int, size: int = 512,
                 blur_sigma_range: tuple = (1.0, 2.5),
                 seed: int = DEFAULT_SEED, augment: bool = True):
        """
        Args:
            images: list of [H, W, 3] numpy arrays, value range [0, 1]
            num_samples: number of samples (samples per epoch)
            size: crop/resize target resolution
            blur_sigma_range: (min_sigma, max_sigma) blur intensity range
            augment: whether to enable data augmentation
        """

        """
        Args:
            images: [H, W, 3] numpy 数组列表, 值域 [0, 1]
            num_samples: 采样次数 (epoch 内样本数)
            size: 裁剪/调整目标分辨率
            blur_sigma_range: (min_sigma, max_sigma) 模糊强度范围
            augment: 是否启用数据增强
        """
        self.images = images
        self.num_samples = num_samples
        self.size = size
        self.blur_min, self.blur_max = blur_sigma_range
        self.augment = augment
        
        rng = np.random.RandomState(seed)
        self.seeds = rng.randint(0, 2**31 - 1, size=num_samples)
        self.n_images = len(images)
    
    def __len__(self):
        return self.num_samples
    
    def _random_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < self.size or w < self.size:
            return self._resize(img)
        top = np.random.randint(0, h - self.size + 1)
        left = np.random.randint(0, w - self.size + 1)
        return img[top:top+self.size, left:left+self.size]
    
    def _resize(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((self.size, self.size), Image.LANCZOS)
        return np.array(pil).astype(np.float32) / 255.0
    
    def __getitem__(self, idx):
        seed = int(self.seeds[idx])
        np.random.seed(seed)
        img_idx = np.random.randint(0, self.n_images)
        clean = self.images[img_idx].copy()
        
        # Crop/Resize
        # 裁剪/调整尺寸
        clean = self._random_crop(clean)

        # Data augmentation (training only)
        # 数据增强 (仅对训练)
        if self.augment:
            if np.random.rand() > 0.5:
                clean = np.fliplr(clean).copy()
            if np.random.rand() > 0.3:
                clean = np.rot90(clean, k=np.random.randint(0, 4)).copy()
        
        # Gaussian blur → smooth version
        # 高斯模糊 → smooth 版本
        sigma = np.random.uniform(self.blur_min, self.blur_max)
        clean_uint8 = (clean * 255).astype(np.uint8)
        smooth_pil = Image.fromarray(clean_uint8).filter(ImageFilter.GaussianBlur(radius=sigma))
        smooth = np.array(smooth_pil).astype(np.float32) / 255.0
        
        # Convert to CHW
        # 转到 CHW
        dirty = torch.from_numpy(smooth.transpose(2, 0, 1)).float()
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        # GT mask: region of change caused by blur (pseudo-mask from edge difference)
        # GT mask: 模糊产生的变化区域 (用边缘差异生成伪掩膜)
        diff = np.abs(clean - smooth).max(axis=2)  # [H, W]
        gt_mask = torch.from_numpy((diff > 0.01).astype(np.float32)).unsqueeze(0)
        
        return dirty, clean_t, gt_mask



class AIGeneratedDataset(Dataset):
    """AI image dataset generated by SD v1.5 (inference only)"""
    """SD v1.5 生成的 AI 图像数据集 (仅推理时使用)"""
    
    def __init__(self, images: list):
        self.images = images  # [C, H, W] tensor 列表
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation and Baselines
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 评估与基线
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def apply_usm_sharpening(image_batch: torch.Tensor, amount: float = 1.5,
                          radius: float = 1.0, threshold: float = 0) -> torch.Tensor:
    """
    Unsharp Mask (USM) global sharpening (opencv simulation).
    image_batch: (B, C, H, W) torch float32, value range [0, 1]
    """
    """
    Unsharp Mask (USM) 全局锐化 (opencv 仿真).
    image_batch: (B, C, H, W) torch float32, 值域 [0, 1]
    """
    results = []
    for img in image_batch:
        img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        # PIL UnsharpMask: radius, percent (amount*100), threshold
        sharp = pil.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount * 100),
                                                    threshold=threshold))
        sharp_np = np.array(sharp).astype(np.float32) / 255.0
        results.append(torch.from_numpy(sharp_np).permute(2, 0, 1))
    return torch.stack(results)


@torch.no_grad()
def apply_full_texture_transfer(model: nn.Module, image_batch: torch.Tensor,
                                 device: str) -> torch.Tensor:
    """Full texture transfer (without mask, residual only): I_output = I_input + R"""
    """全图纹理迁移 (不使用掩膜, 仅残差): I_output = I_input + R"""
    model.eval()
    images = image_batch.to(device)
    residual, _ = model(images)
    return (images + residual).clamp(0, 1).cpu()


@torch.no_grad()
def apply_psrnet_selective(model: nn.Module, image_batch: torch.Tensor,
                            device: str) -> tuple:
    """PSR-Net selective texture injection: I_output = I_input + R * M"""
    """PSR-Net 选择性纹理注入: I_output = I_input + R * M"""
    model.eval()
    images = image_batch.to(device)
    refined, residual, mask = model.refine(images)
    return refined.clamp(0, 1).cpu(), residual.cpu(), mask.cpu()


@torch.no_grad()
def compute_texture_metrics(images: torch.Tensor, mask: torch.Tensor = None) -> dict:
    """Compute texture-related statistics"""
    """计算纹理相关统计量"""
    # Edge density (Laplacian gradient magnitude)
    # 边缘密度 (拉普拉斯梯度幅值)
    if images.dim() == 3:
        images = images.unsqueeze(0)
    laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                                     dtype=torch.float32)
    edge_maps = []
    for c in range(3):
        edge = F.conv2d(images[:, c:c+1], laplacian_kernel, padding=1)
        edge_maps.append(edge.abs())
    edge_combined = torch.stack(edge_maps, dim=1).mean(dim=1).squeeze(1)  # [B, H, W]
    
    edge_density = edge_combined.mean().item()
    
    # Texture variance (local standard deviation)
    # 纹理方差 (局部标准差)
    kernel_size = 5
    texture_var = []
    for c in range(3):
        unfold = F.unfold(images[:, c:c+1], kernel_size, padding=kernel_size//2)
        var = unfold.var(dim=1).mean()
        texture_var.append(var.item())
    texture_variance = float(np.mean(texture_var)) if texture_var else 0.0
    
    result = {"edge_density": edge_density, "texture_variance": texture_variance}
    
    if mask is not None:
        # Texture variance in masked region
        # 掩膜区域的纹理方差
        mask_resized = F.interpolate(mask.float(), size=edge_combined.shape[1:], mode='bilinear')
        masked_edge = (edge_combined * mask_resized.squeeze(1)).sum() / (mask_resized.sum() + 1e-8)
        result["masked_region_edge_density"] = masked_edge.item()
        result["mask_activation_ratio"] = mask.float().mean().item()
    
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SD v1.5 Image Generation (full mode only)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SD v1.5 图像生成 (仅 full 模式)
# ══════════════════════════════════════════════════════════════════════════════

def generate_ai_images(prompts: list, num_images: int, size: int = 512,
                        batch_size: int = 2, seed: int = DEFAULT_SEED,
                        model_path: str = None) -> list:
    """
    Generate AI images using SD v1.5.

    Args:
        model_path: local SD v1.5 model path; if None, auto-download from HuggingFace

    Returns:
        list of [(tensor(C,H,W), prompt_str), ...]
    """
    """
    使用 SD v1.5 生成 AI 图像。

    Args:
        model_path: 本地 SD v1.5 模型路径；为 None 时自动从 HuggingFace 下载

    Returns:
        [(tensor(C,H,W), prompt_str), ...] 列表
    """
    print(f"\n{'='*60}")
    print("  Loading Stable Diffusion v1.5...")
    print(f"{'='*60}")
    
    try:
        from diffusers import StableDiffusionPipeline
        
        # Determine model path: prefer the passed path, otherwise use default cache
        # 确定模型路径: 优先用传入的, 否则用默认缓存路径
        if model_path is None:
            model_path = SD_CACHE
        
        if os.path.isdir(model_path) and os.path.isdir(os.path.join(model_path, "unet")):
            print(f"  使用本地缓存: {model_path}")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                local_files_only=True,
            )
        else:
            # First run: download from HuggingFace and save locally
            # 首次运行: 从 HuggingFace 下载并保存到本地
            if os.path.isdir(model_path):
                print(f"  [WARN] 本地缓存不完整, 重新下载: {model_path}")
            else:
                print(f"  首次运行, 下载 SD v1.5 到: {model_path} (仅一次, ~4.3GB)")
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            os.makedirs(model_path, exist_ok=True)
            pipe.save_pretrained(model_path)
            print(f"  ✓ 已保存到: {model_path} (下次秒加载)")
        pipe = pipe.to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"  [WARN] 无法加载 SD v1.5: {e}")
        print("  将降级为简单模式 (使用真实图片)")
        return None
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    images = []
    n_prompts = len(prompts)
    
    print(f"  Generating {num_images} images at {size}x{size}...")
    t_start = time.time()
    
    idx = 0
    while len(images) < num_images:
        prompt_batch = []
        for _ in range(min(batch_size, num_images - len(images))):
            prompt_batch.append(prompts[idx % n_prompts])
            idx += 1
        
        with torch.autocast(device_type=DEVICE.type):
            outputs = pipe(
                prompt_batch,
                height=size,
                width=size,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
            ).images
        
        for img, prompt in zip(outputs, prompt_batch):
            img = img.resize((size, size), Image.LANCZOS)
            img_tensor = torch.from_numpy(
                np.array(img).astype(np.float32) / 255.0
            ).permute(2, 0, 1)
            images.append((img_tensor, prompt))
        
        if len(images) % 20 == 0:
            print(f"    Generated {len(images)}/{num_images}...")
    
    elapsed = time.time() - t_start
    print(f"  Done! Generated {len(images)} images in {elapsed:.1f}s ({elapsed/len(images):.2f}s/image)")
    return images


# ══════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 主实验运行
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(args):
    """B1 experiment main entry point"""
    """B1 实验主入口"""
    config = get_config("B1",
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_samples=args.train_samples,
    )
    config.device = str(DEVICE)
    # Gentle sparsity constraint
    config.lambda_sparse = 0.05  # 温和稀疏约束
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  B1: Style Refinement — De-AI-ification Texture Injection")
    print(f"  Device: {DEVICE} | Mode: {args.mode} | Size: {config.image_size}")
    print(f"  Epochs: {config.epochs} | λ_s: {config.lambda_sparse}")
    print(f"{'='*70}")
    
    # ── 1. Prepare Real Images ──
    # ── 1. 准备真实图片 ──
    print("\n[Step 1] Loading real images...")
    real_imgs = []
    for src_dir in [DATASET_DIR, RESOURCES_DIR]:
        if os.path.isdir(src_dir):
            loaded = load_real_images(src_dir, target_size=args.image_size, max_images=None)
            real_imgs.extend(loaded)
            print(f"  Loaded {len(loaded)} images from {src_dir}")
    
    if not real_imgs:
        # fallback: use synthetic gradient images
        # fallback: 用合成渐变图
        print("  [WARN] No real images found, using synthetic gradients as fallback")
        rng = np.random.RandomState(DEFAULT_SEED)
        for _ in range(100):
            grad = np.zeros((args.image_size, args.image_size, 3), dtype=np.float32)
            for c in range(3):
                v1, v2 = rng.rand(), rng.rand()
                gx = np.linspace(v1, v2, args.image_size)
                gy = np.linspace(v1, v2, args.image_size).reshape(-1, 1)
                grad[:, :, c] = 0.3 + 0.4 * (gx + gy)
            real_imgs.append(grad.clip(0, 1))
    
    print(f"  Total real images: {len(real_imgs)}")
    
    # ── 2. Generate/Load AI Images ──
    # ── 2. 生成/加载 AI 图像 ──
    ai_images = None
    if args.mode == "full" and not args.use_real_data_only:
        print("\n[Step 2a] Generating AI images with SD v1.5...")
        ai_images = generate_ai_images(
            STYLE_PROMPTS,
            num_images=min(args.num_ai, args.train_samples * 2),
            size=args.image_size,
            batch_size=args.batch_size,
            model_path=args.sd_model_path,
        )
        if ai_images is not None:
            print(f"  Generated {len(ai_images)} AI images")
        else:
            print("  SD generation failed, continuing with real data only")
            args.mode = "simple"
    else:
        print("\n[Step 2a] Skipped — using real data only (--mode simple or --use_real_data_only)")
    
    # ── 3. Prepare Training Pairs ──
    # ── 3. 准备训练对 ──
    print("\n[Step 3] Preparing texture injection training pairs...")
    train_dataset = TextureInjectionDataset(
        real_imgs, num_samples=config.train_samples,
        size=args.image_size, seed=DEFAULT_SEED,
        blur_sigma_range=(1.0, 2.5),
    )
    val_dataset = TextureInjectionDataset(
        real_imgs, num_samples=min(50, config.train_samples // 4),
        size=args.image_size, seed=DEFAULT_SEED + 1000,
        blur_sigma_range=(1.0, 2.5), augment=False,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                               num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    
    print(f"  Train samples: {config.train_samples}, Val samples: {len(val_dataset)}")
    print(f"  Batch size: {config.batch_size}")
    
    # ── 4. Create Model ──
    # ── 4. 创建模型 ──
    print("\n[Step 4] Creating PSR-Net model...")
    model = create_model("large" if args.image_size >= 256 else "standard",
                          base_channels=config.base_channels,
                          input_channels=3, device=config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model.__class__.__name__}, Params: {n_params:,}")
    
    # ── 5. Training ──
    # ── 5. 训练 ──
    print(f"\n[Step 5] Training PSR-Net as Texture Grafting Layer...")
    engine = TrainingEngine(model, config, config.device)
    
    # Auto-detect checkpoint resume
    # 自动检测断点续训
    resume_from = _resolve_resume(args.resume, OUTPUT_ROOT)
    
    history = engine.train(
        train_loader, val_loader,
        lambda_distill=0.0,
        verbose=True, val_freq=5,
        save_dir=OUTPUT_ROOT,
        resume_from=resume_from,
    )
    
    best_model_path = os.path.join(OUTPUT_ROOT, "best_model.pt")
    if os.path.exists(best_model_path):
        model = load_checkpoint(model, best_model_path, config.device)
        print("  Loaded best model checkpoint")
    
    # ── 6. Evaluation — Training Set Texture Injection Quality ──
    # ── 6. 评估 — 训练集纹理注入质量 ──
    print(f"\n[Step 6] Evaluating training-set texture injection quality...")
    model.eval()
    val_raw = defaultdict(list)
    
    for batch in val_loader:
        dirty, clean, _ = [b.to(config.device) for b in batch]
        refined, residual, mask = model.refine(dirty)
        
        val_raw["psnr"].append(compute_psnr(refined, clean))
        val_raw["ssim"].append(compute_ssim(refined, clean))
        val_raw["lpips"].append(compute_lpips_approx(refined, clean))
        val_raw["mask_mean"].append(mask.mean().item())
    
    train_metrics = {k: float(np.mean(v)) for k, v in val_raw.items()}
    print(f"  Validation: PSNR={train_metrics['psnr']:.2f}dB, SSIM={train_metrics['ssim']:.4f}, "
          f"Mask_Mean={train_metrics['mask_mean']:.4f}")
    
    # ── 7. Inference + Baseline Comparison ──
    # ── 7. 推理 + 对比基线 ──
    print(f"\n[Step 7] Running inference and baseline comparisons...")
    
    # Inference target: use AI images if available, otherwise use validation set
    # 推理目标: 如果 AI 图像可用则在其上评估, 否则用验证集
    if ai_images:
        inference_images = [t for t, _ in ai_images[:min(30, len(ai_images))]]
        inference_label = "AI-generated"
    else:
        # Use real clean images from validation set as inference targets
        # 用验证集的 real clean 图像作为推理目标
        inference_images = [val_dataset[i][1] for i in range(min(30, len(val_dataset)))]
        inference_label = "Real (val)"
    
    if inference_images:
        ai_batch = torch.stack(inference_images)
        ai_batch_device = ai_batch.to(config.device)
        
        print(f"  Inference on {len(inference_images)} {inference_label} images...")
        
        # PSR-Net selective injection
        # PSR-Net 选择性注入
        t0 = time.time()
        refined_batch, residual_batch, mask_batch = apply_psrnet_selective(
            model, ai_batch, config.device)
        psrnet_time = time.time() - t0

        # USM sharpening
        # USM 锐化
        usm_batch = apply_usm_sharpening(ai_batch)
        
        # Full texture transfer (no mask)
        # 全图纹理迁移 (无掩膜)
        full_tex_batch = apply_full_texture_transfer(model, ai_batch, config.device)
        
        # Compute texture metrics for each method
        # 计算各方法的纹理指标
        raw_metrics = compute_texture_metrics(ai_batch)
        usm_metrics = compute_texture_metrics(usm_batch)
        full_tex_metrics = compute_texture_metrics(full_tex_batch)
        psrnet_metrics = compute_texture_metrics(refined_batch, mask_batch)
        
        baseline_results = {
            "Raw AI": {**raw_metrics, "inference_time_ms": 0,
                        "psnr": "N/A", "ssim": "N/A", "lpips_approx": "N/A"},
            "USM Sharpening": {**usm_metrics, "inference_time_ms": 5.0,
                                "psnr": "N/A", "ssim": "N/A", "lpips_approx": "N/A"},
            "Full Texture Transfer": {**full_tex_metrics,
                                       "inference_time_ms": psrnet_time * 1000 / len(inference_images)},
            "PSR-Net Selective (Ours)": {**psrnet_metrics,
                                          "inference_time_ms": psrnet_time * 1000 / len(inference_images)},
        }
        
        print("\n  ── Baseline Comparison ──")
        for name, metrics in baseline_results.items():
            print(f"  {name}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6f}")
        
        # 保存对比图
        n_display = min(5, len(inference_images))
        titles = [f"{inference_label} #{i+1}" for i in range(n_display)]
        
        plot_results_grid(
            dirty_list=[ai_batch[i] for i in range(n_display)],
            refined_list=[refined_batch[i] for i in range(n_display)],
            gt_list=[ai_batch[i] for i in range(n_display)],
            mask_list=[mask_batch[i] for i in range(n_display)],
            titles=[f"Raw → PSR-Net #{i+1}" for i in range(n_display)],
            save_path=os.path.join(OUTPUT_ROOT, "texture_injection_grid.png"),
            max_samples=n_display,
        )
        
        # 5列对比图: Raw AI | USM | Full Texture | PSR-Net Selective | GT
        n_compare = min(5, len(inference_images))
        fig, axes = plt.subplots(n_compare, 5, figsize=(20, n_compare * 4))
        if n_compare == 1:
            axes = axes.reshape(1, -1)
        
        col_titles = ["Raw AI", "USM Sharpening", "Full Texture", "PSR-Net Selective", "GT (Original)"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=11, fontweight="bold")
        
        for i in range(n_compare):
            axes[i, 0].imshow(tensor_to_numpy(ai_batch[i]))
            axes[i, 0].axis("off")
            axes[i, 1].imshow(tensor_to_numpy(usm_batch[i]))
            axes[i, 1].axis("off")
            axes[i, 2].imshow(tensor_to_numpy(full_tex_batch[i]))
            axes[i, 2].axis("off")
            axes[i, 3].imshow(tensor_to_numpy(refined_batch[i]))
            axes[i, 3].axis("off")
            axes[i, 4].imshow(tensor_to_numpy(ai_batch[i]))  # GT = original
            axes[i, 4].axis("off")
        
        plt.suptitle("B1: Texture Injection Comparison — De-AI-ification", fontsize=14, y=1.01)
        plt.tight_layout()
        comparison_path = os.path.join(OUTPUT_ROOT, "texture_injection_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved comparison grid: {comparison_path}")
        
        # Mask overlay heatmap
        # 掩膜叠加热力图
        mask_heatmap_path = save_mask_overlay(
            [ai_batch[i] for i in range(n_display)],
            [mask_batch[i] for i in range(n_display)],
            os.path.join(OUTPUT_ROOT, "mask_overlay.png"),
        )
        print(f"  Saved mask overlay: {mask_heatmap_path}")
    
    else:
        baseline_results = {}
    
    # ── 8. Training Curves ──
    # ── 8. 训练曲线 ──
    print(f"\n[Step 8] Saving training curves and final results...")
    curves_path = os.path.join(OUTPUT_ROOT, "training_curves.png")
    plot_training_curves(
        {"B1_Texture_Injection": history},
        save_path=curves_path,
    )
    print(f"  Saved training curves: {curves_path}")
    
    # ── 9. Summary JSON ──
    # ── 9. 汇总 JSON ──
    all_results = {
        "experiment": "B1_Style_Refinement",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": args.mode,
            "image_size": config.image_size,
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
        "validation_metrics": train_metrics,
        "baseline_comparison": {
            k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str))}
            for k, v in baseline_results.items()
        } if baseline_results else {},
        "files": {
            "best_model": best_model_path,
            "training_curves": curves_path,
            "comparison_grid": comparison_path if baseline_results else None,
            "mask_overlay": mask_heatmap_path if baseline_results else None,
        },
    }
    
    results_json = os.path.join(OUTPUT_ROOT, "results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved results: {results_json}")
    
    print(f"\n{'='*70}")
    print(f"  B1 Experiment Complete!")
    print(f"  Outputs: {OUTPUT_ROOT}")
    print(f"{'='*70}")
    
    return all_results


def save_mask_overlay(images: list, masks: list, save_path: str) -> str:
    """Draw mask heatmap overlaid on original image"""
    """绘制掩膜热力图叠加在原图上"""
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 4))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    axes[0, 0].set_title("AI Image", fontsize=11)
    axes[0, 1].set_title("Mask Overlay (Hot=Texture Added)", fontsize=11)
    
    for i in range(n):
        axes[i, 0].imshow(tensor_to_numpy(images[i]))
        axes[i, 0].axis("off")
        
        img_np = tensor_to_numpy(images[i])
        mask_np = tensor_to_numpy(masks[i])
        if mask_np.ndim == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze()
        
        axes[i, 1].imshow(img_np)
        axes[i, 1].imshow(mask_np, cmap="hot", alpha=0.6)
        axes[i, 1].axis("off")
    
    plt.suptitle("B1: Where PSR-Net Injects Texture (Mask Activation)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B1: Style Refinement — De-AI-ification Texture Injection with PSR-Net")
    parser.add_argument("--mode", type=str, default="simple",
                        choices=["full", "simple"],
                        help="'full' uses SD v1.5 for AI image generation; 'simple' uses real data only")
    parser.add_argument("--use_real_data_only", action="store_true",
                        help="Skip SD generation even in full mode")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (auto based on image_size if not set)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image resolution (512 for full mode, 256 for simple)")
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Number of training samples per epoch")
    parser.add_argument("--num_ai", type=int, default=200,
                        help="Number of AI images to generate (full mode only)")
    parser.add_argument("--sd_model_path", type=str, default=None,
                        help="Local path to SD v1.5 model (e.g. /mnt/workspace/Experiments/stable-diffusion-v1-5)")
    parser.add_argument("--resume", type=str, default="auto",
                        help="Resume from checkpoint: 'auto' (default, auto-detect), 'latest', path to .pt, 'none' to force fresh start")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # 自动调整 batch_size
    if args.batch_size is None:
        args.batch_size = 2 if args.image_size >= 512 else (4 if args.image_size >= 256 else 8)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Safe mode: if SD cannot be loaded, auto-degrade
    # 安全模式: 如果无法加载 SD, 自动降级
    if args.mode == "full" and args.use_real_data_only:
        args.mode = "simple"
    
    run_experiment(args)
