#!/usr/bin/env python3
"""
SD image cache preparation script
-----------------------------------
Reads pre-generated 512x512 PNG images from resources/, downscales to each experiment's
required resolution, and saves as .pt cache files for direct loading by SDImageDataset.

Must be run after generate_sd_images.py completes.

SD 图片缓存准备脚本
-------------------
从 resources/ 中读取预生成的 512×512 PNG 图片，下采样到各实验所需的分辨率，
保存为 .pt 缓存文件，供 SDImageDataset 直接加载。

必须在 generate_sd_images.py 运行完成后执行。

Usage:
  python prepare_sd_cache.py                    # Generate all caches for 64/128/256/512
  python prepare_sd_cache.py --size 64          # Only generate 64x64
  python prepare_sd_cache.py --sizes 64 128 256 # Generate specified resolutions

Output:
  resources/sd_images/sd_s64_n10000_seed0.pt
  resources/sd_images/sd_s128_n10000_seed0.pt
  resources/sd_images/sd_s256_n10000_seed0.pt
  resources/sd_images/sd_s512_n10000_seed0.pt

Dependencies: pip install pillow numpy torch

用法:
  python prepare_sd_cache.py                    # 生成 64/128/256/512 全部缓存
  python prepare_sd_cache.py --size 64          # 仅生成 64×64
  python prepare_sd_cache.py --sizes 64 128 256 # 生成指定分辨率

输出:
  resources/sd_images/sd_s64_n10000_seed0.pt
  resources/sd_images/sd_s128_n10000_seed0.pt
  resources/sd_images/sd_s256_n10000_seed0.pt
  resources/sd_images/sd_s512_n10000_seed0.pt

依赖: pip install pillow numpy torch
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
# ── 路径 ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESOURCES    = os.path.join(BASE_DIR, "resources")
MANIFEST     = os.path.join(RESOURCES, "manifest.json")
SD_CACHE_DIR = os.path.join(RESOURCES, "sd_images")
CACHE_SEED   = 42              # Aligned with ExperimentConfig.seed=42 and SDImageDataset default seed
                               # 与 ExperimentConfig.seed=42 和 SDImageDataset 默认 seed 对齐

# Default resolutions used by experiments (see A4/run.py)
# 各实验默认使用的分辨率（参见 A4/run.py）
DEFAULT_SIZES = [64, 128, 256, 512]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare SD image cache from pre-generated PNGs")
    p.add_argument("--resources", type=str, default=RESOURCES,
                   help="预生成图片目录")
    p.add_argument("--output", type=str, default=SD_CACHE_DIR,
                   help=".pt 缓存输出目录")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                   help="目标分辨率列表")
    p.add_argument("--max_images", type=int, default=0,
                   help="每个分辨率最多使用 N 张图（0=全部）")
    return p.parse_args()


def load_images_from_resources(resources_dir: str, max_images: int = 0) -> list:
    """
    Load all PNG images from the resources/ directory.
    Iterates through prompt_XXXX/ subdirectories, collects all .png files.
    Returns a list of [(path, np_array_512)].

    从 resources/ 目录加载所有 PNG 图片。
    遍历 prompt_XXXX/ 子目录，收集所有 .png 文件。
    返回 [(path, np_array_512)] 列表。
    """
    images = []
    prompt_dirs = sorted(Path(resources_dir).glob("prompt_*"))

    for pdir in prompt_dirs:
        if not pdir.is_dir():
            continue
        png_files = sorted(pdir.glob("*.png"))
        for png_path in png_files:
            images.append(str(png_path))
            if max_images > 0 and len(images) >= max_images:
                return images

    return images


def build_cache(images: list, size: int, output_dir: str) -> str:
    """
    Resize a list of images to target_size x target_size, save as .pt file.

    .pt format: list of torch.Tensors, each tensor shape (H, W, C), float32, range [0, 1].
    This matches the format expected by SDImageDataset.

    将图片列表 resize 到 target_size × target_size，保存为 .pt 文件。

    .pt 格式：torch.Tensor 列表，每个 tensor 形状为 (H, W, C)，float32，值域 [0, 1]。
    这匹配 SDImageDataset 期望的格式。
    """
    total = len(images)
    print(f"\n  Building {size}×{size} cache from {total} images...")

    resized = []
    for i, img_path in enumerate(images):
        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != (size, size):
                img = img.resize((size, size), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0  # HWC, [0,1]
            resized.append(arr)
        except Exception as e:
            print(f"    ⚠️  Failed to load {img_path}: {e}")
            continue

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{total}...")

    # Save as .pt (compatible with SDImageDataset format)
    # 保存为 .pt（与 SDImageDataset 格式兼容）
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, f"sd_s{size}_n{len(resized)}_seed{CACHE_SEED}.pt")
    tensors = [torch.from_numpy(arr) for arr in resized]
    torch.save(tensors, cache_path, _use_new_zipfile_serialization=False)

    print(f"    ✅ Saved {len(resized)} images → {cache_path}")
    print(f"       File size: {os.path.getsize(cache_path) / (1024**2):.1f} MB")
    return cache_path


def main():
    args = parse_args()

    if not os.path.isdir(args.resources):
        print(f"❌ resources/ 目录不存在: {args.resources}")
        print(f"   请先运行 generate_sd_images.py 生成图片")
        sys.exit(1)

    # 检查 manifest
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r") as f:
            manifest = json.load(f)
        print(f"[Manifest] {len(manifest)} prompts, "
              f"{sum(len(m.get('images', [])) for m in manifest)} images total")
    else:
        print(f"[Manifest] 未找到 manifest.json，将直接扫描目录")

    # 加载所有图片路径
    print(f"\n[Scan] 扫描 resources/ 中的图片...")
    images = load_images_from_resources(args.resources, args.max_images)
    print(f"[Scan] 找到 {len(images)} 张 PNG 图片")

    if len(images) == 0:
        print("❌ 未找到任何图片，请先运行 generate_sd_images.py")
        sys.exit(1)

    # 生成各分辨率的缓存
    print(f"\n[Cache] 目标分辨率: {args.sizes}")
    for size in args.sizes:
        build_cache(images, size, args.output)

    print(f"\n{'='*60}")
    print(f"✅ 全部缓存已生成")
    print(f"   输出目录: {args.output}")
    print(f"\n现在训练时使用 --use_sd，SDImageDataset 将直接加载此缓存。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
