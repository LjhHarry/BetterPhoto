#!/usr/bin/env python3
"""
Batch SD v1.5 image generation script
--------------------------------------
Reads prompts.txt, generates 4 512x512 images per prompt, outputs to resources/.

批量 SD v1.5 图像生成脚本
-----------------------
读取 prompts.txt，每个提示词生成 4 张 512×512 图像，输出到 resources/。

Usage:
  # Use default SD cache path
  python generate_sd_images.py

  # Specify SD model path
  python generate_sd_images.py --model_path /path/to/stable-diffusion-v1-5

  # Custom parameters
  python generate_sd_images.py --width 512 --height 512 --steps 30 --guidance 7.5

  # Only generate first N prompts (for testing)
  python generate_sd_images.py --limit 10

Output structure:
  resources/
    ├── prompt_0000/
    │   ├── 0000.png
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── 0003.png
    ├── prompt_0001/
    │   ├── 0000.png
    │   ├── ...
    └── manifest.json       ← prompt index file

Dependencies: pip install diffusers transformers accelerate

用法:
  # 使用默认 SD 缓存路径
  python generate_sd_images.py

  # 指定 SD 模型路径
  python generate_sd_images.py --model_path /path/to/stable-diffusion-v1-5

  # 自定义参数
  python generate_sd_images.py --width 512 --height 512 --steps 30 --guidance 7.5

  # 只生成前 N 个 prompt（测试用）
  python generate_sd_images.py --limit 10

输出结构:
  resources/
    ├── prompt_0000/
    │   ├── 0000.png
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── 0003.png
    ├── prompt_0001/
    │   ├── 0000.png
    │   ├── ...
    └── manifest.json       ← prompt 索引文件

依赖: pip install diffusers transformers accelerate
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np


# ── Default configuration ─────────────────────────────────────────────────────
# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_SD_PATH = os.environ.get(
    "SD_MODEL_PATH",
    os.path.expanduser("~/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/latest"),
)
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
IMAGES_PER_PROMPT = 4


def parse_args():
    p = argparse.ArgumentParser(description="Batch SD v1.5 image generator")
    p.add_argument("--model_path", type=str, default=DEFAULT_SD_PATH,
                   help="SD v1.5 模型路径 (本地目录)")
    p.add_argument("--prompts_file", type=str, default=PROMPTS_FILE,
                   help="提示词文件路径 (每行一个)")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                   help="输出目录")
    p.add_argument("--width", type=int, default=512, help="图像宽度")
    p.add_argument("--height", type=int, default=512, help="图像高度")
    p.add_argument("--steps", type=int, default=30, help="推理步数")
    p.add_argument("--guidance", type=float, default=7.5, help="CFG scale")
    p.add_argument("--seed", type=int, default=42, help="基础随机种子")
    p.add_argument("--limit", type=int, default=0,
                   help="只生成前 N 个 prompt（0=全部）")
    p.add_argument("--resume", action="store_true",
                   help="从上次中断处继续（跳过已有4张图的 prompt）")
    p.add_argument("--batch_size", type=int, default=1,
                   help="每次生成的图像数（1 或 4，4 需要更多显存）")
    p.add_argument("--device", type=str, default="",
                   help="设备 (cuda/cpu，默认自动检测)")
    p.add_argument("--no_fp16", action="store_true",
                   help="使用 FP32 而非 FP16（需要更多显存）")
    return p.parse_args()


def load_prompts(path: str, limit: int = 0) -> list:
    """Read prompts file, return [(index, prompt), ...] list"""
    """读取提示词文件，返回 [(index, prompt), ...] 列表"""
    if not os.path.exists(path):
        print(f"Error: Prompts file not found: {path}")
        print("Please create a prompts.txt file with one prompt per line.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if limit > 0:
        prompts = prompts[:limit]
    return prompts


def load_pipeline(model_path: str, device: str, use_fp16: bool = True):
    """Load SD v1.5 pipeline"""
    """加载 SD v1.5 pipeline"""
    from diffusers import StableDiffusionPipeline

    print(f"[SD] Loading from: {model_path}")
    dtype = torch.float16 if use_fp16 else torch.float32

    if os.path.isdir(model_path) and os.path.isdir(os.path.join(model_path, "unet")):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None,
            local_files_only=True,
        )
    else:
        print(f"[SD] Local model not found at {model_path}")
        print(f"[SD] Trying to download from HuggingFace...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            safety_checker=None,
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def make_seed(base_seed: int, prompt_idx: int, image_idx: int) -> int:
    """Generate a deterministic seed for each prompt + image combination"""
    """每个 prompt + image 的组合生成一个确定性的种子"""
    return base_seed + prompt_idx * 100 + image_idx


def count_existing_images(prompt_dir: Path) -> int:
    """Check how many valid images already exist in a prompt directory"""
    """检查某个 prompt 目录下已有几张有效图片"""
    if not prompt_dir.exists():
        return 0
    count = 0
    for i in range(IMAGES_PER_PROMPT):
        img_path = prompt_dir / f"{i:04d}.png"
        if img_path.exists() and img_path.stat().st_size > 0:
            count += 1
    return count


def main():
    args = parse_args()

    # ── Device ──
    # ── 设备 ──
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ── Read prompts ──
    # ── 读取 prompts ──
    prompts = load_prompts(args.prompts_file, args.limit)
    total_prompts = len(prompts)
    total_images = total_prompts * IMAGES_PER_PROMPT
    print(f"[Prompts] {total_prompts} prompts → {total_images} images")
    print(f"[Output]  {args.output_dir}")

    # ── Load pipeline ──
    # ── 加载 pipeline ──
    print(f"\n{'='*60}")
    pipe = load_pipeline(args.model_path, device, use_fp16=not args.no_fp16)
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Manifest ──
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = []

    # ── Resume from checkpoint ──
    # ── 断点续传 ──
    resume_idx = 0
    if args.resume:
        while resume_idx < total_prompts:
            prompt_dir = Path(args.output_dir) / f"prompt_{resume_idx:04d}"
            existing = count_existing_images(prompt_dir)
            if existing < IMAGES_PER_PROMPT:
                break
            resume_idx += 1
        if resume_idx > 0:
            print(f"[Resume] Skipping first {resume_idx} prompts (already complete)")

    # ── Generation loop ──
    # ── 生成循环 ──
    start_time = time.time()
    generated_this_session = 0

    for p_idx in range(resume_idx, total_prompts):
        prompt = prompts[p_idx]
        prompt_dir = Path(args.output_dir) / f"prompt_{p_idx:04d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Check existing images (supports resume)
        # 检查已有图片（支持断点续传）
        existing = count_existing_images(prompt_dir)

        if existing == IMAGES_PER_PROMPT and args.resume:
            print(f"[{p_idx+1}/{total_prompts}] prompt_{p_idx:04d} — ✅ 已存在，跳过")
            manifest.append({
                "index": p_idx,
                "prompt": prompt,
                "dir": str(prompt_dir.relative_to(args.output_dir)),
                "images": [str(prompt_dir / f"{i:04d}.png") for i in range(IMAGES_PER_PROMPT)],
            })
            continue

        if existing > 0:
            print(f"[{p_idx+1}/{total_prompts}] prompt_{p_idx:04d} — 已有 {existing}/4 张，补全剩余")

        # Generate
        # 生成
        images = []
        try:
            for img_idx in range(IMAGES_PER_PROMPT):
                img_path = prompt_dir / f"{img_idx:04d}.png"

                if img_path.exists() and img_path.stat().st_size > 0:
                    # Image already exists, skip
                    # 已有这张图，跳过
                    continue

                seed = make_seed(args.seed, p_idx, img_idx)
                generator = torch.Generator(device=device).manual_seed(seed)

                result = pipe(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    width=args.width,
                    height=args.height,
                    generator=generator,
                )
                img = result.images[0]
                img.save(str(img_path))
                images.append(str(img_path))
                generated_this_session += 1

        except Exception as e:
            print(f"  ❌ prompt_{p_idx:04d} 生成失败: {e}")
            # Save prompt info for inspection
            # 保存 prompt 信息以便检查
            with open(prompt_dir / "error.txt", "w") as f:
                f.write(f"Error: {e}\nPrompt: {prompt}\n")
            continue

        # Update manifest
        # 更新 manifest
        manifest.append({
            "index": p_idx,
            "prompt": prompt,
            "dir": str(prompt_dir.relative_to(args.output_dir)),
            "images": [str(prompt_dir / f"{i:04d}.png") for i in range(IMAGES_PER_PROMPT)],
        })

        # Progress
        # 进度
        elapsed = time.time() - start_time
        done = p_idx + 1 - resume_idx
        rate = done / max(elapsed, 1)  # prompts / sec
        eta = (total_prompts - p_idx - 1) / max(rate, 1e-6)
        print(f"[{p_idx+1}/{total_prompts}] prompt_{p_idx:04d} "
              f"({done}/{total_prompts-resume_idx}) | "
              f"{elapsed/60:.0f}m elapsed | ETA {eta/60:.0f}m | "
              f"{generated_this_session} imgs this run")

        # Save manifest every 50 prompts (fault tolerance)
        # 每 50 个 prompt 保存一次 manifest（容错）
        if (p_idx + 1) % 50 == 0:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── Final save manifest ──
    # ── 最终保存 manifest ──
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ── Complete ──
    # ── 完成 ──
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ 完成!")
    print(f"   Prompts: {total_prompts}")
    print(f"   生成图像: {generated_this_session} 张 (本次)")
    print(f"   总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"   输出目录: {args.output_dir}")
    print(f"   Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
     # ── Aliyun auto-shutdown (for overnight runs) ──
     # ── 阿里云自动停机（半夜跑完专用） ──
    if args.shutdown:
        print("\n⚠️  任务完成！将在 60 秒后执行阿里云停机命令...")
        print("   (如需紧急取消，请新开终端运行: shutdown -c)")
        
        try:
            for i in range(60, 0, -1):
                print(f"\r   倒计时: {i} 秒 ", end="", flush=True)
                time.sleep(1)
            print()
            
            # Aliyun PAI-DSW / ECS generic shutdown command
            # 阿里云 PAI-DSW / ECS 通用停机命令
            # Note: ECS users must check "Pay-as-you-go instances stop without charge" in the console!
            # 注意：ECS 用户必须提前在控制台勾选「按量付费实例停机不收费」！
            os.system("sudo shutdown -h now") 
            
        except KeyboardInterrupt:
            print("\n\n❌ 已手动取消停机，实例将继续运行并计费！")