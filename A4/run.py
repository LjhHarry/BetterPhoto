#!/usr/bin/env python
"""
Experiment A4: High-Resolution Experiments (64/128/256/512) + Real Datasets

Scale PSR-Net to higher resolutions and test on synthetic + real datasets.
Generates resolution comparison plots and saves all metrics to JSON.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# -- Path setup --
_current_dir = os.path.dirname(os.path.abspath(__file__))
_experiments_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _experiments_dir)

from common.config import ExperimentConfig
from common.model_factory import create_model, get_model_info
from common.evaluation import (
    compute_psnr, compute_ssim, compute_mask_contrast_ratio,
    measure_inference_performance, sanitize_metric_array,
)
from common.data_utils import (
    SyntheticDataset, RealImageDataset, load_real_images,
    apply_random_degradation, SDImageDataset,
)
from common.visualization import tensor_to_numpy, plot_training_curves
from common.training import TrainingEngine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

# SD 缓存目录（与 common/data_utils.py SDImageDataset._RESOURCES_DIR 一致）
_SD_CACHE_DIR = os.path.join(
    os.path.dirname(_current_dir), "resources", "sd_images")


def _auto_detect_sd_cache(image_size, num_samples, seed=42):
    """检测 resources/sd_images/ 是否有匹配的 SD 缓存文件。"""
    import glob as _glob
    pattern = os.path.join(_SD_CACHE_DIR, f"sd_s{image_size}_n*_seed{seed}.pt")
    candidates = _glob.glob(pattern)
    usable = []
    for c in candidates:
        try:
            n = int(os.path.basename(c).split("_n")[1].split("_")[0])
            if n >= num_samples:
                usable.append((n, c))
        except (IndexError, ValueError):
            continue
    if usable:
        usable.sort()
        return usable[0][1]  # 返回最小的满足条件的缓存路径
    return None


def train_one_resolution(image_size, batch_size, epochs, config_overrides, degradation_type="block", use_sd="auto"):
    """Train PSR-Net at a given resolution and return results."""
    print(f"\n{'='*60}")
    print(f"Training at resolution {image_size}x{image_size} | Batch={batch_size} | Epochs={epochs}")
    print(f"{'='*60}")

    device = torch.device(DEVICE_STR)
    if device.type != "cuda":
        print("[WARN] No GPU detected, running on CPU — high-resolution experiments will be slow.")

    # Config
    config = ExperimentConfig(
        name=f"A4_res{image_size}",
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        train_samples=200,
        test_samples=20,
        lambda_sparse=0.1,
    )
    for k, v in (config_overrides or {}).items():
        if hasattr(config, k):
            setattr(config, k, v)

    config.device = DEVICE_STR
    config.output_dir = _current_dir

    # Use standard model (matching paper ~1M params)
    model = create_model(model_type="standard", base_channels=config.base_channels,
                         input_channels=config.input_channels, device=DEVICE_STR)

    n_params = get_model_info(model)["total_params"]
    print(f"Model: OverPaintNet (standard) | Parameters: {n_params:,}")

    # 解析 use_sd: "auto" → 自动检测缓存, True/False → 显式覆盖
    if use_sd == "auto":
        cache_path = _auto_detect_sd_cache(config.image_size, config.train_samples, config.seed)
        _use_sd = cache_path is not None
        if _use_sd:
            print(f"  [AUTO] SD cache detected: {os.path.basename(cache_path)} → using SD images")
        else:
            print(f"  [AUTO] No SD cache for size={config.image_size} n>={config.train_samples}, "
                  f"falling back to SyntheticDataset (random noise)")
    elif use_sd is True:
        _use_sd = True
        print(f"  [EXPLICIT] --use_sd forced")
    else:
        _use_sd = False
        print(f"  [EXPLICIT] --use_sd not set, using SyntheticDataset (random noise)")

    # Train/test datasets
    if _use_sd:
        train_ds = SDImageDataset(
            num_samples=config.train_samples,
            size=config.image_size,
            seed=config.seed,
            device=DEVICE_STR,
        )
        test_ds = SDImageDataset(
            num_samples=config.test_samples,
            size=config.image_size,
            seed=config.seed + 1000,
            device=DEVICE_STR,
        )
    else:
        train_ds = SyntheticDataset(
            num_samples=config.train_samples,
            size=config.image_size,
            seed=config.seed,
        )
        test_ds = SyntheticDataset(
            num_samples=config.test_samples,
            size=config.image_size,
            seed=config.seed + 1000,
        )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # Training
    t_train_start = time.time()
    engine = TrainingEngine(model, config, device=DEVICE_STR)
    history = engine.train(train_loader, test_loader,
                           save_dir=os.path.join(_current_dir, "outputs"),
                           verbose=True)
    t_train_total = time.time() - t_train_start

    # Inference performance
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    perf = measure_inference_performance(model, dummy_input, num_runs=50)

    # Evaluate on test set
    model.eval()
    all_psnr, all_ssim, all_l1, all_mask_mean, all_mask_contrast = [], [], [], [], []
    sample_dirty, sample_refined, sample_gt, sample_mask, sample_gtmask = [], [], [], [], []

    with torch.no_grad():
        for dirty, clean, gt_mask in test_loader:
            dirty, clean, gt_mask = dirty.to(device), clean.to(device), gt_mask.to(device)
            refined, residual, mask = model.refine(dirty)

            all_psnr.append(compute_psnr(refined, clean))
            all_ssim.append(compute_ssim(refined, clean))
            all_l1.append(F.l1_loss(refined, clean).item())
            all_mask_mean.append(mask.mean().item())
            all_mask_contrast.append(compute_mask_contrast_ratio(mask, gt_mask))

            # Collect samples for visualization (up to 3)
            if len(sample_dirty) < 3:
                sample_dirty.append(dirty[0].cpu())
                sample_refined.append(refined[0].cpu())
                sample_gt.append(clean[0].cpu())
                sample_mask.append(mask[0].cpu())
                sample_gtmask.append(gt_mask[0].cpu())

    results = {
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_params": n_params,
        "training_time_seconds": round(t_train_total, 2),
        "inference_time_ms": round(perf["inference_time_ms"], 4),
        "fps": round(perf["fps"], 2),
        "gpu_memory_mb": round(perf["gpu_memory_mb"], 2),
        "psnr_mean": round(float(np.mean(all_psnr)), 4),
        "psnr_std": round(float(np.std(all_psnr)), 4),
        "ssim_mean": round(float(np.mean(all_ssim)), 4),
        "ssim_std": round(float(np.std(all_ssim)), 4),
        "l1_loss_mean": round(float(np.mean(all_l1)), 6),
        "l1_loss_std": round(float(np.std(all_l1)), 6),
        "mask_mean": round(float(np.mean(all_mask_mean)), 6),
        "mask_contrast_ratio_mean": round(float(np.mean(sanitize_metric_array(all_mask_contrast))) if sanitize_metric_array(all_mask_contrast) else 0, 4),
        "mask_contrast_ratio_std": round(float(np.std(sanitize_metric_array(all_mask_contrast))) if sanitize_metric_array(all_mask_contrast) else 0, 4),
    }

    print(f"\nResolution {image_size} Results:")
    print(f"  PSNR={results['psnr_mean']:.2f} dB | SSIM={results['ssim_mean']:.4f}")
    print(f"  L1={results['l1_loss_mean']:.6f} | Mask Contrast={results['mask_contrast_ratio_mean']:.2f}")
    print(f"  Training: {results['training_time_seconds']:.0f}s | Inference: {results['inference_time_ms']:.2f}ms")
    print(f"  GPU Mem: {results['gpu_memory_mb']:.1f}MB | Params: {n_params:,}")

    return results, history, (sample_dirty, sample_refined, sample_gt, sample_mask, sample_gtmask)


def train_on_real_images(image_size, epochs, config_overrides=None, real_images_dir=None):
    """Train on real image patches with mixed degradations."""
    print(f"\n{'='*60}")
    print(f"Training on REAL images at {image_size}x{image_size} | Epochs={epochs}")
    print(f"{'='*60}")

    device = torch.device(DEVICE_STR)

    # 确定真实图片目录
    if real_images_dir is None:
        # 默认: resources/real_images/
        real_images_dir = os.path.join(os.path.dirname(_current_dir), "resources", "real_images")

    if not os.path.isdir(real_images_dir):
        print(f"[WARN] 真实图片目录不存在: {real_images_dir}")
        print(f"[WARN] 请将真实照片放入该目录，或通过 --real_images_dir 指定路径")
        print(f"[WARN] 支持格式: .jpg .jpeg .png .bmp .tiff")
        print(f"[WARN] 跳过真实图片训练。")
        return None, None

    all_images = load_real_images(real_images_dir, target_size=image_size, max_images=50)
    print(f"  Loaded {len(all_images)} real images from {real_images_dir}")

    if not all_images:
        print("[WARN] 目录中没有可加载的图片，跳过真实图片训练。")
        return None, None

    print(f"  Total real images: {len(all_images)}")
    batch_size = {64: 16, 128: 8, 256: 4, 512: 2}.get(image_size, 4)

    config = ExperimentConfig(
        name=f"A4_real_res{image_size}",
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        train_samples=200,
        test_samples=20,
        lambda_sparse=0.1,
    )
    for k, v in (config_overrides or {}).items():
        if hasattr(config, k):
            setattr(config, k, v)

    config.device = DEVICE_STR
    config.output_dir = _current_dir

    # Datasets
    train_ds = RealImageDataset(
        images=all_images,
        num_samples=config.train_samples,
        degradation_types=["block", "blur", "jpeg"],
        seed=config.seed,
    )
    test_ds = RealImageDataset(
        images=all_images,
        num_samples=config.test_samples,
        degradation_types=["block", "blur", "jpeg"],
        seed=config.seed + 1000,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    model = create_model(model_type="standard", base_channels=config.base_channels,
                         input_channels=config.input_channels, device=DEVICE_STR)
    n_params = get_model_info(model)["total_params"]
    print(f"Model: OverPaintNet (standard) | Parameters: {n_params:,}")

    engine = TrainingEngine(model, config, device=DEVICE_STR)
    history = engine.train(train_loader, test_loader,
                           save_dir=os.path.join(_current_dir, "outputs"),
                           verbose=True)

    # Evaluate
    model.eval()
    all_psnr, all_ssim, all_l1, all_mask_contrast = [], [], [], []

    with torch.no_grad():
        for dirty, clean, gt_mask in test_loader:
            dirty, clean, gt_mask = dirty.to(device), clean.to(device), gt_mask.to(device)
            refined, _, mask = model.refine(dirty)
            all_psnr.append(compute_psnr(refined, clean))
            all_ssim.append(compute_ssim(refined, clean))
            all_l1.append(F.l1_loss(refined, clean).item())
            all_mask_contrast.append(compute_mask_contrast_ratio(mask, gt_mask))

    results = {
        "dataset": "real",
        "image_size": image_size,
        "psnr_mean": round(float(np.mean(all_psnr)), 4),
        "psnr_std": round(float(np.std(all_psnr)), 4),
        "ssim_mean": round(float(np.mean(all_ssim)), 4),
        "l1_loss_mean": round(float(np.mean(all_l1)), 6),
        "mask_contrast_ratio_mean": round(float(np.mean(sanitize_metric_array(all_mask_contrast))) if sanitize_metric_array(all_mask_contrast) else 0, 4),
        "num_real_images": len(all_images),
    }
    print(f"\nReal {image_size} Results: PSNR={results['psnr_mean']:.2f} | SSIM={results['ssim_mean']:.4f}")
    return results, history


def generate_resolution_plots(all_results, output_dir):
    """Generate resolution comparison plots."""
    resolutions = sorted(set(r["image_size"] for r in all_results))

    def _safe_get(key, size):
        """安全获取指定分辨率的指标值，缺失时返回 0"""
        for r in all_results:
            if r["image_size"] == size and key in r:
                return r[key]
        return 0.0

    psnr_vals = [_safe_get("psnr_mean", s) for s in resolutions]
    ssim_vals = [_safe_get("ssim_mean", s) for s in resolutions]
    l1_vals = [_safe_get("l1_loss_mean", s) for s in resolutions]
    time_vals = [_safe_get("training_time_seconds", s) for s in resolutions]
    mc_vals = [_safe_get("mask_contrast_ratio_mean", s) for s in resolutions]
    fps_vals = [_safe_get("fps", s) for s in resolutions]

    # PSNR vs Resolution bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    colors = ["#2E75B6", "#6AA84F", "#E06666", "#F1C232"]

    # PSNR
    bars = axes[0, 0].bar([f"{s}x{s}" for s in resolutions], psnr_vals, color=colors[:len(resolutions)], edgecolor="white")
    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 0].set_title("PSNR vs Resolution")
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, psnr_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", fontsize=9)

    # Training time
    axes[0, 1].plot([f"{s}x{s}" for s in resolutions], time_vals, "o-", color="#C0504D",
                    linewidth=2, markersize=10)
    axes[0, 1].set_ylabel("Training Time (seconds)")
    axes[0, 1].set_title("Training Time vs Resolution")
    axes[0, 1].grid(True, alpha=0.3)
    for i, (s, t) in enumerate(zip(resolutions, time_vals)):
        axes[0, 1].annotate(f"{t:.0f}s", (f"{s}x{s}", t),
                            textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    # SSIM
    bars = axes[1, 0].bar([f"{s}x{s}" for s in resolutions], ssim_vals, color=colors[:len(resolutions)], edgecolor="white")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].set_title("SSIM vs Resolution")
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, ssim_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.4f}", ha="center", fontsize=9)

    # FPS
    bars = axes[1, 1].bar([f"{s}x{s}" for s in resolutions], fps_vals, color=colors[:len(resolutions)], edgecolor="white")
    axes[1, 1].set_ylabel("FPS")
    axes[1, 1].set_title("Inference Speed (FPS)")
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, fps_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", fontsize=9)

    plt.suptitle("PSR-Net High-Resolution Scaling Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "resolution_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved resolution comparison plot: {save_path}")

    # Combined metrics line chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1, color2, color3 = "#2E75B6", "#C0504D", "#6AA84F"
    x_labels = [f"{s}x{s}" for s in resolutions]
    x = range(len(resolutions))

    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("PSNR (dB)", color=color1)
    line1 = ax1.plot(x, psnr_vals, "o-", color=color1, linewidth=2, markersize=8, label="PSNR")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)

    ax2 = ax1.twinx()
    ax2.set_ylabel("SSIM / Mask Contrast", color=color2)
    line2 = ax2.plot(x, ssim_vals, "s--", color=color2, linewidth=2, markersize=8, label="SSIM")
    line3 = ax2.plot(x, mc_vals, "d:", color=color3, linewidth=2, markersize=8, label="Mask Contrast")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("PSR-Net Metrics vs Resolution")

    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "resolution_metrics_line.png")
    plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics line chart: {save_path2}")


def generate_visual_quality_grid(sample_data, output_dir, all_results=None):
    """Generate 4x4 visual quality comparison grid (rows: 64/128/256/512, cols: dirty/refined/mask/gt)."""
    from common.visualization import plot_results_grid

    all_dirty, all_refined, all_gt, all_mask, all_gtmask = [], [], [], [], []

    for res_samples in sample_data:
        dirty_l, refined_l, gt_l, mask_l, gtmask_l = res_samples
        for i in range(min(1, len(dirty_l))):
            all_dirty.append(dirty_l[i])
            all_refined.append(refined_l[i])
            all_gt.append(gt_l[i])
            all_mask.append(mask_l[i])
            all_gtmask.append(gtmask_l[i])

    if all_dirty:
        resolutions = sorted([r["image_size"] for r in all_results]) if all_results else [64, 128, 256, 512]
        titles = [f"{s}x{s}" for s in resolutions[:len(all_dirty)]]

        save_path = os.path.join(output_dir, "visual_quality_grid.png")
        plot_results_grid(
            all_dirty, all_refined, all_gt,
            mask_list=all_mask, gt_mask_list=all_gtmask,
            titles=titles, save_path=save_path,
            max_samples=len(all_dirty),
        )
        print(f"Saved visual quality grid: {save_path}")


def main():
    global DEVICE_STR

    parser = argparse.ArgumentParser(description="A4: High-Resolution Experiments")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per resolution")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--image_size", type=int, default=None, help="Run single resolution only (e.g. 256)")
    parser.add_argument("--skip_real", action="store_true", help="Skip real image training")
    parser.add_argument("--device", type=str, default=None,
                        help=f"Device override (default: {DEVICE_STR})")
    parser.add_argument("--use_sd", action="store_true", default=None,
                        help="Force use SD v1.5 generated images (default: auto-detect)")
    parser.add_argument("--no_sd", action="store_true",
                        help="Force use SyntheticDataset (random noise), skip SD auto-detect")
    parser.add_argument("--real_images_dir", type=str, default=None,
                        help="Path to real images directory (default: resources/real_images/)")
    args = parser.parse_args()

    # 解析 use_sd: --use_sd → True, --no_sd → False, 都不传 → "auto"
    if args.use_sd:
        _use_sd_flag = True
    elif args.no_sd:
        _use_sd_flag = False
    else:
        _use_sd_flag = "auto"

    if args.device:
        DEVICE_STR = args.device

    output_dir = os.path.join(_current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # 打印数据源配置
    sd_mode_str = {True: "ON (--use_sd)", False: "OFF (--no_sd)", "auto": "AUTO (detect cache)"}[_use_sd_flag]
    print(f"Data source: SD images = {sd_mode_str}")
    if args.real_images_dir:
        print(f"Real images dir: {args.real_images_dir}")
    else:
        print(f"Real images dir: {os.path.join(os.path.dirname(_current_dir), 'resources', 'real_images')} (default)")

    device = torch.device(DEVICE_STR)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Running on CPU — training will be significantly slower.")

    # Resolution configurations
    if args.image_size:
        configs = [(args.image_size, args.batch_size or max(2, 16 * 64 // args.image_size))]
    else:
        configs = [
            (64, 16),
            (128, 8),
            (256, 4),
            (512, 2),
        ]
        print("\n" + "="*60)
        print("PSR-Net High-Resolution Scaling Experiment")
        print(f"Resolutions: {[c[0] for c in configs]}")
        print(f"Device: {DEVICE_STR}")
        print("="*60)

    # Train at each resolution
    all_results = []
    all_histories = {}
    sample_data = []

    for image_size, batch_size in configs:
        bs = args.batch_size or batch_size
        results, history, samples = train_one_resolution(
            image_size=image_size,
            batch_size=bs,
            epochs=args.epochs,
            config_overrides={},
            use_sd=_use_sd_flag,
        )
        all_results.append(results)
        all_histories[f"res_{image_size}"] = history
        sample_data.append(samples)

        # Train on real images (skip >256 to save time unless on GPU)
        if not args.skip_real and image_size <= 256:
            real_results, real_history = train_on_real_images(
                image_size, args.epochs, real_images_dir=args.real_images_dir)
            if real_results:
                all_results.append(real_results)
                all_histories[f"real_{image_size}"] = real_history

    # Generate plots
    global all_results_ref
    all_results_ref = all_results
    synth_results = [r for r in all_results if "dataset" not in r]
    if synth_results:
        generate_resolution_plots(synth_results, output_dir)
        generate_visual_quality_grid(sample_data, output_dir, all_results=synth_results)

    # Save results JSON
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to: {results_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    header = f"{'Resolution':>10} | {'PSNR':>8} | {'SSIM':>8} | {'L1':>10} | {'MaskCont':>8} | {'Train(s)':>8} | {'FPS':>6} | {'Params':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        sz = r.get("image_size", "real")
        ds = r.get("dataset", "synth")
        label = f"{sz}{'R' if ds == 'real' else ''}"
        print(f"{label:>10} | {r.get('psnr_mean', 0):>8.2f} | {r.get('ssim_mean', 0):>8.4f} | "
              f"{r.get('l1_loss_mean', 0):>10.6f} | {r.get('mask_contrast_ratio_mean', 0):>8.2f} | "
              f"{r.get('training_time_seconds', 0):>8.0f} | {r.get('fps', 0):>6.1f} | "
              f"{r.get('num_params', 0):>10,}")

    print("\nA4 experiment complete!")


if __name__ == "__main__":
    main()
