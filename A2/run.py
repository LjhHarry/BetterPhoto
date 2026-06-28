"""
Experiment A2: Iterative Selective Redrawing (2-3 rounds)

Validates iterative refinement: I^(k+1) = I^(k) + R^(k+1) * M^(k+1).
After training PSR-Net on block occlusion data, applies 3 rounds of progressive refinement.
Tracks per-round mask mean (should decrease), PSNR, SSIM, and mask contrast ratio.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- path setup ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import get_config
from common.model_factory import create_model
from common.evaluation import (compute_psnr, compute_ssim, compute_mask_contrast_ratio,
                                evaluate_all, format_results_table, sanitize_metric_array)
from common.data_utils import (SyntheticDataset, generate_synthetic_pair,
                              generate_batch_synthetic, SDImageDataset)
from common.visualization import tensor_to_numpy, plot_iterative_refinement
from common.training import TrainingEngine


def iterative_refine(model, dirty, num_rounds=3, device="cpu"):
    """
    Apply iterative refinement.
    Returns lists: refined_images, masks, per_round_metrics
    """
    model.eval()
    refined_images = []
    masks = []
    per_round_metrics = []

    current = dirty.clone().to(device)

    for round_idx in range(num_rounds):
        with torch.no_grad():
            refined, residual, mask = model.refine(current)

        refined_images.append(refined)
        masks.append(mask)

        # Per-round metrics (per sample)
        round_metrics = {
            "mask_mean": float(mask.mean().item()),
        }
        per_round_metrics.append(round_metrics)

        # Next round input is current refined image
        current = refined.clone()

    return refined_images, masks, per_round_metrics


def evaluate_iterative_rounds(model, test_loader, gt_list, dirty_list, num_rounds=3, device="cpu"):
    """
    Evaluate metrics across all rounds for the entire test set.
    Returns per-round aggregate metrics.
    """
    model.eval()
    all_rounds_metrics = [{"psnr": [], "ssim": [], "l1_loss": [], "mask_mean": [],
                            "mask_contrast": []} for _ in range(num_rounds)]

    with torch.no_grad():
        for batch in test_loader:
            dirty, clean, gt_mask = [b.to(device) for b in batch]

            current = dirty.clone()
            for r in range(num_rounds):
                refined, _, mask = model.refine(current)

                for i in range(len(refined)):
                    all_rounds_metrics[r]["psnr"].append(compute_psnr(refined[i:i+1], clean[i:i+1]))
                    all_rounds_metrics[r]["ssim"].append(compute_ssim(refined[i:i+1], clean[i:i+1]))
                    all_rounds_metrics[r]["l1_loss"].append(F.l1_loss(refined[i:i+1], clean[i:i+1]).item())
                    all_rounds_metrics[r]["mask_mean"].append(mask[i].mean().item())
                    all_rounds_metrics[r]["mask_contrast"].append(
                        compute_mask_contrast_ratio(mask[i:i+1], gt_mask[i:i+1]))

                current = refined.clone()

    # Aggregate
    round_summary = []
    for r, metrics in enumerate(all_rounds_metrics):
        summary = {}
        for k, v in metrics.items():
            clean_v = sanitize_metric_array(v)
            if clean_v:
                summary[f"{k}_mean"] = float(np.mean(clean_v))
                summary[f"{k}_std"] = float(np.std(clean_v))
            else:
                summary[f"{k}_mean"] = 0.0
                summary[f"{k}_std"] = 0.0
        round_summary.append(summary)

    return round_summary


def plot_round_comparison(rounds_metrics, save_path):
    """
    Bar chart comparing 1-round, 2-round, 3-round PSNR and SSIM.
    Also shows mask mean decrease curve.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    num_rounds = len(rounds_metrics)
    labels = [f"Round {i+1}" for i in range(num_rounds)]

    # PSNR
    psnr_vals = [rounds_metrics[r]["psnr_mean"] for r in range(num_rounds)]
    colors = ["#2E75B6", "#5B9BD5", "#9DC3E6"]
    bars = axes[0].bar(labels, psnr_vals, color=colors[:num_rounds], edgecolor="white")
    max_psnr = max(psnr_vals)
    axes[0].set_ylim(max_psnr - 5, max_psnr + 2)
    for bar, val in zip(bars, psnr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.2f}", ha="center", fontsize=9)
    axes[0].set_title("PSNR (dB) per Round", fontsize=11)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # SSIM
    ssim_vals = [rounds_metrics[r]["ssim_mean"] for r in range(num_rounds)]
    bars = axes[1].bar(labels, ssim_vals, color=colors[:num_rounds], edgecolor="white")
    for bar, val in zip(bars, ssim_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", fontsize=9)
    axes[1].set_title("SSIM per Round", fontsize=11)
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Mask Mean (should decrease)
    mask_vals = [rounds_metrics[r]["mask_mean_mean"] for r in range(num_rounds)]
    axes[2].plot(labels, mask_vals, "o-", color="#C0504D", linewidth=2, markersize=8)
    for i, (lbl, val) in enumerate(zip(labels, mask_vals)):
        axes[2].annotate(f"{val:.6f}", (lbl, val), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=9)
    axes[2].set_title("Mask Mean (Sparsity) per Round", fontsize=11)
    axes[2].set_ylabel("Mask Mean")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("A2: Iterative Selective Redrawing — Round Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="A2: Iterative Selective Redrawing")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of refinement rounds")
    parser.add_argument("--use_sd", action="store_true",
                        help="Use SD v1.5 generated images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    config = get_config("A2",
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        lambda_sparse=0.1)
    config.output_dir = output_dir
    config.device = device.__str__()

    print(f"\n{'='*60}")
    print(f"  Experiment A2: Iterative Selective Redrawing")
    print(f"  Image size: {config.image_size}, Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}, Rounds: {args.num_rounds}")
    print(f"{'='*60}")

    # =====================================================================
    # Step 1: Train standard PSR-Net on block occlusion data
    # =====================================================================
    print("\n[Step 1] Training standard PSR-Net on block occlusion data...")
    t0 = time.time()

    DatasetClass = SDImageDataset if args.use_sd else SyntheticDataset
    if args.use_sd:
        train_dataset = DatasetClass(
            num_samples=config.train_samples, size=config.image_size,
            seed=config.seed, device=device.__str__(),
        )
        test_dataset = DatasetClass(
            num_samples=config.test_samples, size=config.image_size,
            seed=config.seed + 1000, device=device.__str__(),
        )
    else:
        train_dataset = DatasetClass(
            num_samples=config.train_samples, size=config.image_size,
            num_defects=3, defect_size=8, seed=config.seed,
        )
        # Harder test set: 6 defects (2x training) so 1 round can't fully repair,
        # giving iterative rounds 2/3 room to show progressive refinement.
        test_dataset = DatasetClass(
            num_samples=config.test_samples, size=config.image_size,
            num_defects=6, defect_size=10, seed=config.seed + 1000,
        )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = create_model("standard", base_channels=config.base_channels,
                         input_channels=config.input_channels, device=device)
    engine = TrainingEngine(model, config, device)
    history = engine.train(train_loader, test_loader, verbose=True,
                           val_freq=max(1, config.epochs // 10),
                           save_dir=output_dir)

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s")

    # =====================================================================
    # Step 2: Single-round baseline evaluation
    # =====================================================================
    print("\n[Step 2] Single-round baseline evaluation...")
    model.eval()
    single_dirty, single_clean, single_gtmask = [], [], []
    single_refined, single_masks = [], []

    with torch.no_grad():
        for batch in test_loader:
            dirty, clean, gt_mask = [b.to(device) for b in batch]
            refined, _, mask = model.refine(dirty)
            single_refined.append(refined)
            single_dirty.append(dirty)
            single_clean.append(clean)
            single_masks.append(mask)
            single_gtmask.append(gt_mask)

    single_ref = torch.cat(single_refined, dim=0)
    single_d = torch.cat(single_dirty, dim=0)
    single_c = torch.cat(single_clean, dim=0)
    single_m = torch.cat(single_masks, dim=0)
    single_gm = torch.cat(single_gtmask, dim=0)

    single_metrics = evaluate_all(
        [single_ref[i] for i in range(len(single_ref))],
        [single_c[i] for i in range(len(single_c))],
        dirty_list=[single_d[i] for i in range(len(single_d))],
        masks=[single_m[i] for i in range(len(single_m))],
        gt_masks=[single_gm[i] for i in range(len(single_gm))],
        metrics=["psnr", "ssim", "l1", "mask_contrast", "mask_mean"],
    )

    # =====================================================================
    # Step 3: Multi-round iterative refinement evaluation
    # =====================================================================
    print(f"\n[Step 3] Multi-round iterative refinement ({args.num_rounds} rounds)...")

    rounds_metrics = evaluate_iterative_rounds(
        model, test_loader, single_c, single_d,
        num_rounds=args.num_rounds, device=device.__str__())

    for r, rm in enumerate(rounds_metrics):
        print(f"  Round {r+1}: PSNR={rm['psnr_mean']:.2f}, SSIM={rm['ssim_mean']:.4f}, "
              f"MaskMean={rm['mask_mean_mean']:.6f}, Contrast={rm.get('mask_contrast_mean', 0):.2f}")

    # =====================================================================
    # Step 4: Generate iterative refinement visualization on a single sample
    # =====================================================================
    print("\n[Step 4] Generating visualizations...")

    # Pick one sample from test set
    sample_dirty, sample_clean, sample_gtmask = test_dataset[0]
    sample_dirty = sample_dirty.unsqueeze(0).to(device)
    sample_clean_t = sample_clean.unsqueeze(0)

    refined_images, masks, _ = iterative_refine(
        model, sample_dirty, num_rounds=args.num_rounds, device=device.__str__())

    # Iterative process visualization (dirty → I1 → I2 → I3 → GT + masks)
    iter_viz_path = os.path.join(output_dir, "iterative_refinement.png")
    plot_iterative_refinement(
        sample_dirty.squeeze(0).cpu(),
        [r.squeeze(0).cpu() for r in refined_images],
        [m.squeeze(0).cpu() for m in masks],
        sample_clean_t.squeeze(0),
        save_path=iter_viz_path,
    )
    print(f"  Saved: {iter_viz_path}")

    # Round comparison bar chart
    round_cmp_path = os.path.join(output_dir, "round_comparison.png")
    # Add single-round metrics as "Round 1" for comparison
    plot_round_comparison(rounds_metrics, round_cmp_path)
    print(f"  Saved: {round_cmp_path}")

    # =====================================================================
    # Step 5: Save results
    # =====================================================================
    print("\n[Step 5] Saving results...")

    results = {
        "experiment": "A2",
        "description": "Iterative Selective Redrawing",
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "lambda_sparse": config.lambda_sparse,
            "num_rounds": args.num_rounds,
            "device": device.__str__(),
        },
        "single_round_baseline": {k: round(float(v), 6) for k, v in single_metrics.items()},
        "iterative_rounds": [],
    }

    for r, rm in enumerate(rounds_metrics):
        clean_rm = {}
        for k, v in rm.items():
            if isinstance(v, (np.floating, float)):
                val = float(v)
                clean_rm[k] = round(val, 6) if not (np.isinf(val) or np.isnan(val)) else None
            else:
                clean_rm[k] = v
        clean_rm["round"] = r + 1
        results["iterative_rounds"].append(clean_rm)

    # Add delta analysis
    if len(rounds_metrics) >= 2:
        delta_psnr = rounds_metrics[-1]["psnr_mean"] - rounds_metrics[0]["psnr_mean"]
        delta_ssim = rounds_metrics[-1]["ssim_mean"] - rounds_metrics[0]["ssim_mean"]
        delta_mask = rounds_metrics[0]["mask_mean_mean"] - rounds_metrics[-1]["mask_mean_mean"]
        results["delta_analysis"] = {
            "psnr_improvement": round(float(delta_psnr), 4),
            "ssim_improvement": round(float(delta_ssim), 4),
            "mask_sparsity_increase": round(float(delta_mask), 6),
        }

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_json_path}")

    # =====================================================================
    # Summary Table
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  A2 Results Summary")
    print(f"{'='*80}")
    header = f"{'Method':20s} {'PSNR(dB)':>10s} {'SSIM':>8s} {'MaskMean':>12s} {'Contrast':>10s}"
    print(header)
    print("-" * 80)
    print(f"  {'Single-Round':18s} {single_metrics.get('psnr_mean', 0):>10.2f} "
          f"{single_metrics.get('ssim_mean', 0):>8.4f} {single_metrics.get('mask_mean_mean', 0):>12.6f} "
          f"{single_metrics.get('mask_contrast_ratio_mean', 0):>10.2f}")
    for r, rm in enumerate(rounds_metrics):
        print(f"  {f'Round {r+1}':18s} {rm['psnr_mean']:>10.2f} {rm['ssim_mean']:>8.4f} "
              f"{rm['mask_mean_mean']:>12.6f} {rm.get('mask_contrast_mean', 0):>10.2f}")
    if len(rounds_metrics) >= 2:
        print("-" * 80)
        print(f"  {'Delta (R1→R{len(rounds_metrics)})':18s} {delta_psnr:>10.2f} {delta_ssim:>8.4f} "
              f"{delta_mask:>12.6f}")
    print(f"{'='*80}")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Results JSON: {results_json_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
