"""
A8: Multi-Seed Training + Standard Deviation

Trains PSR-Net with 5 random seeds on block-occlusion synthetic data (64x64),
recording per-epoch metrics. Computes mean ± std across seeds and produces
training curves with error bands plus a final-metrics bar chart.

Outputs (A8/outputs/):
  - results.json       Per-seed and summary (mean/std/min/max) statistics
  - convergence_curves.png   Training curves with ±1σ shaded bands
  - final_metrics_bars.png   Bar chart with error bars
  - per_seed_convergence.png Per-seed comparison overlay
"""

import argparse
import json
import os
import sys
from collections import defaultdict

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
from common.evaluation import (
    compute_psnr, compute_ssim, compute_iou,
    compute_mask_contrast_ratio, sanitize_metric_array,
)
from common.training import CheckpointManager, _format_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===================================================================
# Single-seed training
# ===================================================================

def train_one_seed(seed, config, device, save_dir=None, DatasetClass=SyntheticDataset,
                   use_distill=False):
    """
    Train PSR-Net with a given random seed.  Returns the full per-epoch
    history and the trained model.
    """
    import time
    torch.manual_seed(seed)
    np.random.seed(seed)

    seed_save_dir = os.path.join(save_dir, f"seed_{seed}") if save_dir else None
    ckpt_mgr = CheckpointManager(seed_save_dir, keep_last=3) if seed_save_dir else None

    # SDImageDataset needs device; SyntheticDataset does not
    if DatasetClass == SDImageDataset:
        train_ds = DatasetClass(
            num_samples=config["train_samples"],
            size=config["image_size"],
            seed=seed,
            device=str(device),
        )
        test_ds = DatasetClass(
            num_samples=config["test_samples"],
            size=config["image_size"],
            seed=seed + 10000,
            device=str(device),
        )
    else:
        train_ds = DatasetClass(
            num_samples=config["train_samples"],
            size=config["image_size"],
            seed=seed,
        )
        test_ds = DatasetClass(
            num_samples=config["test_samples"],
            size=config["image_size"],
            seed=seed + 10000,
        )
    train_ldr = DataLoader(train_ds, config["batch_size"], shuffle=True)
    test_ldr = DataLoader(test_ds, config["batch_size"], shuffle=False)

    model = create_model("standard", base_channels=config["base_channels"], device=device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    lambda_s = config["lambda_sparse"]
    warmup = config["warmup_epochs"]
    epochs = config["epochs"]

    history = {
        "epoch": [], "seed": seed,
        "train_loss": [], "train_l1": [],
        "val_psnr": [], "val_ssim": [], "val_l1_loss": [],
        "val_mask_mean": [], "val_mask_contrast": [], "val_iou": [],
        "val_epoch": [],
    }

    best_psnr = -float("inf")
    best_state = None
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        lamb = lambda_s * (epoch / warmup) if epoch < warmup else lambda_s
        loss_sum, l1_sum = 0.0, 0.0
        for dirty, clean, gt_mask in train_ldr:
            dirty, clean, gt_mask = dirty.to(device), clean.to(device), gt_mask.to(device)
            optimizer.zero_grad()
            residual, mask = model(dirty)
            refined = dirty + residual * mask
            loss_l1_val = F.l1_loss(refined, clean)
            # Paper loss: L = L1 + λ_s * mean(M).  No GT mask in loss.
            loss = loss_l1_val + lamb * mask.mean()
            if use_distill:
                # Optional BCE distillation (off by default — not in paper)
                loss_distill = F.binary_cross_entropy(
                    mask.clamp(1e-6, 1-1e-6), gt_mask)
                loss = loss + loss_distill
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            l1_sum += loss_l1_val.item()
        scheduler.step()

        history["train_loss"].append(loss_sum / len(train_ldr))
        history["train_l1"].append(l1_sum / len(train_ldr))

        model.eval()
        val_psnr, val_ssim, val_l1, val_mm, val_mc, val_iou_l = [], [], [], [], [], []
        with torch.no_grad():
            for dirty, clean, gt_mask in test_ldr:
                dirty, clean, gt_mask = dirty.to(device), clean.to(device), gt_mask.to(device)
                refined, _res, mask = model.refine(dirty)
                val_psnr.append(compute_psnr(refined, clean))
                val_ssim.append(compute_ssim(refined, clean))
                val_l1.append(F.l1_loss(refined, clean).item())
                val_mm.append(mask.mean().item())
                val_mc.append(compute_mask_contrast_ratio(mask, gt_mask))
                val_iou_l.append(compute_iou(mask, gt_mask))

        ep_psnr = float(np.mean(val_psnr))
        history["val_psnr"].append(ep_psnr)
        history["val_ssim"].append(float(np.mean(val_ssim)))
        history["val_l1_loss"].append(float(np.mean(val_l1)))
        history["val_mask_mean"].append(float(np.mean(val_mm)))
        history["val_mask_contrast"].append(float(np.mean(sanitize_metric_array(val_mc))) if sanitize_metric_array(val_mc) else 0.0)
        history["val_iou"].append(float(np.mean(val_iou_l)))
        history["epoch"].append(epoch)

        if ep_psnr > best_psnr:
            best_psnr = ep_psnr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # 进度 + 定期保存
        if epoch % 5 == 0 or epoch == epochs - 1:
            elapsed = time.time() - train_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0
            print(f"  [Seed {seed}] Epoch {epoch:3d}/{epochs} [{100*(epoch+1)//epochs}%] "
                  f"| PSNR={ep_psnr:.1f} | IoU={history['val_iou'][-1]:.4f} "
                  f"| ETA={_format_time(eta)}")
        if ckpt_mgr and (epoch % 10 == 0 or epoch == epochs - 1):
            ckpt_mgr.save(model, optimizer, epoch, history,
                         {"psnr": ep_psnr, "ssim": history["val_ssim"][-1], "iou": history["val_iou"][-1]})

    if best_state:
        model.load_state_dict(best_state)

    return model, history


# ===================================================================
# Plotting helpers
# ===================================================================

def plot_convergence_curves(all_histories, seeds, out_dir):
    """
    For each metric, plot mean ± std shaded bands across seeds.
    """
    metrics = {
        "val_psnr": ("PSNR (dB)", "convergence_psnr.png"),
        "val_ssim": ("SSIM", "convergence_ssim.png"),
        "val_mask_mean": ("Mask Mean (Sparsity)", "convergence_mask_mean.png"),
        "val_mask_contrast": ("Mask Contrast Ratio", "convergence_mask_contrast.png"),
        "val_iou": ("IoU", "convergence_iou.png"),
        "train_loss": ("Training Loss", "convergence_loss.png"),
    }

    # Gather per-epoch curves into aligned arrays
    max_epochs = max(len(h["epoch"]) for h in all_histories.values())
    curves = {}
    for metric in metrics:
        matrix = []
        for sd in seeds:
            key = f"seed_{sd}"
            vals = all_histories[key].get(metric, [])
            # Pad or truncate to max_epochs
            if len(vals) < max_epochs:
                vals = vals + [vals[-1]] * (max_epochs - len(vals))
            else:
                vals = vals[:max_epochs]
            matrix.append(vals)
        arr = np.array(matrix)  # (n_seeds, n_epochs)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        curves[metric] = (mean, std)

    # Plot each metric separately
    for metric, (ylabel, fname) in metrics.items():
        mean, std = curves[metric]
        x = np.arange(len(mean))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, mean, color="#2E75B6", linewidth=2, label="Mean")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#2E75B6", label="±1σ")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} — Mean ± Std (n={len(seeds)})", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    # ---- Combined figure with 6 subplots ----
    metric_list = list(metrics.items())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (metric, (ylabel, _)) in enumerate(metric_list):
        ax = axes[idx // 3][idx % 3]
        mean, std = curves[metric]
        x = np.arange(len(mean))
        ax.plot(x, mean, color="#2E75B6", linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color="#2E75B6")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
    plt.suptitle("PSR-Net Convergence Curves with ±1σ Error Bands", fontsize=14)
    plt.tight_layout()
    combo_path = os.path.join(out_dir, "convergence_curves.png")
    plt.savefig(combo_path, dpi=150)
    plt.close()
    print(f"Convergence curves saved to {combo_path}")

    return curves


def plot_final_metrics_bars(summary_stats, out_dir):
    """Bar chart of mean final metrics with error bars (std)."""
    metrics_display = {
        "val_psnr": "PSNR (dB)",
        "val_ssim": "SSIM",
        "val_mask_mean": "Mask Mean",
        "val_mask_contrast": "Mask Contrast",
        "val_iou": "IoU",
        "val_l1_loss": "L1 Loss",
    }

    names = []
    means = []
    stds = []
    for key, label in metrics_display.items():
        mk = f"{key}_mean"
        sk = f"{key}_std"
        if mk in summary_stats:
            names.append(label)
            means.append(summary_stats[mk])
            stds.append(summary_stats.get(sk, 0))

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor="black",
                  capsize=8, error_kw={"linewidth": 1.5})
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("PSR-Net Final Metrics — Mean ± Std (5 seeds)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean, std in zip(bars, means, stds):
        if abs(mean) < 10:
            label = f"{mean:.3f}±{std:.3f}"
        else:
            label = f"{mean:.2f}±{std:.2f}"
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + max(means)*0.01,
                label, ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "final_metrics_bars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Final metrics bars saved to {path}")


def plot_per_seed_comparison(all_histories, seeds, out_dir):
    """Overlay per-seed PSNR curves for convergence comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    for sd, color in zip(seeds, colors):
        hist = all_histories[f"seed_{sd}"]
        epochs = hist.get("epoch", list(range(len(hist["val_psnr"]))))
        ax.plot(epochs, hist["val_psnr"], color=color, linewidth=1.5, alpha=0.8, label=f"seed={sd}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-Seed Convergence Comparison (PSNR)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_seed_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Per-seed convergence saved to {path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="A8: Multi-Seed Training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_sparse", type=float, default=0.1,
                        help="Sparsity coefficient (paper optimal: 0.1)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    # Add --use_sd flag
    parser.add_argument("--use_sd", action="store_true",
                        help="Use SD v1.5 generated images")
    parser.add_argument("--use_distill", action="store_true", default=False,
                        help="Enable BCE mask distillation (NOT in paper; paper uses "
                             "self-supervision without GT mask in loss)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "lr": args.lr,
        "lambda_sparse": args.lambda_sparse,
        "warmup_epochs": args.epochs // 2,
        "base_channels": 64,
    }

    seeds = args.seeds
    all_histories = {}
    all_final = {}
    DatasetClass = SDImageDataset if args.use_sd else SyntheticDataset

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"  Training with seed={seed}")
        print(f"{'='*50}")

        model, history = train_one_seed(seed, config, device,
                                        save_dir=out_dir,
                                        DatasetClass=DatasetClass,
                                        use_distill=args.use_distill)
        all_histories[f"seed_{seed}"] = history

        final_metrics = {
            "val_psnr": history["val_psnr"][-1],
            "val_ssim": history["val_ssim"][-1],
            "val_l1_loss": history["val_l1_loss"][-1],
            "val_mask_mean": history["val_mask_mean"][-1],
            "val_mask_contrast": history["val_mask_contrast"][-1],
            "val_iou": history["val_iou"][-1],
            "params": int(count_parameters(model)),
        }
        all_final[f"seed_{seed}"] = final_metrics
        print(f"  Final: PSNR={final_metrics['val_psnr']:.2f}  "
              f"SSIM={final_metrics['val_ssim']:.4f}  "
              f"IoU={final_metrics['val_iou']:.4f}  "
              f"Mask Mean={final_metrics['val_mask_mean']:.4f}")

    # ---- Compute summary statistics ----
    summary = {}
    metric_keys = list(all_final[f"seed_{seeds[0]}"].keys())
    for mk in metric_keys:
        vals = [all_final[f"seed_{s}"][mk] for s in seeds]
        vals_arr = np.array(vals)
        summary[f"{mk}_mean"] = float(np.mean(vals_arr))
        summary[f"{mk}_std"] = float(np.std(vals_arr))
        summary[f"{mk}_min"] = float(np.min(vals_arr))
        summary[f"{mk}_max"] = float(np.max(vals_arr))

    # ---- Save JSON ----
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "seeds": seeds,
            "config": {k: str(v) if isinstance(v, type) else v for k, v in config.items()},
            "per_seed_final": all_final,
            "summary": summary,
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ---- Plots ----
    plot_convergence_curves(all_histories, seeds, out_dir)
    plot_final_metrics_bars(summary, out_dir)
    plot_per_seed_comparison(all_histories, seeds, out_dir)

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print("  MULTI-SEED SUMMARY TABLE")
    print("=" * 80)
    header = (f"{'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(header)
    print("-" * 80)

    display_names = {
        "val_psnr": "PSNR (dB)",
        "val_ssim": "SSIM",
        "val_l1_loss": "L1 Loss",
        "val_mask_mean": "Mask Mean",
        "val_mask_contrast": "Mask Contrast",
        "val_iou": "IoU",
        "params": "Parameters",
    }

    for mk, label in display_names.items():
        if mk == "params":
            print(f"{label:<22} {summary.get('params_mean',0):10.0f} "
                  f"{summary.get('params_std',0):10.0f} "
                  f"{summary.get('params_min',0):10.0f} "
                  f"{summary.get('params_max',0):10.0f}")
        elif f"{mk}_mean" in summary:
            print(f"{label:<22} {summary[f'{mk}_mean']:10.4f} "
                  f"{summary[f'{mk}_std']:10.4f} "
                  f"{summary[f'{mk}_min']:10.4f} "
                  f"{summary[f'{mk}_max']:10.4f}")
    print("=" * 80)

    print("\nA8 experiment complete!")


if __name__ == "__main__":
    main()
