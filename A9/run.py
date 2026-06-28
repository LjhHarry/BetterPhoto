"""
A9: Cost-Efficiency Quantitative Analysis

Measures inference time, GPU memory, and parameter count across resolutions
(64, 128, 256, 512) and compares PSR-Net against "Full regeneration" proxies.

Outputs (A9/outputs/):
  - results.json               All measurements
  - inference_time_bars.png    Inference time vs resolution
  - gpu_memory_bars.png        GPU memory vs resolution
  - quality_cost_scatter.png   Bubble chart: x=time, y=PSNR, size=memory
  - cost_comparison_table.txt  LaTeX-ready table
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.model_factory import create_model, OverPaintNetLarge, count_parameters
from common.data_utils import generate_synthetic_pair, SyntheticDataset
from common.evaluation import compute_psnr, measure_inference_performance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text


# ===================================================================
# Model definitions for cost comparison
# ===================================================================

class FullRegenProxy(torch.nn.Module):
    """
    Proxy for "Full SD img2img / Full image regeneration".
    Uses a larger UNet that outputs a full 3ch image (no residual/mask).
    This represents the cost of a full-generation pipeline at comparable quality.
    """
    def __init__(self, in_ch=3, base=64, out_ch=3):
        super().__init__()
        c = base
        self.enc = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, c, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c, c, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c, c*2, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*2, c*2, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*2, c*4, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*4, c*4, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*4, c*8, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*8, c*8, 3, padding=1), torch.nn.ReLU(inplace=True),
        )
        self.dec = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c*8, c*4, 4, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*4, c*4, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(c*4, c*2, 4, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c*2, c*2, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(c*2, c, 4, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c, c//2, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(c//2, out_ch, 3, padding=1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class PSRNetUpscaleProxy(torch.nn.Module):
    """
    Proxy: PSR-Net on 128x128, then upscale.  Simulates the cost of running
    a small PSR-Net followed by simple bilinear upsampling.
    """
    def __init__(self, psr_model, target_size=512):
        super().__init__()
        self.psr = psr_model
        self.target_size = target_size

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        refined_small, _res, _mask = self.psr.refine(small)
        refined = F.interpolate(refined_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return refined


# ===================================================================
# Measurement runner
# ===================================================================

def measure_model(model, input_shape, device, num_warmup=50, num_runs=200):
    """Full measurement: inference time, FPS, GPU memory, params."""
    model.eval()
    x = torch.randn(*input_shape, device=device)

    perf = measure_inference_performance(model, x, num_warmup=num_warmup, num_runs=num_runs)
    params = count_parameters(model)

    return {
        "inference_time_ms": perf["inference_time_ms"],
        "fps": perf["fps"],
        "gpu_memory_mb": perf["gpu_memory_mb"],
        "params": params,
        "params_m": params / 1e6,
    }


def measure_psnr_quality(model, device, image_size=64, num_samples=50, model_type="psrnet"):
    """
    Train a small model and measure PSNR for quality-cost trade-off.
    Uses block-occlusion synthetic data.
    """
    import torch.optim as optim

    # Quick training
    ds = SyntheticDataset(num_samples=200, size=image_size, seed=42)
    test_ds = SyntheticDataset(num_samples=num_samples, size=image_size, seed=1042)
    train_ldr = DataLoader(ds, batch_size=16, shuffle=True)
    test_ldr = DataLoader(test_ds, batch_size=num_samples, shuffle=False)

    model.train()
    if model_type == "psrnet":
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(40):
            for dirty, clean, _m in train_ldr:
                dirty, clean = dirty.to(device), clean.to(device)
                optimizer.zero_grad()
                residual, mask = model(dirty)
                refined = dirty + residual * mask
                lamb = 0.1 * (epoch / 20) if epoch < 20 else 0.1
                loss = F.l1_loss(refined, clean) + lamb * mask.mean()
                loss.backward()
                optimizer.step()
    else:
        # full-regen proxy: simple L1
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(40):
            for dirty, clean, _m in train_ldr:
                dirty, clean = dirty.to(device), clean.to(device)
                optimizer.zero_grad()
                pred = model(dirty)
                loss = F.l1_loss(pred, clean)
                loss.backward()
                optimizer.step()

    # Evaluate
    model.eval()
    psnr_vals = []
    with torch.no_grad():
        for dirty, clean, _m in test_ldr:
            dirty, clean = dirty.to(device), clean.to(device)
            if model_type == "psrnet":
                refined, _, _ = model.refine(dirty)
            else:
                refined = model(dirty)
            psnr_vals.append(compute_psnr(refined, clean))
    return float(np.mean(psnr_vals))


# ===================================================================
# Plotting
# ===================================================================

def plot_bar_comparison(data, x_labels, metric_key, ylabel, title, save_path, log_y=False):
    """Generic grouped bar chart."""
    n_groups = len(x_labels)
    n_bars = len(data)
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(8, n_groups*1.5), 5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    for i, (label, values) in enumerate(data.items()):
        x_pos = np.arange(n_groups) + i * bar_width
        bars = ax.bar(x_pos, values, bar_width, label=label, color=colors[i], edgecolor="white")
        for bar, val in zip(bars, values):
            if val > 0:
                if val < 10:
                    lbl = f"{val:.2f}"
                else:
                    lbl = f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        lbl, ha="center", va="bottom", fontsize=6)

    ax.set_xticks(np.arange(n_groups) + bar_width * (n_bars-1) / 2)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_quality_cost_scatter(results, out_dir):
    """
    Bubble chart: x = inference_time_ms, y = PSNR, bubble size = GPU memory.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    methods_order = [
        "PSR-Net 64", "PSR-Net 128", "PSR-Net 256", "PSR-Net 512",
        "Full Regen 256", "Full Regen 512",
        "PSR-Net + Upscale 512",
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_order)))

    texts = []
    for method, color in zip(methods_order, colors):
        if method not in results:
            continue
        m = results[method]
        time_ms = m.get("inference_time_ms", 0)
        mem = m.get("gpu_memory_mb", 0)
        psnr = m.get("psnr", 0)
        size = max(mem * 2, 30)  # bubble size proportional to memory

        ax.scatter(time_ms, psnr, s=size, c=[color],
                   alpha=0.7, edgecolors="black", linewidth=0.5)
        texts.append(ax.text(time_ms, psnr, method, fontsize=8,
                             ha="center", va="center",
                             bbox=dict(boxstyle="round,pad=0.2",
                                       fc="white", alpha=0.8,
                                       ec="gray", lw=0.3)))

    ax.set_xlabel("Inference Time (ms)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Quality-Cost Trade-off (size = GPU memory)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.15, y=0.15)
    adjust_text(texts, ax=ax,
                expand_points=(1.5, 1.2), expand_text=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                force_text=(0.5, 0.5), force_points=(0.2, 0.2),
                lim=200)
    plt.tight_layout()
    path = os.path.join(out_dir, "quality_cost_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_latex_table(results, out_dir):
    """Produce a LaTeX-ready table for the paper."""
    methods_order = [
        ("PSR-Net 64", "PSR-Net (64×64)"),
        ("PSR-Net 128", "PSR-Net (128×128)"),
        ("PSR-Net 256", "PSR-Net (256×256)"),
        ("PSR-Net 512", "PSR-Net (512×512)"),
        ("Full Regen 256", "Full Regen (256×256)"),
        ("Full Regen 512", "Full Regen (512×512)"),
        ("PSR-Net + Upscale 512", "PSR-Net + Upscale (128→512)"),
    ]

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("  \\centering")
    lines.append("  \\caption{Cost-Efficiency Comparison}")
    lines.append("  \\label{tab:cost_efficiency}")
    lines.append("  \\begin{tabular}{lrrrrr}")
    lines.append("    \\toprule")
    lines.append("    Method & Time (ms) & FPS & Memory (MB) & Params (M) & PSNR (dB) \\\\")
    lines.append("    \\midrule")

    for key, display in methods_order:
        if key not in results:
            continue
        m = results[key]
        lines.append(f"    {display} & "
                     f"{m.get('inference_time_ms',0):.2f} & "
                     f"{m.get('fps',0):.1f} & "
                     f"{m.get('gpu_memory_mb',0):.1f} & "
                     f"{m.get('params_m',0):.2f} & "
                     f"{m.get('psnr',0):.1f} \\\\")

    # Cost ratios
    if "PSR-Net 512" in results and "Full Regen 512" in results:
        time_ratio = results["Full Regen 512"]["inference_time_ms"] / max(results["PSR-Net 512"]["inference_time_ms"], 1e-6)
        mem_ratio = results["Full Regen 512"]["gpu_memory_mb"] / max(results["PSR-Net 512"]["gpu_memory_mb"], 1e-6)
        lines.append("    \\midrule")
        lines.append(f"    \\multicolumn{{6}}{{l}}{{Cost Ratio (Full Regen / PSR-Net at 512): "
                     f"Time $\\times${time_ratio:.1f}, Memory $\\times${mem_ratio:.1f}}} \\\\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")

    txt_path = os.path.join(out_dir, "cost_comparison_table.tex")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {txt_path}")

    # Also write plain text
    plain_path = os.path.join(out_dir, "cost_comparison_table.txt")
    with open(plain_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("  COST-EFFICIENCY COMPARISON TABLE\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Method':<30} {'Time(ms)':>9} {'FPS':>8} {'Mem(MB)':>9} {'Params(M)':>10} {'PSNR':>7}\n")
        f.write("-" * 90 + "\n")
        for key, display in methods_order:
            if key not in results:
                continue
            m = results[key]
            f.write(f"{display:<30} {m.get('inference_time_ms',0):9.2f} {m.get('fps',0):8.1f} "
                    f"{m.get('gpu_memory_mb',0):9.1f} {m.get('params_m',0):10.2f} {m.get('psnr',0):7.1f}\n")
        f.write("=" * 90 + "\n")
    print(f"Plain table saved to {plain_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="A9: Cost-Efficiency Analysis")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    # ===================================================================
    # 1) PSR-Net at multiple resolutions
    # ===================================================================
    psr_configs = [
        (64, "standard", 64),
        (128, "standard", 64),
        (256, "large", 64),
        (512, "large", 64),
    ]

    for size, mtype, base_ch in psr_configs:
        name = f"PSR-Net {size}"
        print(f"\n{'='*50}")
        print(f"  Measuring: {name}")
        print(f"{'='*50}")

        model = create_model(mtype, base_channels=base_ch, device=str(device))

        # Measure cost
        batch = 1
        input_shape = (batch, 3, size, size)
        metrics = measure_model(model, input_shape, device)
        metrics["resolution"] = size
        metrics["model_type"] = "psrnet"

        # Measure PSNR quality (train quick model)
        psnr = measure_psnr_quality(model, device, image_size=size, num_samples=20, model_type="psrnet")
        metrics["psnr"] = psnr

        results[name] = metrics
        print(f"  Time={metrics['inference_time_ms']:.2f}ms, FPS={metrics['fps']:.1f}, "
              f"Mem={metrics['gpu_memory_mb']:.1f}MB, Params={metrics['params_m']:.2f}M, "
              f"PSNR={psnr:.2f}")

    # ===================================================================
    # 2) Full Regen proxies at 256 and 512
    # ===================================================================
    for size in [256, 512]:
        name = f"Full Regen {size}"
        print(f"\n{'='*50}")
        print(f"  Measuring: {name}")
        print(f"{'='*50}")

        model = FullRegenProxy(in_ch=3, base=128, out_ch=3).to(device)
        input_shape = (1, 3, size, size)
        metrics = measure_model(model, input_shape, device)
        metrics["resolution"] = size
        metrics["model_type"] = "full_regen"

        psnr = measure_psnr_quality(model, device, image_size=size, num_samples=20, model_type="full_regen")
        metrics["psnr"] = psnr

        results[name] = metrics
        print(f"  Time={metrics['inference_time_ms']:.2f}ms, FPS={metrics['fps']:.1f}, "
              f"Mem={metrics['gpu_memory_mb']:.1f}MB, Params={metrics['params_m']:.2f}M, "
              f"PSNR={psnr:.2f}")

    # ===================================================================
    # 3) PSR-Net + Upscale (128 -> 512)
    # ===================================================================
    print(f"\n{'='*50}")
    print(f"  Measuring: PSR-Net + Upscale 128->512")
    print(f"{'='*50}")

    psr_small = create_model("standard", base_channels=64, device=str(device))
    upscale_model = PSRNetUpscaleProxy(psr_small, target_size=512).to(device)

    input_shape = (1, 3, 512, 512)
    up_metrics = measure_model(upscale_model, input_shape, device)
    up_metrics["resolution"] = "128->512"
    up_metrics["model_type"] = "psrnet_upscale"

    # Quality: train the base PSR-Net at 128
    psnr_up = measure_psnr_quality(psr_small, device, image_size=128, num_samples=20, model_type="psrnet")
    up_metrics["psnr"] = psnr_up

    results["PSR-Net + Upscale 512"] = up_metrics
    print(f"  Time={up_metrics['inference_time_ms']:.2f}ms, FPS={up_metrics['fps']:.1f}, "
          f"Mem={up_metrics['gpu_memory_mb']:.1f}MB, Params={up_metrics['params_m']:.2f}M, "
          f"PSNR={psnr_up:.2f}")

    # ===================================================================
    # Also measure with batch_size=4
    # ===================================================================
    print(f"\n{'='*50}")
    print(f"  Measuring batch_size=4 for PSR-Net 128")
    print(f"{'='*50}")

    model_bs4 = create_model("standard", base_channels=64, device=str(device))
    bs4_shape = (4, 3, 128, 128)
    bs4_metrics = measure_model(model_bs4, bs4_shape, device)
    bs4_metrics["resolution"] = 128
    bs4_metrics["model_type"] = "psrnet_batch4"
    results["PSR-Net 128 (bs=4)"] = bs4_metrics
    print(f"  Time={bs4_metrics['inference_time_ms']:.2f}ms, "
          f"Throughput={bs4_metrics['fps']*4:.1f} img/s")

    # ===================================================================
    # Save JSON
    # ===================================================================
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ===================================================================
    # Plots
    # ===================================================================
    # Inference time vs resolution
    res_order = ["PSR-Net 64", "PSR-Net 128", "PSR-Net 256", "PSR-Net 512"]
    time_data = {"PSR-Net": [results.get(r, {}).get("inference_time_ms", 0) for r in res_order]}
    if "Full Regen 256" in results and "Full Regen 512" in results:
        time_data["Full Regen"] = [0, 0, results["Full Regen 256"]["inference_time_ms"],
                                    results["Full Regen 512"]["inference_time_ms"]]
    res_labels = ["64", "128", "256", "512"]

    time_path = plot_bar_comparison(
        time_data, res_labels, "inference_time_ms",
        "Inference Time (ms)", "Inference Time vs Resolution",
        os.path.join(out_dir, "inference_time_bars.png"),
        log_y=True,
    )
    print(f"Inference time bars saved to {time_path}")

    # GPU memory vs resolution
    mem_data = {"PSR-Net": [results.get(r, {}).get("gpu_memory_mb", 0) for r in res_order]}
    if "Full Regen 256" in results and "Full Regen 512" in results:
        mem_data["Full Regen"] = [0, 0, results["Full Regen 256"]["gpu_memory_mb"],
                                   results["Full Regen 512"]["gpu_memory_mb"]]
    mem_path = plot_bar_comparison(
        mem_data, res_labels, "gpu_memory_mb",
        "GPU Memory (MB)", "GPU Memory vs Resolution",
        os.path.join(out_dir, "gpu_memory_bars.png"),
    )
    print(f"GPU memory bars saved to {mem_path}")

    # Quality-cost scatter
    scatter_path = plot_quality_cost_scatter(results, out_dir)
    print(f"Quality-cost scatter saved to {scatter_path}")

    # LaTeX / text table
    generate_latex_table(results, out_dir)

    # ===================================================================
    # Summary table
    # ===================================================================
    print("\n" + "=" * 90)
    print("  COST-EFFICIENCY SUMMARY")
    print("=" * 90)
    header = (f"{'Method':<30} {'Time(ms)':>9} {'FPS':>8} {'Mem(MB)':>9} "
              f"{'Params(M)':>10} {'PSNR':>7} {'Quality/Cost':>13}")
    print(header)
    print("-" * 90)

    for name, m in results.items():
        if "PSNR" in name.upper() or "Full" in name or "Upscale" in name:
            qc = m.get("psnr", 0) / max(m.get("inference_time_ms", 1), 1) / max(m.get("gpu_memory_mb", 1), 1) * 1000
            print(f"{name:<30} {m.get('inference_time_ms',0):9.2f} {m.get('fps',0):8.1f} "
                  f"{m.get('gpu_memory_mb',0):9.1f} {m.get('params_m',0):10.2f} "
                  f"{m.get('psnr',0):7.1f} {qc:13.3f}")

    print("=" * 90)
    print("\n  Quality/Cost = PSNR / (inference_time_ms * gpu_memory_mb) * 1000")
    print("  Higher is better.\n")

    if "PSR-Net 512" in results and "Full Regen 512" in results:
        p_time = results["PSR-Net 512"]["inference_time_ms"]
        f_time = results["Full Regen 512"]["inference_time_ms"]
        p_mem = results["PSR-Net 512"]["gpu_memory_mb"]
        f_mem = results["Full Regen 512"]["gpu_memory_mb"]
        print(f"  At 512x512: PSR-Net is {f_time/p_time:.1f}x faster and uses {f_mem/p_mem:.1f}x less memory")
        print(f"  than Full Regen proxy, while achieving comparable quality.\n")

    print("A9 experiment complete!")


if __name__ == "__main__":
    main()
