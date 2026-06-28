"""
A5 Patch: Generate missing results.json + charts from existing checkpoint
A5 补丁：从已有 checkpoint 生成缺失的 results.json + 图表
Usage: python A5/patch_from_checkpoint.py
用法: python A5/patch_from_checkpoint.py
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_factory import create_model
from common.data_utils import generate_synthetic_pair, SyntheticDataset
from common.evaluation import evaluate_all, compute_psnr, compute_ssim, compute_iou, compute_mask_contrast_ratio
from common.visualization import plot_pareto_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# ── 1. Attempt to restore from incremental results.json ──
# ── 1. 尝试从 incremental results.json 恢复 ──
results_path = os.path.join(output_dir, "results.json")
all_results = []
if os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results")

# ── 2. Find all checkpoint directories and evaluate ──
# ── 2. 找到所有 checkpoint 目录并评估 ──
ckpt_base = output_dir
lambda_dirs = []
for name in os.listdir(ckpt_base):
    full = os.path.join(ckpt_base, name)
    if os.path.isdir(full) and name.startswith("checkpoints"):
        final_path = os.path.join(full, "final_model.pt")
        if os.path.exists(final_path):
            # Extract lambda value
            lam = 0.1  # default
            if "lambda_" in name:
                try:
                    lam = float(name.split("lambda_")[-1])
                except:
                    pass
            lambda_dirs.append((lam, full, final_path))

# Also check old path
old_final = os.path.join(ckpt_base, "checkpoints", "final_model.pt")
if os.path.exists(old_final) and not any(d[2] == old_final for d in lambda_dirs):
    lambda_dirs.append((0.1, os.path.join(ckpt_base, "checkpoints"), old_final))

# Collect existing lambda values from incremental saves
# 收集增量保存中已有的 lambda 值
existing_lambdas = {r["lambda_s"] for r in all_results}

print(f"Found {len(lambda_dirs)} checkpoint dirs")

# Test set
np.random.seed(999)
test_samples = 50
image_size = 64
test_dirty, test_clean, test_masks = [], [], []
for i in range(test_samples):
    d, c, m = generate_synthetic_pair(image_size, num_defects=3, defect_size=8, seed=999+i)
    test_dirty.append(torch.from_numpy(d.transpose(2,0,1)))
    test_clean.append(torch.from_numpy(c.transpose(2,0,1)))
    test_masks.append(torch.from_numpy(m.transpose(2,0,1)))

for lam, ckpt_dir, final_path in lambda_dirs:
    if lam in existing_lambdas:
        print(f"  λ_s={lam:.4f}: already in results.json, skip")
        continue
    
    print(f"  λ_s={lam:.4f}: evaluating from {final_path}...")
    try:
        model = create_model("standard", base_channels=64, device=device)
        state = torch.load(final_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        
        refined_list, mask_list = [], []
        with torch.no_grad():
            for dirty, clean, gt_m in zip(test_dirty, test_clean, test_masks):
                dirty_b = dirty.unsqueeze(0).to(device)
                refined, residual, mask = model.refine(dirty_b)
                refined_list.append(refined.cpu().squeeze(0))
                mask_list.append(mask.cpu().squeeze(0))
        
        metrics = evaluate_all(
            refined_list, test_clean, test_dirty, mask_list, test_masks,
            metrics=["psnr","ssim","l1","iou","mask_contrast","mask_mean","l1_improvement"]
        )
        metrics["lambda_s"] = lam
        all_results.append(metrics)
        print(f"    PSNR={metrics.get('psnr_mean',0):.2f}, IoU={metrics.get('iou_mean',0):.4f}")
        
        del model, refined_list, mask_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"    ❌ Failed: {e}")

# ── 3. Save results.json ──
# ── 3. 保存 results.json ──
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved {len(all_results)} results to {results_path}")

# ── 4. Generate plots ──
# ── 4. 生成图表 ──
if len(all_results) >= 1:
    lambdas = [r["lambda_s"] for r in all_results]
    psnrs = [r.get("psnr_mean", 0) for r in all_results]
    contrasts = [r.get("mask_contrast_ratio_mean", 0) for r in all_results]
    ssims = [r.get("ssim_mean", 0) for r in all_results]
    ious = [r.get("iou_mean", 0) for r in all_results]
    mask_means = [r.get("mask_mean_mean", 0) for r in all_results]
    
    # Pareto curve (only useful with ≥3 points, but generate anyway)
    if len(lambdas) >= 2:
        plot_pareto_curve(lambdas, psnrs, contrasts, 
                         os.path.join(output_dir, "pareto_curve.png"))
    
    # Lambda metrics grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, vals, ylabel, color in [
        (axes[0,0], psnrs, "PSNR (dB)", "#2E75B6"),
        (axes[0,1], ssims, "SSIM", "#C0504D"),
        (axes[1,0], ious, "IoU", "#4CAF50"),
        (axes[1,1], mask_means, "Mask Mean", "#FF9800"),
    ]:
        if len(lambdas) > 1:
            ax.plot(lambdas, vals, 'o-', color=color, linewidth=2, markersize=8)
        else:
            ax.bar([0], vals, color=color)
            ax.set_xticks([0])
            ax.set_xticklabels([f"λ={lambdas[0]}"])
        ax.set_xlabel("λ_s")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f"λ_s Sweep Analysis ({len(all_results)} point(s))", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"{'λ_s':8s} {'PSNR':8s} {'SSIM':8s} {'IoU':8s} {'Contrast':12s} {'MaskMean':10s}")
    print("-"*70)
    for r in sorted(all_results, key=lambda x: x["lambda_s"]):
        print(f"{r['lambda_s']:<8.4f} {r.get('psnr_mean',0):<8.2f} {r.get('ssim_mean',0):<8.4f} "
              f"{r.get('iou_mean',0):<8.4f} {r.get('mask_contrast_ratio_mean',0):<12.0f} "
              f"{r.get('mask_mean_mean',0):<10.6f}")

print("\n✅ A5 patch complete")
print("⚠️  Full sweep requires re-running: python A5/run.py --epochs 60")
