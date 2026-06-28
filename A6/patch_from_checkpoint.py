"""
A6 Patch: Generate missing results.json + charts from existing checkpoint
A6 补丁：从已有 checkpoint 生成缺失的 results.json + 图表
Usage: python A6/patch_from_checkpoint.py
用法: python A6/patch_from_checkpoint.py
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_factory import create_model
from common.data_utils import generate_synthetic_pair, SyntheticDataset, apply_gaussian_blur, apply_jpeg_artifact, apply_pixel_noise
from common.evaluation import evaluate_all, compute_psnr, compute_ssim, compute_iou, compute_mask_contrast_ratio, measure_inference_performance, format_results_table
from common.visualization import plot_method_comparison

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
image_size = 64

# ── 1. Load existing results ──
# ── 1. 加载已有结果 ──
results_path = os.path.join(output_dir, "results.json")
all_metrics = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        all_metrics = json.load(f)
    print(f"Loaded {len(all_metrics)} existing results")

# ── 2. Find checkpoint directories ──
# ── 2. 找到 checkpoint 目录 ──
ckpt_dirs = []
for name in os.listdir(output_dir):
    full = os.path.join(output_dir, name)
    if os.path.isdir(full) and name.startswith("checkpoints"):
        final_path = os.path.join(full, "final_model.pt")
        if os.path.exists(final_path):
            # Extract variant name
            variant = name.replace("checkpoints_", "") if name != "checkpoints" else "no_sparse"
            ckpt_dirs.append((variant, full, final_path))

if not ckpt_dirs:
    print("❌ No checkpoints found")
    sys.exit(1)

# ── 3. Test data ──
degradation_types = ["block", "blur", "jpeg", "noise"]
np.random.seed(9999)
test_data = {}
for dtype in degradation_types:
    test_data[dtype] = {"dirty":[], "clean":[], "mask":[]}
    for i in range(25):
        d, c, m = generate_synthetic_pair(image_size, num_defects=3, defect_size=8, seed=9999+i)
        if dtype == "blur":
            d, m = apply_gaussian_blur(c.copy())
        elif dtype == "jpeg":
            d, m = apply_jpeg_artifact(c.copy())
        elif dtype == "noise":
            d, m = apply_pixel_noise(c.copy())
        test_data[dtype]["dirty"].append(torch.from_numpy(d.transpose(2,0,1)).float())
        test_data[dtype]["clean"].append(torch.from_numpy(c.transpose(2,0,1)).float())
        test_data[dtype]["mask"].append(torch.from_numpy(m.transpose(2,0,1)).float())

# ── 4. Evaluate each variant ──
# ── 4. 评估每个 variant ──
for variant, ckpt_dir, final_path in ckpt_dirs:
    if variant in all_metrics:
        print(f"  {variant}: already in results.json, skip")
        continue
    
    print(f"\n  {variant}: evaluating from {final_path}")
    try:
        model = create_model("standard", base_channels=64, device=device)
        state = torch.load(final_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        
        variant_metrics = {}
        for dtype in degradation_types:
            td = test_data[dtype]
            refined_list, mask_list = [], []
            with torch.no_grad():
                for dirty, clean, gt_m in zip(td["dirty"], td["clean"], td["mask"]):
                    dirty_b = dirty.unsqueeze(0).to(device)
                    refined, residual, mask = model.refine(dirty_b)
                    refined_list.append(refined.cpu().squeeze(0))
                    mask_list.append(mask.cpu().squeeze(0))
            
            m = evaluate_all(refined_list, td["clean"], td["dirty"], mask_list, td["mask"],
                           metrics=["psnr","ssim","l1","lpips","iou","mask_contrast","mask_mean","l1_improvement"])
            variant_metrics[dtype] = m
        
        avg_metrics = {}
        for key in variant_metrics[degradation_types[0]].keys():
            vals = [variant_metrics[d][key] for d in degradation_types if key in variant_metrics[d]]
            if vals:
                avg_metrics[key] = float(np.mean(vals))
        
        dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
        perf = measure_inference_performance(model, dummy_input)
        avg_metrics.update(perf)
        avg_metrics["params"] = sum(p.numel() for p in model.parameters())
        
        all_metrics[variant] = {**avg_metrics, "per_type": variant_metrics}
        print(format_results_table(avg_metrics, f"{variant} - Average"))
        
        del model, refined_list, mask_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        import traceback; traceback.print_exc()

# ── 5. Save ──
# ── 5. 保存 ──
serializable = {}
for name, metrics in all_metrics.items():
    serializable[name] = {}
    for k, v in metrics.items():
        if k == "per_type":
            serializable[name][k] = {dt: {kk: float(vv) if isinstance(vv,(np.floating,np.integer)) else vv 
                                           for kk,vv in vm.items()} 
                                     for dt, vm in v.items()}
        else:
            serializable[name][k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
with open(results_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nSaved {len(all_metrics)} variants to {results_path}")

# ── 6. Generate plots ──
# ── 6. 生成图表 ──
if len(all_metrics) >= 1:
    # Per-type breakdown
    variants = list(all_metrics.keys())
    colors = ["#2E75B6", "#C0504D", "#4CAF50"][:len(variants)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_2d = ["psnr_mean", "ssim_mean", "iou_mean", "mask_contrast_ratio_mean", "mask_mean_mean", "l1_improvement_pct_mean"]
    titles = ["PSNR", "SSIM", "IoU", "Mask Contrast", "Mask Mean", "L1 Improvement %"]
    
    for ax, metric, title in zip(axes.flat, metrics_2d, titles):
        x = np.arange(len(degradation_types))
        width = 0.8 / max(len(variants), 1)
        for i, (variant, color) in enumerate(zip(variants, colors)):
            vals = [all_metrics[variant]["per_type"][d].get(metric, 0) for d in degradation_types]
            ax.bar(x + i*width, vals, width, label=variant, color=color, alpha=0.8)
        ax.set_xticks(x + width * (len(variants)-1) / 2)
        ax.set_xticklabels(degradation_types)
        ax.set_title(title); ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    
    plt.suptitle("PSR-Net Variants: Per-Degradation-Type Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_type_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    plot_method_comparison({k: v for k, v in all_metrics.items()},
                           metrics=["psnr_mean","ssim_mean","iou_mean","mask_contrast_ratio_mean"],
                           save_path=os.path.join(output_dir, "method_comparison.png"))
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Metric':25s} " + " ".join(f"{v:>15s}" for v in variants))
    print("-"*80)
    for km in ["psnr_mean","ssim_mean","iou_mean","mask_contrast_ratio_mean","l1_improvement_pct_mean"]:
        vals = [f"{all_metrics[v].get(km,0):.3f}" for v in variants]
        print(f"{km:25s} " + " ".join(f"{v:>15s}" for v in vals))

print("\n✅ A6 patch complete")
print("⚠️  Full 3-variant sweep requires re-running: python A6/run.py --epochs 80")
