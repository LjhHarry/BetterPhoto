import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import ExperimentConfig
from common.model_factory import create_model
from common.data_utils import SyntheticDataset, generate_synthetic_pair, apply_gaussian_blur, apply_jpeg_artifact, apply_pixel_noise
from common.evaluation import (compute_psnr, compute_ssim, compute_iou, compute_mask_contrast_ratio, 
                                compute_l1_improvement, compute_lpips_approx, compute_fid, compute_activation_stats,
                                evaluate_all, format_results_table, measure_inference_performance)
from common.visualization import plot_results_grid, plot_ablation_bars, plot_method_comparison
from common.training import TrainingEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3 variants
    configs = {
        "no_sparse":    ExperimentConfig(name="no_sparse", lambda_sparse=0.0, epochs=args.epochs, 
                                          image_size=args.image_size, batch_size=args.batch_size,
                                          warmup_epochs=0, train_samples=1000, test_samples=100),
        "mild_sparse":  ExperimentConfig(name="mild_sparse", lambda_sparse=0.03, epochs=args.epochs,
                                          image_size=args.image_size, batch_size=args.batch_size,
                                          warmup_epochs=args.epochs//2, train_samples=1000, test_samples=100),
        "strong_sparse":ExperimentConfig(name="strong_sparse", lambda_sparse=0.1, epochs=args.epochs,
                                          image_size=args.image_size, batch_size=args.batch_size,
                                          warmup_epochs=args.epochs//2, train_samples=1000, test_samples=100),
    }
    
    # 4 degradation types for testing
    degradation_types = ["block", "blur", "jpeg", "noise"]
    
    # Generate test data per degradation type
    np.random.seed(9999)
    test_data = {}
    for dtype in degradation_types:
        test_data[dtype] = {"dirty":[], "clean":[], "mask":[]}
        for i in range(25):
            # Generate a base clean image
            d, c, m = generate_synthetic_pair(args.image_size, num_defects=3, defect_size=8, seed=9999+i)
            if dtype == "block":
                pass  # already block occlusion
            elif dtype == "blur":
                d, m = apply_gaussian_blur(c.copy())
            elif dtype == "jpeg":
                d, m = apply_jpeg_artifact(c.copy())
            elif dtype == "noise":
                d, m = apply_pixel_noise(c.copy())
            test_data[dtype]["dirty"].append(torch.from_numpy(d.transpose(2,0,1)).float())
            test_data[dtype]["clean"].append(torch.from_numpy(c.transpose(2,0,1)).float())
            test_data[dtype]["mask"].append(torch.from_numpy(m.transpose(2,0,1)).float())
    
    # Train and evaluate each variant
    all_metrics = {}
    results_path = os.path.join(output_dir, "results.json")
    
    for name, config in configs.items():
        print(f"\n{'='*50}\nTraining {name} (λ_s={config.lambda_sparse})\n{'='*50}")
        
        train_dataset = SyntheticDataset(config.train_samples, config.image_size, seed=42)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        try:
            model = create_model("standard", base_channels=64, device=device)
            engine = TrainingEngine(model, config, device)
            history = engine.train(train_loader, 
                                   save_dir=os.path.join(output_dir, f"checkpoints_{name}"))
            
            # Evaluate per degradation type
            variant_metrics = {}
            for dtype in degradation_types:
                td = test_data[dtype]
                model.eval()
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
            
            # Average across degradation types
            avg_metrics = {}
            for key in variant_metrics[degradation_types[0]].keys():
                vals = [variant_metrics[d][key] for d in degradation_types if key in variant_metrics[d]]
                if vals:
                    avg_metrics[key] = float(np.mean(vals))
            
            # Inference performance
            dummy_input = torch.randn(1, 3, args.image_size, args.image_size).to(device)
            perf = measure_inference_performance(model, dummy_input)
            avg_metrics.update(perf)
            avg_metrics["params"] = sum(p.numel() for p in model.parameters())
            
            all_metrics[name] = {**avg_metrics, "per_type": variant_metrics}
            print(f"  ✅ {name} done")
            print(format_results_table(avg_metrics, f"{name} - Average Metrics"))
            
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            import traceback; traceback.print_exc()
        
        # 增量保存
        serializable = {}
        for nm, metrics in all_metrics.items():
            serializable[nm] = {}
            for k, v in metrics.items():
                if k == "per_type":
                    serializable[nm][k] = {dt: {kk: float(vv) if isinstance(vv,(np.floating,np.integer)) else vv 
                                                 for kk,vv in vm.items()} 
                                           for dt, vm in v.items()}
                else:
                    serializable[nm][k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        
        # 释放 GPU
        del model, engine, train_loader, train_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nCompleted {len(all_metrics)}/3 variants")
    
    # --- Visualizations (robust to partial results) ---
    if len(all_metrics) == 0:
        print("⚠️ No variants completed, skipping visualizations")
        return
    
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Per-type breakdown
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_2d = ["psnr_mean", "ssim_mean", "iou_mean", "mask_contrast_ratio_mean", "mask_mean_mean", "l1_improvement_pct_mean"]
    titles = ["PSNR", "SSIM", "IoU", "Mask Contrast", "Mask Mean", "L1 Improvement %"]
    
    variants = list(all_metrics.keys())
    colors = ["#2E75B6", "#C0504D", "#4CAF50"][:len(variants)]
    
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
    
    # Method comparison chart
    plot_method_comparison({k: v for k, v in all_metrics.items()}, 
                           metrics=["psnr_mean","ssim_mean","iou_mean","mask_contrast_ratio_mean"],
                           save_path=os.path.join(output_dir, "method_comparison.png"))
    
    # LaTeX table
    latex_table = generate_latex_table(all_metrics)
    with open(os.path.join(output_dir, "metrics_table.tex"), "w") as f:
        f.write(latex_table)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE METRICS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Metric':25s} {'No Sparse':15s} {'Mild (0.03)':15s} {'Strong (0.1)':15s}")
    print("-"*80)
    key_metrics = ["psnr_mean","ssim_mean","iou_mean","mask_contrast_ratio_mean","l1_improvement_pct_mean","inference_time_ms"]
    for km in key_metrics:
        vals = [str(round(all_metrics[v].get(km, 0), 3)) for v in variants]
        print(f"{km:25s} {vals[0]:15s} {vals[1]:15s} {vals[2]:15s}")

def generate_latex_table(all_metrics):
    """Generate LaTeX table for paper"""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comprehensive evaluation of PSR-Net variants across multiple metrics}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Metric} & \textbf{No Sparse} & \textbf{Mild (0.03)} & \textbf{Strong (0.1)} \\")
    lines.append(r"\hline")
    
    metrics_map = [
        ("PSNR (dB)", "psnr_mean", ".1f"),
        ("SSIM", "ssim_mean", ".4f"),
        ("IoU", "iou_mean", ".4f"),
        ("Mask Contrast Ratio", "mask_contrast_ratio_mean", ".0f"),
        ("Mask Mean", "mask_mean_mean", ".6f"),
        ("L1 Improvement (\\%)", "l1_improvement_pct_mean", ".1f"),
        ("Inference Time (ms)", "inference_time_ms", ".1f"),
    ]
    
    variants = list(all_metrics.keys())
    for label, key, fmt in metrics_map:
        vals = [all_metrics[v].get(key, 0) for v in variants]
        formatted = " & ".join([f"{v:{fmt}}" for v in vals])
        lines.append(f"{label} & {formatted} \\\\")
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

if __name__ == "__main__":
    main()
