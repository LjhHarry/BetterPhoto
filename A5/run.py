import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import ExperimentConfig
from common.model_factory import create_model
from common.data_utils import SyntheticDataset, generate_synthetic_pair
from common.evaluation import compute_psnr, compute_ssim, compute_iou, compute_mask_contrast_ratio, evaluate_all
from common.visualization import plot_pareto_curve, plot_results_grid
from common.training import TrainingEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lambda values to sweep
    lambda_values = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    # Shared test set
    np.random.seed(999)
    test_samples = 50
    test_dirty, test_clean, test_masks = [], [], []
    for i in range(test_samples):
        d, c, m = generate_synthetic_pair(args.image_size, num_defects=3, defect_size=8, seed=999+i)
        test_dirty.append(torch.from_numpy(d.transpose(2,0,1)))
        test_clean.append(torch.from_numpy(c.transpose(2,0,1)))
        test_masks.append(torch.from_numpy(m.transpose(2,0,1)))
    
    all_results = []
    
    for lam in lambda_values:
        print(f"\n{'='*50}\nTraining with lambda_s = {lam}\n{'='*50}")
        
        config = ExperimentConfig(
            name=f"lambda_{lam}", epochs=args.epochs,
            image_size=args.image_size, batch_size=args.batch_size,
            lambda_sparse=lam, warmup_epochs=args.epochs//2,
            train_samples=500, test_samples=test_samples,
        )
        
        # Create dataset
        train_dataset = SyntheticDataset(config.train_samples, config.image_size, seed=42)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Train
        model = create_model("standard", base_channels=64, device=device)
        engine = TrainingEngine(model, config, device)
        history = engine.train(train_loader, save_dir=os.path.join(output_dir, "checkpoints"))
        
        # Evaluate on test set
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

        # 检测 mask 坍缩：当 mask 均值极小时记录警告
        mask_mean_val = metrics.get("mask_mean_mean", 0)
        if mask_mean_val < 1e-10:
            metrics["mask_collapsed"] = True
            print(f"  ⚠️  WARNING: Mask collapsed at λ_s={lam} (mask_mean={mask_mean_val:.2e})"
                  f" — sparsity penalty too strong, model stopped producing masks")
        else:
            metrics["mask_collapsed"] = False

        all_results.append(metrics)
        print(f"  PSNR={metrics.get('psnr_mean',0):.2f}, SSIM={metrics.get('ssim_mean',0):.4f}, "
              f"IoU={metrics.get('iou_mean',0):.4f}, Contrast={metrics.get('mask_contrast_ratio_mean',0):.0f}")
    
    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot Pareto curve
    lambdas = [r["lambda_s"] for r in all_results]
    psnrs = [r.get("psnr_mean",0) for r in all_results]
    contrasts = [r.get("mask_contrast_ratio_mean",0) for r in all_results]
    plot_pareto_curve(lambdas, psnrs, contrasts, os.path.join(output_dir, "pareto_curve.png"))
    
    # Also plot SSIM and IoU vs lambda
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14,10))
    axes[0,0].plot(lambdas, psnrs, 'o-', color='#2E75B6', linewidth=2, markersize=8)
    axes[0,0].set_xlabel("λ_s"); axes[0,0].set_ylabel("PSNR (dB)"); axes[0,0].grid(alpha=0.3)
    ssims = [r.get("ssim_mean",0) for r in all_results]
    axes[0,1].plot(lambdas, ssims, 's-', color='#C0504D', linewidth=2, markersize=8)
    axes[0,1].set_xlabel("λ_s"); axes[0,1].set_ylabel("SSIM"); axes[0,1].grid(alpha=0.3)
    ious = [r.get("iou_mean",0) for r in all_results]
    axes[1,0].plot(lambdas, ious, '^-', color='#4CAF50', linewidth=2, markersize=8)
    axes[1,0].set_xlabel("λ_s"); axes[1,0].set_ylabel("IoU"); axes[1,0].grid(alpha=0.3)
    mask_means = [r.get("mask_mean_mean",0) for r in all_results]
    axes[1,1].plot(lambdas, mask_means, 'D-', color='#FF9800', linewidth=2, markersize=8)
    axes[1,1].set_xlabel("λ_s"); axes[1,1].set_ylabel("Mask Mean"); axes[1,1].grid(alpha=0.3)
    plt.suptitle("λ_s Sweep Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lambda_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'λ_s':8s} {'PSNR':8s} {'SSIM':8s} {'IoU':8s} {'Contrast':12s} {'MaskMean':10s}")
    print("-"*80)
    for r in all_results:
        print(f"{r['lambda_s']:<8.3f} {r.get('psnr_mean',0):<8.2f} {r.get('ssim_mean',0):<8.4f} "
              f"{r.get('iou_mean',0):<8.4f} {r.get('mask_contrast_ratio_mean',0):<12.0f} "
              f"{r.get('mask_mean_mean',0):<10.6f}")
    
    # Find optimal lambda
    best_idx = np.argmax(np.array(psnrs) * np.log1p(np.array(contrasts)))
    print(f"\nOptimal λ_s = {lambdas[best_idx]} (PSNR={psnrs[best_idx]:.2f}, Contrast={contrasts[best_idx]:.0f})")

if __name__ == "__main__":
    main()
