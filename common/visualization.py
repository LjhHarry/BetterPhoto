"""
Visualization utilities -- shared plotting functions for all experiments
可视化工具 —— 所有实验共用的绘图函数
Supports: result comparison plots, training curves, Pareto frontier, ablation bars, LaTeX tables
支持：结果对比图、训练曲线、Pareto图、消融柱状图、LaTeX表格
"""
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

# Chinese font configuration
# 中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Tensor to HWC numpy array
    Tensor 转 HWC numpy"""
    if t.dim() == 4:
        t = t[0]
    img = t.detach().cpu().numpy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    elif img.shape[0] == 1:
        img = img.squeeze(0)
    return np.clip(img, 0, 1)


def plot_results_grid(dirty_list: List[torch.Tensor], 
                      refined_list: List[torch.Tensor],
                      gt_list: List[torch.Tensor],
                      mask_list: Optional[List[torch.Tensor]] = None,
                      gt_mask_list: Optional[List[torch.Tensor]] = None,
                      titles: List[str] = None,
                      save_path: str = "results_grid.png",
                      figsize: Tuple[int, int] = None,
                      max_samples: int = 6):
    """
    Plot a grid of result comparison images
    绘制结果网格对比图
    Columns: Dirty | Refined | GT | Mask | GT Mask
    """
    n = min(len(dirty_list), max_samples)
    n_cols = 3 + (1 if mask_list else 0) + (1 if gt_mask_list else 0)
    n_rows = n
    
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    col_labels = ["Dirty", "Refined", "GT"]
    if mask_list:
        col_labels.append("Mask")
    if gt_mask_list:
        col_labels.append("GT Mask")
    
    for ax, label in zip(axes[0], col_labels):
        ax.set_title(label, fontsize=12)
    
    for i in range(n):
        col = 0
        axes[i, col].imshow(tensor_to_numpy(dirty_list[i]))
        axes[i, col].axis("off")
        col += 1
        
        axes[i, col].imshow(tensor_to_numpy(refined_list[i]))
        axes[i, col].axis("off")
        col += 1
        
        axes[i, col].imshow(tensor_to_numpy(gt_list[i]))
        axes[i, col].axis("off")
        col += 1
        
        if mask_list and i < len(mask_list):
            axes[i, col].imshow(tensor_to_numpy(mask_list[i]), cmap="hot")
            axes[i, col].axis("off")
            col += 1
        
        if gt_mask_list and i < len(gt_mask_list):
            axes[i, col].imshow(tensor_to_numpy(gt_mask_list[i]), cmap="hot")
            axes[i, col].axis("off")
    
    if titles:
        for i, title in enumerate(titles[:n]):
            axes[i, 0].set_ylabel(title, fontsize=10, rotation=0, 
                                   labelpad=60, va="center")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(histories: Dict[str, Dict], 
                          save_path: str = "training_curves.png",
                          figsize: Tuple[int, int] = (12, 10)):
    """
    Plot training curves
    绘制训练曲线
    histories: {"config_name": {"epoch": [...], "loss": [...], "mask_mean": [...], ...}}
    """
    n_metrics = 3  # loss, mask_mean, psnr
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for (name, hist), color in zip(histories.items(), colors):
        epochs = hist.get("epoch", range(len(hist.get("loss", []))))
        
        # L1 Loss
        if "train_loss" in hist:
            axes[0].plot(epochs, hist["train_loss"], color=color, label=name, alpha=0.7)
        axes[0].set_ylabel("L1 Reconstruction Loss")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Mask Mean
        if "mask_mean" in hist:
            axes[1].plot(epochs, hist["mask_mean"], color=color, label=name, alpha=0.7)
        axes[1].set_ylabel("Mask Mean (Sparsity)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        # PSNR
        if "psnr" in hist:
            axes[2].plot(epochs, hist["psnr"], color=color, label=name, alpha=0.7)
        axes[2].set_ylabel("PSNR (dB)")
        axes[2].set_xlabel("Epoch")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_pareto_curve(lambda_values: List[float], 
                       psnr_values: List[float],
                       mask_contrast_values: List[float],
                       save_path: str = "pareto_curve.png",
                       figsize: Tuple[int, int] = (10, 6)):
    """
    Plot Pareto frontier: PSNR vs mask contrast ratio (λ_s sweep)
    绘制 Pareto 曲线: PSNR vs 掩膜对比度 (λ_s 扫描)
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    color1 = "#2E75B6"
    color2 = "#C0504D"
    
    ax1.set_xlabel("λ_s (Sparsity Regularization Coefficient)")
    ax1.set_ylabel("PSNR (dB)", color=color1)
    ax1.plot(lambda_values, psnr_values, "o-", color=color1, linewidth=2, 
             markersize=8, label="PSNR")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mask Contrast Ratio", color=color2)
    ax2.plot(lambda_values, mask_contrast_values, "s--", color=color2, linewidth=2,
             markersize=8, label="Mask Contrast")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yscale("log")
    
    # Annotate the optimal (best) value
    # 标注最优值
    best_idx = np.argmax(np.array(psnr_values) * np.array(mask_contrast_values) ** 0.1)
    ax1.annotate(f"λ={lambda_values[best_idx]}\nPSNR={psnr_values[best_idx]:.1f}",
                 xy=(lambda_values[best_idx], psnr_values[best_idx]),
                 xytext=(10, -20), textcoords="offset points",
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9, color=color1)
    
    fig.suptitle("PSNR vs Mask Contrast Pareto Frontier (λ_s Sweep)", fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_ablation_bars(results: Dict[str, Dict], 
                        save_path: str = "ablation_bars.png",
                        figsize: Tuple[int, int] = (14, 5)):
    """
    Plot ablation study bar chart
    绘制消融实验柱状图
    results: {"method_name": {"psnr": ..., "ssim": ..., "mask_contrast": ...}}
    """
    methods = list(results.keys())
    metrics = ["psnr", "ssim", "mask_contrast_ratio"]
    metric_labels = ["PSNR (dB)", "SSIM", "Mask Contrast Ratio"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [results[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values, color=colors, edgecolor="white")
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        
        # Annotate value labels on bars
        # 数值标注
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                        f"{val:.2f}" if val < 100 else f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_iterative_refinement(dirty: torch.Tensor,
                               refined_steps: List[torch.Tensor],
                               masks: List[torch.Tensor],
                               gt: torch.Tensor,
                               save_path: str = "iterative_refinement.png",
                               figsize: Tuple[int, int] = None):
    """
    Plot iterative refinement process
    绘制迭代精修过程
    """
    n_steps = len(refined_steps)
    n_cols = n_steps + 2  # dirty, step1, step2, ..., gt
    n_rows = 2  # images + masks
    
    if figsize is None:
        figsize = (n_cols * 3, 6)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # First row: images
    # 第一行：图像
    axes[0, 0].imshow(tensor_to_numpy(dirty))
    axes[0, 0].set_title("Dirty", fontsize=10)
    axes[0, 0].axis("off")
    
    for i, (ref, mask) in enumerate(zip(refined_steps, masks)):
        axes[0, i+1].imshow(tensor_to_numpy(ref))
        axes[0, i+1].set_title(f"Refined Step {i+1}", fontsize=10)
        axes[0, i+1].axis("off")
        
        axes[1, i+1].imshow(tensor_to_numpy(mask), cmap="hot")
        axes[1, i+1].set_title(f"Mask S{i+1} (mean={mask.mean().item():.4f})", fontsize=9)
        axes[1, i+1].axis("off")
    
    axes[0, -1].imshow(tensor_to_numpy(gt))
    axes[0, -1].set_title("GT", fontsize=10)
    axes[0, -1].axis("off")
    
    axes[1, 0].axis("off")
    axes[1, -1].axis("off")
    
    plt.suptitle("Iterative Selective Redrawing Process", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_cost_efficiency(results: Dict[str, Dict],
                          save_path: str = "cost_efficiency.png",
                          figsize: Tuple[int, int] = (12, 5)):
    """
    Plot cost-efficiency analysis chart
    绘制成本效益分析图
    results: {"method": {"psnr": ..., "inference_time_ms": ..., "gpu_memory_mb": ...}}
    """
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # PSNR vs Inference Time
    times = [results[m].get("inference_time_ms", 0) for m in methods]
    psnrs = [results[m].get("psnr", 0) for m in methods]
    sizes = [results[m].get("params_m", 0) * 50 for m in methods]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    texts_left, texts_right = [], []
    for m, t, p, s, c in zip(methods, times, psnrs, sizes, colors):
        axes[0].scatter(t, p, s=s, color=c, alpha=0.8, edgecolors="black")
        texts_left.append(axes[0].text(t, p, m, fontsize=8,
                                       ha="center", va="center",
                                       bbox=dict(boxstyle="round,pad=0.2",
                                                 fc="white", alpha=0.8,
                                                 ec="gray", lw=0.3)))
    axes[0].set_xlabel("Inference Time (ms)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Quality vs Speed")
    axes[0].grid(True, alpha=0.3)
    
    # PSNR vs GPU Memory
    mems = [results[m].get("gpu_memory_mb", 0) for m in methods]
    for m, mem, p, s, c in zip(methods, mems, psnrs, sizes, colors):
        axes[1].scatter(mem, p, s=s, color=c, alpha=0.8, edgecolors="black")
        texts_right.append(axes[1].text(mem, p, m, fontsize=8,
                                        ha="center", va="center",
                                        bbox=dict(boxstyle="round,pad=0.2",
                                                  fc="white", alpha=0.8,
                                                  ec="gray", lw=0.3)))
    axes[1].set_xlabel("GPU Memory (MB)")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Quality vs Memory")
    axes[1].grid(True, alpha=0.3)

    axes[0].margins(x=0.15, y=0.15)
    axes[1].margins(x=0.15, y=0.15)
    adjust_text(texts_left, ax=axes[0],
                expand_points=(1.5, 1.2), expand_text=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                force_text=(0.5, 0.5), force_points=(0.2, 0.2),
                lim=200)
    adjust_text(texts_right, ax=axes[1],
                expand_points=(1.5, 1.2), expand_text=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                force_text=(0.5, 0.5), force_points=(0.2, 0.2),
                lim=200)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_method_comparison(all_results: Dict[str, Dict],
                            metrics: List[str] = None,
                            save_path: str = "method_comparison.png",
                            figsize: Tuple[int, int] = None):
    """
    Plot method comparison chart
    绘制方法对比图
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "mask_contrast_ratio", "iou", "l1_improvement_pct"]
    
    n_metrics = len(metrics)
    if figsize is None:
        figsize = (n_metrics * 3.5, 5)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    methods = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for ax, metric in zip(axes, metrics):
        values = []
        for m in methods:
            val = all_results[m].get(metric, all_results[m].get(f"{metric}_mean", 0))
            values.append(val)
        
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor="white")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        
        # Annotate value labels on bars
        # 标注数值
        for bar, val in zip(bars, values):
            if abs(val) < 100:
                label = f"{val:.2f}"
            else:
                label = f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.03,
                    label, ha="center", va="bottom", fontsize=6)
    
    plt.suptitle("Method Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_pixel_fidelity_map(original: torch.Tensor, refined: torch.Tensor,
                             mask: torch.Tensor, save_path: str = "pixel_fidelity.png",
                             figsize: Tuple[int, int] = (15, 5)):
    """
    Plot pixel fidelity visualization
    绘制像素保真率可视化
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    axes[0].imshow(tensor_to_numpy(original))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")
    
    axes[1].imshow(tensor_to_numpy(refined))
    axes[1].set_title("Refined", fontsize=10)
    axes[1].axis("off")
    
    axes[2].imshow(tensor_to_numpy(mask), cmap="hot")
    axes[2].set_title("Mask (Modified Regions)", fontsize=10)
    axes[2].axis("off")
    
    # Difference map
    # 差异图
    diff = (torch.abs(refined - original) * 10).clamp(0, 1)
    axes[3].imshow(tensor_to_numpy(diff), cmap="hot")
    axes[3].set_title("Difference (x10)", fontsize=10)
    axes[3].axis("off")
    
    # Compute fidelity rate
    # 计算保真率
    unchanged = 1.0 - mask
    diff_map = torch.abs(refined - original)
    fidelity = ((diff_map < 1e-6).float() * unchanged).sum() / (unchanged.sum() + 1e-8)
    
    plt.suptitle(f"Pixel Fidelity Analysis (M=0 region preserved: {fidelity.item()*100:.2f}%)", 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
