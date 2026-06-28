"""
A4 Patch: Generate missing visualizations and JSON from existing checkpoint
A4 补丁：从已有 checkpoint 生成缺失的可视化和 JSON
Usage: python A4/patch_generate_plots.py [--use_sd] [--real_images_dir PATH]
用法: python A4/patch_generate_plots.py [--use_sd] [--real_images_dir PATH]
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model_factory import create_model, load_checkpoint
from common.data_utils import generate_synthetic_pair, generate_batch_synthetic, SDImageDataset
from common.evaluation import compute_psnr, compute_ssim, compute_iou, compute_mask_contrast_ratio, sanitize_metric_array

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="A4 Patch: Generate plots from checkpoint")
parser.add_argument("--use_sd", action="store_true",
                    help="Use SD v1.5 images from resources/sd_images/ cache (default: synthetic)")
parser.add_argument("--device", type=str, default=None,
                    help="Device override (default: auto-detect)")
args = parser.parse_args()

device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device_str)
print(f"Device: {device_str} | Data: {'SD images' if args.use_sd else 'synthetic (random noise)'}")

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

checkpoint_path = os.path.join(output_dir, "best_model.pt")
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    sys.exit(1)

print(f"Loading checkpoint: {checkpoint_path}")
model = create_model(model_type="standard", base_channels=64, input_channels=3, device=device_str)
model = load_checkpoint(model, checkpoint_path, device=device_str)
model.eval()
print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# Determine resolution from checkpoint file size (~70MB => likely 512)
print("Generating evaluation data...")
test_samples = 20
resolutions = [64, 128, 256, 512]
all_results = []

for res in resolutions:
    # Generate test data — use SD cache if requested
    if args.use_sd:
        ds = SDImageDataset(
            num_samples=min(test_samples, 10),
            size=res,
            seed=1042,  # Aligned with run.py test seed=seed+1000
                       # 与 run.py test seed=seed+1000 对齐
            device=device_str,
        )
        test_dirty, test_clean, test_masks = [], [], []
        for i in range(len(ds)):
            d, c, m = ds[i]
            test_dirty.append(d)
            test_clean.append(c)
            test_masks.append(m)
    else:
        test_dirty, test_clean, test_masks = [], [], []
        for i in range(min(test_samples, 10)):
            d, c, m = generate_synthetic_pair(res, num_defects=3,
                                               defect_size=max(4, res//16), seed=999 + i)
            test_dirty.append(torch.from_numpy(d.transpose(2, 0, 1)).float())
            test_clean.append(torch.from_numpy(c.transpose(2, 0, 1)).float())
            test_masks.append(torch.from_numpy(m.transpose(2, 0, 1)).float())

    psnrs, ssims, ious, contrasts, mask_means = [], [], [], [], []
    refined_samples, mask_samples, dirty_samples = [], [], []

    with torch.no_grad():
        for dirty, clean, gt_m in zip(test_dirty, test_clean, test_masks):
            dirty_b = dirty.unsqueeze(0).to(device)
            refined, _, mask = model.refine(dirty_b)

            psnrs.append(compute_psnr(refined, clean.unsqueeze(0).to(device)))
            ssims.append(compute_ssim(refined, clean.unsqueeze(0).to(device)))
            ious.append(compute_iou(mask.cpu(), gt_m.unsqueeze(0), threshold=None))
            contrasts.append(compute_mask_contrast_ratio(mask.cpu(), gt_m.unsqueeze(0)))
            mask_means.append(mask.mean().item())

            refined_samples.append(refined.cpu().squeeze(0))
            mask_samples.append(mask.cpu().squeeze(0))
            dirty_samples.append(dirty)

    results = {
        "resolution": res,
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "iou_mean": float(np.mean(ious)),
        "mask_contrast_mean": float(np.mean(sanitize_metric_array(contrasts))) if sanitize_metric_array(contrasts) else 0.0,
        "mask_mean": float(np.mean(mask_means)),
    }
    all_results.append(results)
    print(f"  {res}x{res}: PSNR={results['psnr_mean']:.1f}, SSIM={results['ssim_mean']:.4f}, IoU={results['iou_mean']:.4f}")

# Save results.json
results_path = os.path.join(output_dir, "results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {results_path}")

# ---- Resolution comparison plots ----
sizes = [r["resolution"] for r in all_results]
psnrs = [r["psnr_mean"] for r in all_results]
ssims = [r["ssim_mean"] for r in all_results]
ious = [r["iou_mean"] for r in all_results]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(range(len(sizes)), psnrs, color="#2E75B6", edgecolor="white")
axes[0].set_xticks(range(len(sizes)))
axes[0].set_xticklabels([f"{s}" for s in sizes])
axes[0].set_ylabel("PSNR (dB)")
axes[0].set_title("PSNR vs Resolution")
for i, v in enumerate(psnrs):
    axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

axes[1].bar(range(len(sizes)), ssims, color="#C0504D", edgecolor="white")
axes[1].set_xticks(range(len(sizes)))
axes[1].set_xticklabels([f"{s}" for s in sizes])
axes[1].set_ylabel("SSIM")
axes[1].set_title("SSIM vs Resolution")
for i, v in enumerate(ssims):
    axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

axes[2].bar(range(len(sizes)), ious, color="#4CAF50", edgecolor="white")
axes[2].set_xticks(range(len(sizes)))
axes[2].set_xticklabels([f"{s}" for s in sizes])
axes[2].set_ylabel("IoU")
axes[2].set_title("Mask IoU vs Resolution")
for i, v in enumerate(ious):
    axes[2].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

plt.suptitle("PSR-Net Resolution Scalability", fontsize=14)
plt.tight_layout()
path = os.path.join(output_dir, "resolution_comparison.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

# ---- Visual quality grid (4 rows x 4 cols) ----
n_res = min(len(resolutions), 4)
fig, axes = plt.subplots(n_res, 4, figsize=(16, n_res * 4))
if n_res == 1:
    axes = axes.reshape(1, -1)

col_labels = ["Dirty", "Refined", "GT", "Mask"]
for ax, label in zip(axes[0], col_labels):
    ax.set_title(label, fontsize=12)

for r, res in enumerate(resolutions[:n_res]):
    if args.use_sd:
        ds = SDImageDataset(num_samples=1, size=res, seed=42, device=device_str)
        dirty_t, clean_t, _ = ds[0]
        dirty_t = dirty_t.unsqueeze(0).float().to(device)
    else:
        d, c, m = generate_synthetic_pair(res, num_defects=3,
                                           defect_size=max(4, res//16), seed=42)
        dirty_t = torch.from_numpy(d.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        clean_t = c  # HWC numpy for display

    with torch.no_grad():
        refined, _, mask = model.refine(dirty_t)

    def to_np(t):
        img = t.squeeze(0).cpu().numpy()
        if img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)
        elif img.ndim == 2:
            pass
        return np.clip(img, 0, 1)

    axes[r, 0].imshow(to_np(dirty_t))
    axes[r, 0].set_ylabel(f"{res}x{res}", fontsize=10, rotation=0, labelpad=30)
    axes[r, 0].axis("off")
    axes[r, 1].imshow(to_np(refined))
    axes[r, 1].axis("off")

    if args.use_sd:
        # Clean 来自 SDImageDataset (CHW tensor)
        axes[r, 2].imshow(to_np(clean_t.unsqueeze(0)))
    else:
        axes[r, 2].imshow(np.clip(clean_t, 0, 1))  # clean is HWC numpy from generate_synthetic_pair
    axes[r, 2].axis("off")
    axes[r, 3].imshow(to_np(mask), cmap="hot")
    axes[r, 3].axis("off")

plt.suptitle(f"PSR-Net Multi-Resolution Repair Quality{' (SD images)' if args.use_sd else ''}", fontsize=14)
plt.tight_layout()
path = os.path.join(output_dir, "visual_quality_grid.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

print("\n✅ A4 patch complete!")
