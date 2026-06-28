"""
Experiment A1: Adversarial Degradation Simulation & Historical Checkpoint Degradation

Verifies PSR-Net generalizes beyond random block occlusion by testing four degradation types:
  a) Basic block occlusion (baseline)
  b) Gaussian blur degradation
  c) JPEG artifact degradation
  d) Learned adversarial degradation (from a trained DegradationNet)

Also simulates "historical checkpoint" degradation: train a generator for 5 epochs,
use epoch 1/3 outputs as degraded, epoch 5 as GT.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# ---- path setup ----
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import get_config
from common.model_factory import create_model
from common.evaluation import (compute_psnr, compute_ssim, compute_mask_contrast_ratio,
                                evaluate_all, format_results_table)
from common.data_utils import (generate_synthetic_pair, generate_batch_synthetic,
                                apply_gaussian_blur, apply_jpeg_artifact,
                                _apply_block_occlusion,
                                SyntheticDataset, create_dataloaders)
from common.visualization import tensor_to_numpy
from common.training import TrainingEngine, CheckpointManager, _format_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Degradation Network
# =============================================================================

class DegradationNet(nn.Module):
    """3-layer CNN that learns to add artifacts to clean images."""

    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_degradation_net(device, image_size=64, epochs=30, batch_size=16, 
                          verbose=True, save_dir=None):
    """
    Train DegradationNet to learn adversarial degradation.
    Strategy: given clean images, generate degraded versions via random block occlusion,
    then train DegradationNet to predict pixel-wise modification that converts clean → degraded.
    """
    model = DegradationNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ckpt_mgr = CheckpointManager(save_dir or ".", keep_last=3) if save_dir else None
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 200 // batch_size

        for _ in range(max(1, n_batches)):
            dirty_batch, clean_batch, _ = generate_batch_synthetic(
                batch_size, size=image_size, num_defects=3, defect_size=8)
            dirty_batch = dirty_batch.to(device)
            clean_batch = clean_batch.to(device)

            target_artifact = dirty_batch - clean_batch
            pred_artifact = model(clean_batch)

            loss = F.mse_loss(pred_artifact, target_artifact)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, n_batches)
        
        # 进度 + ETA
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            elapsed = time.time() - train_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1) if epoch > 0 else 0
            print(f"  [DegNet] Epoch {epoch:3d}/{epochs} [{100*(epoch+1)//epochs}%] "
                  f"| MSE={avg_loss:.6f} | ETA={_format_time(eta)}")
        
        # 定期保存
        if ckpt_mgr and (epoch % 10 == 0 or epoch == epochs - 1):
            ckpt_mgr.save(model, optimizer, epoch, {"loss": [avg_loss]},
                         {"mse": -avg_loss})  # negative because we want to minimize

    return model


@torch.no_grad()
def apply_adversarial_degradation(dirty_img: np.ndarray, clean_img: np.ndarray,
                                   deg_net: nn.Module, device) -> np.ndarray:
    """
    Apply learned adversarial degradation on a dirty image.
    Adds DegradationNet's predicted noise to the image.
    Returns (degraded_img, mask).
    """
    h, w = dirty_img.shape[:2]
    clean_t = torch.from_numpy(clean_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    pred_artifact = deg_net(clean_t).squeeze(0)

    # Mix: add adversarial artifact to original dirty
    adv_noise = pred_artifact.cpu().numpy().transpose(1, 2, 0)
    degraded = np.clip(dirty_img + 0.3 * adv_noise, 0.0, 1.0)

    # Mask: regions with significant change
    diff = np.abs(degraded - dirty_img).mean(axis=2, keepdims=True)
    mask = (diff > 0.05).astype(np.float32)
    if mask.sum() == 0:
        mask[0, 0, 0] = 1.0

    return degraded.astype(np.float32), mask.astype(np.float32)


# =============================================================================
# Historical Checkpoint Simulation
# =============================================================================

def simulate_historical_checkpoints(device, image_size=64, epochs=5, batch_size=16, verbose=True):
    """
    Train a simple generator for N epochs, saving intermediate checkpoints.
    Epoch 1/3 outputs = degraded, epoch 5 = GT.
    Returns (checkpoint_outputs, clean_reference_images).
    """
    # Create a small "generator" (just a simple CNN for demonstration)
    class SimpleGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 3, padding=1),
            )

        def forward(self, x):
            return torch.tanh(self.net(x))  # output in [-1, 1]

    generator = SimpleGenerator().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=1e-3)

    # Generate fixed reference images
    num_samples = 10
    ref_images = []
    ref_inputs = []
    for i in range(num_samples):
        _, clean, _ = generate_synthetic_pair(size=image_size, num_defects=0, seed=100 + i)
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1)).float().unsqueeze(0)
        # Input: slightly noised version
        noise = torch.randn_like(clean_t) * 0.1
        ref_inputs.append(clean_t + noise)
        ref_images.append(clean_t)

    ref_inputs_cat = torch.cat(ref_inputs, dim=0).to(device)
    ref_images_cat = torch.cat(ref_images, dim=0).to(device)

    checkpoint_outputs = {}

    for epoch in range(1, epochs + 1):
        # Train
        generator.train()
        for step in range(50):  # 50 steps per epoch
            idx = step % num_samples
            inp = ref_inputs_cat[idx:idx+1]
            gt = ref_images_cat[idx:idx+1]
            out = generator(inp)
            loss = F.mse_loss(out, (gt * 2 - 1))  # tanh range [-1, 1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint output
        if epoch in [1, 3, 5]:
            generator.eval()
            with torch.no_grad():
                all_outs = []
                for i in range(num_samples):
                    inp = ref_inputs_cat[i:i+1]
                    out = generator(inp)
                    img = (out.squeeze(0) * 0.5 + 0.5).clamp(0, 1)  # tanh → [0,1]
                    all_outs.append(img.cpu())
                checkpoint_outputs[epoch] = all_outs

    if verbose:
        print(f"  [Checkpoints] Saved outputs at epochs {sorted(checkpoint_outputs.keys())}")

    return checkpoint_outputs, [r.squeeze(0).cpu() for r in ref_images]


def create_checkpoint_degraded_pairs(checkpoint_outputs, clean_images, image_size):
    """
    From historical checkpoints, create (dirty, clean, mask) tuples.
    Epoch 1 outputs → degraded, epoch 5 → GT.
    The mask marks the differences between epoch 1 and epoch 5.
    """
    pairs = []
    epoch1_outs = checkpoint_outputs[1]
    epoch5_outs = checkpoint_outputs[5]

    for e1, e5 in zip(epoch1_outs, epoch5_outs):
        e1_np = e1.numpy().transpose(1, 2, 0)
        e5_np = e5.numpy().transpose(1, 2, 0)

        diff = np.abs(e1_np - e5_np).mean(axis=2, keepdims=True)
        mask = (diff > 0.05).astype(np.float32)
        if mask.sum() == 0:
            mask[0, 0, 0] = 1.0

        dirty = e1_np.astype(np.float32)
        clean = e5_np.astype(np.float32)
        pairs.append((dirty, clean, mask))

    return pairs


# =============================================================================
# Per-Degradation Training
# =============================================================================

class DegradationDataset(torch.utils.data.Dataset):
    """Dataset for a specific degradation type on synthetic images."""

    def __init__(self, num_samples, image_size, degradation_fn, seed=42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.degradation_fn = degradation_fn
        np.random.seed(seed)
        self.seeds = np.random.randint(0, 2**31 - 1, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(self.seeds[idx])
        size = self.image_size
        clean_np = np.random.rand(size, size, 3).astype(np.float32) * 0.5 + 0.25
        degraded_np, mask_np = self.degradation_fn(clean_np)

        return (torch.from_numpy(degraded_np.transpose(2, 0, 1)).float(),
                torch.from_numpy(clean_np.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask_np.transpose(2, 0, 1)).float())


def make_block_fn(size):
    def fn(img):
        from common.data_utils import _apply_block_occlusion
        return _apply_block_occlusion(img)
    return fn

def make_blur_fn(size):
    def fn(img):
        return apply_gaussian_blur(img, num_blobs=3, max_sigma=5.0)
    return fn

def make_jpeg_fn(size):
    def fn(img):
        return apply_jpeg_artifact(img, num_regions=3, quality=10)
    return fn

def make_adversarial_fn(deg_net, device):
    def fn(img):
        clean = img.copy()
        degraded, mask = _apply_block_occlusion(img)
        adv_degraded, adv_mask = apply_adversarial_degradation(degraded, clean, deg_net, device)
        combined_mask = np.maximum(mask, adv_mask)
        return adv_degraded.astype(np.float32), combined_mask.astype(np.float32)
    return fn


def train_psrnet_for_degradation(degradation_name, degradation_fn, config, device, verbose=True):
    """Train PSR-Net for a specific degradation type and return model + history."""
    if verbose:
        print(f"\n  [{degradation_name}] Training PSR-Net...")

    train_dataset = DegradationDataset(config.train_samples, config.image_size,
                                        degradation_fn, seed=config.seed)
    test_dataset = DegradationDataset(config.test_samples, config.image_size,
                                       degradation_fn, seed=config.seed + 1000)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = create_model("standard", base_channels=config.base_channels,
                         input_channels=config.input_channels, device=device)
    engine = TrainingEngine(model, config, device)
    history = engine.train(train_loader, test_loader, verbose=verbose,
                           val_freq=max(1, config.epochs // 10),
                           save_dir=config.output_dir)

    return model, history, test_loader


# =============================================================================
# Visualization
# =============================================================================

def plot_degradation_comparison(results_data, save_path):
    """
    4x4 grid: row=degradation type, col = dirty/refined/mask/GT
    results_data: {"degradation_name": {"dirty": tensor, "refined": tensor,
                     "mask": tensor, "gt": tensor}}
    """
    deg_names = list(results_data.keys())
    n_rows = len(deg_names)
    n_cols = 4  # dirty, refined, mask, GT
    col_titles = ["Dirty", "Refined", "Mask", "GT"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r, name in enumerate(deg_names):
        data = results_data[name]
        axes[r, 0].imshow(tensor_to_numpy(data["dirty"]))
        axes[r, 0].set_ylabel(name, fontsize=10, rotation=90, va="center", labelpad=20)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(tensor_to_numpy(data["refined"]))
        axes[r, 1].axis("off")

        axes[r, 2].imshow(tensor_to_numpy(data["mask"]), cmap="hot")
        axes[r, 2].axis("off")

        axes[r, 3].imshow(tensor_to_numpy(data["gt"]))
        axes[r, 3].axis("off")

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold")

    plt.suptitle("A1: Degradation Type Comparison — PSR-Net Refinement", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="A1: Adversarial Degradation Simulation")
    parser.add_argument("--epochs", type=int, default=80, help="PSR-Net training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--degnet_epochs", type=int, default=40, help="DegradationNet training epochs")
    parser.add_argument("--checkpoint_epochs", type=int, default=5, help="Historical checkpoint generator epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    config = get_config("A1",
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        image_size=args.image_size)
    config.output_dir = output_dir
    config.device = device.__str__()

    print(f"\n{'='*60}")
    print(f"  Experiment A1: Adversarial Degradation Simulation")
    print(f"  Image size: {config.image_size}, Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"{'='*60}")

    # =====================================================================
    # Step 1: Train DegradationNet for adversarial degradation
    # =====================================================================
    print("\n[Step 1] Training DegradationNet (adversarial artifact generator)...")
    t0 = time.time()
    deg_net = train_degradation_net(device, image_size=config.image_size,
                                     epochs=args.degnet_epochs,
                                     batch_size=config.batch_size,
                                     save_dir=os.path.join(output_dir, "degnet_ckpt"))
    deg_net.eval()
    print(f"  DegradationNet trained in {time.time() - t0:.1f}s")

    # =====================================================================
    # Step 2: Define four degradation types
    # =====================================================================
    degradation_types = {
        "Block": make_block_fn(config.image_size),
        "GaussianBlur": make_blur_fn(config.image_size),
        "JPEG": make_jpeg_fn(config.image_size),
        "Adversarial": make_adversarial_fn(deg_net, device),
    }

    # =====================================================================
    # Step 3: Train PSR-Net for each degradation type
    # =====================================================================
    print("\n[Step 2] Training PSR-Net for each degradation type...")
    trained_models = {}
    results_metrics = {}

    for deg_name, deg_fn in degradation_types.items():
        t0 = time.time()
        model, history, test_loader = train_psrnet_for_degradation(
            deg_name, deg_fn, config, device)
        elapsed = time.time() - t0
        trained_models[deg_name] = model

        # Full evaluation on test set
        model.eval()
        refined_list, dirty_list, clean_list, mask_list, gt_mask_list = [], [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                dirty, clean, gt_mask = [b.to(device) for b in batch]
                refined, _, mask = model.refine(dirty)
                refined_list.append(refined)
                dirty_list.append(dirty)
                clean_list.append(clean)
                mask_list.append(mask)
                gt_mask_list.append(gt_mask)

        # Concatenate batches
        refined_cat = torch.cat(refined_list, dim=0)
        dirty_cat = torch.cat(dirty_list, dim=0)
        clean_cat = torch.cat(clean_list, dim=0)
        mask_cat = torch.cat(mask_list, dim=0)
        gt_mask_cat = torch.cat(gt_mask_list, dim=0)

        eval_metrics = evaluate_all(
            [refined_cat[i] for i in range(len(refined_cat))],
            [clean_cat[i] for i in range(len(clean_cat))],
            dirty_list=[dirty_cat[i] for i in range(len(dirty_cat))],
            masks=[mask_cat[i] for i in range(len(mask_cat))],
            gt_masks=[gt_mask_cat[i] for i in range(len(gt_mask_cat))],
            metrics=["psnr", "ssim", "l1", "mask_contrast", "mask_mean", "iou"],
        )

        results_metrics[deg_name] = {
            **eval_metrics,
            "training_time_s": round(elapsed, 1),
        }
        print(f"  [{deg_name}] PSNR={eval_metrics.get('psnr_mean', 0):.2f} dB, "
              f"SSIM={eval_metrics.get('ssim_mean', 0):.4f}, "
              f"Time={elapsed:.1f}s")

    # =====================================================================
    # Step 4: Historical Checkpoint Degradation
    # =====================================================================
    print("\n[Step 3] Simulating Historical Checkpoint Degradation...")
    t0 = time.time()
    checkpoint_outputs, clean_images = simulate_historical_checkpoints(
        device, image_size=config.image_size,
        epochs=args.checkpoint_epochs, batch_size=config.batch_size)
    print(f"  Checkpoint simulation done in {time.time() - t0:.1f}s")

    # Create pairs and train PSR-Net on checkpoint data
    checkpoint_pairs = create_checkpoint_degraded_pairs(
        checkpoint_outputs, clean_images, config.image_size)

    # Train PSR-Net on checkpoint degraded data
    cp_config = get_config("A1", epochs=config.epochs, batch_size=min(4, len(checkpoint_pairs)),
                           image_size=config.image_size)
    cp_config.output_dir = output_dir
    cp_config.device = device.__str__()

    # Simple dataset from checkpoint pairs
    class CheckpointDataset(torch.utils.data.Dataset):
        def __init__(self, pairs, num_repeats=40):
            self.pairs = pairs
            self.num_repeats = num_repeats

        def __len__(self):
            return len(self.pairs) * self.num_repeats

        def __getitem__(self, idx):
            p_idx = idx % len(self.pairs)
            dirty, clean, mask = self.pairs[p_idx]
            return (torch.from_numpy(dirty.transpose(2, 0, 1)).float(),
                    torch.from_numpy(clean.transpose(2, 0, 1)).float(),
                    torch.from_numpy(mask.transpose(2, 0, 1)).float())

    cp_train = CheckpointDataset(checkpoint_pairs, num_repeats=40)
    cp_test = CheckpointDataset(checkpoint_pairs, num_repeats=4)
    cp_train_loader = DataLoader(cp_train, batch_size=4, shuffle=True)
    cp_test_loader = DataLoader(cp_test, batch_size=4, shuffle=False)

    print("\n  [Checkpoint] Training PSR-Net on historical checkpoint data...")
    cp_model = create_model("standard", base_channels=config.base_channels,
                             input_channels=config.input_channels, device=device)
    cp_engine = TrainingEngine(cp_model, cp_config, device)
    cp_history = cp_engine.train(cp_train_loader, cp_test_loader,
                                  val_freq=max(1, cp_config.epochs // 10),
                                  save_dir=output_dir)
    trained_models["HistoricalCheckpoint"] = cp_model

    # Evaluate checkpoint model
    cp_model.eval()
    cp_refined, cp_dirty, cp_clean, cp_mask, cp_gtmask = [], [], [], [], []
    with torch.no_grad():
        for batch in cp_test_loader:
            dirty, clean, gt_mask = [b.to(device) for b in batch]
            refined, _, mask = cp_model.refine(dirty)
            cp_refined.append(refined)
            cp_dirty.append(dirty)
            cp_clean.append(clean)
            cp_mask.append(mask)
            cp_gtmask.append(gt_mask)

    cp_refined_cat = torch.cat(cp_refined, dim=0)
    cp_dirty_cat = torch.cat(cp_dirty, dim=0)
    cp_clean_cat = torch.cat(cp_clean, dim=0)
    cp_mask_cat = torch.cat(cp_mask, dim=0)
    cp_gtmask_cat = torch.cat(cp_gtmask, dim=0)

    cp_metrics = evaluate_all(
        [cp_refined_cat[i] for i in range(len(cp_refined_cat))],
        [cp_clean_cat[i] for i in range(len(cp_clean_cat))],
        dirty_list=[cp_dirty_cat[i] for i in range(len(cp_dirty_cat))],
        masks=[cp_mask_cat[i] for i in range(len(cp_mask_cat))],
        gt_masks=[cp_gtmask_cat[i] for i in range(len(cp_gtmask_cat))],
        metrics=["psnr", "ssim", "l1", "mask_contrast", "mask_mean", "iou"],
    )
    results_metrics["HistoricalCheckpoint"] = cp_metrics

    # =====================================================================
    # Step 5: Generate visualization
    # =====================================================================
    print("\n[Step 4] Generating visualizations...")

    # Collect one representative sample per degradation type
    viz_data = {}
    for deg_name, model in trained_models.items():
        if deg_name == "HistoricalCheckpoint":
            # Use first sample from checkpoint data
            d = cp_dirty_cat[0:1].to(device)
            c = cp_clean_cat[0:1].to(device)
            with torch.no_grad():
                ref, _, m = model.refine(d)
            viz_data[deg_name] = {
                "dirty": d.squeeze(0),
                "refined": ref.squeeze(0),
                "mask": m.squeeze(0),
                "gt": c.squeeze(0),
            }
        else:
            # Generate a single test sample
            clean_np = np.random.rand(config.image_size, config.image_size, 3).astype(np.float32) * 0.5 + 0.25
            degraded_np, mask_np = degradation_types[deg_name](clean_np)
            d_t = torch.from_numpy(degraded_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            c_t = torch.from_numpy(clean_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                ref, _, m = model.refine(d_t)
            viz_data[deg_name] = {
                "dirty": d_t.squeeze(0),
                "refined": ref.squeeze(0),
                "mask": m.squeeze(0),
                "gt": c_t.squeeze(0),
            }

    comparison_path = os.path.join(output_dir, "degradation_comparison.png")
    plot_degradation_comparison(viz_data, comparison_path)
    print(f"  Saved: {comparison_path}")

    # =====================================================================
    # Step 6: Save results
    # =====================================================================
    print("\n[Step 5] Saving results...")

    # Clean non-serializable values
    clean_metrics = {}
    for deg_name, metrics in results_metrics.items():
        entry = {}
        for k, v in metrics.items():
            if isinstance(v, (float, np.floating)) and (np.isinf(v) or np.isnan(v)):
                entry[k] = None
            else:
                entry[k] = round(float(v), 6) if isinstance(v, (float, np.floating)) else v
        clean_metrics[deg_name] = entry

    results_json_path = os.path.join(output_dir, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "A1",
            "description": "Adversarial Degradation Simulation & Historical Checkpoint Degradation",
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "image_size": config.image_size,
                "lambda_sparse": config.lambda_sparse,
                "device": device.__str__(),
            },
            "metrics": clean_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_json_path}")

    # =====================================================================
    # Summary Table
    # =====================================================================
    print(f"\n{'='*80}")
    print(f"  A1 Results Summary")
    print(f"{'='*80}")
    header = f"{'Degradation Type':28s} {'PSNR(dB)':>10s} {'SSIM':>8s} {'MaskContrast':>14s} {'IoU':>8s} {'Time(s)':>8s}"
    print(header)
    print("-" * 80)
    for deg_name in results_metrics:
        m = results_metrics[deg_name]
        print(f"  {deg_name:26s} {m.get('psnr_mean', 0):>10.2f} {m.get('ssim_mean', 0):>8.4f} "
              f"{m.get('mask_contrast_ratio_mean', 0):>14.2f} {m.get('iou_mean', 0):>8.4f} "
              f"{m.get('training_time_s', 0):>8.1f}")
    print(f"{'='*80}")
    print(f"\n  Output directory: {output_dir}")
    print(f"  Results JSON: {results_json_path}")
    print(f"  Visualization: {comparison_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
