"""
Unified training engine -- shared training loop for all experiments
统一训练引擎 —— 所有实验共用的训练循环
v2.0: Periodic checkpoint saving + tqdm progress bar + resume training
v2.0: 定期保存检查点 + tqdm 进度条 + 断点续训
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict

try:
    from .evaluation import compute_psnr, compute_ssim, compute_mask_contrast_ratio, sanitize_metric_array
except ImportError:
    from evaluation import compute_psnr, compute_ssim, compute_mask_contrast_ratio, sanitize_metric_array


# ── Progress bar (no dependencies, compatible with all environments) ─────────
# ── 进度条（无依赖，兼容所有环境）──────────────────────────────────────────
def _format_time(seconds: float) -> str:
    """Format time duration
    格式化时间"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m{int(seconds%60)}s"
    return f"{int(seconds//3600)}h{int((seconds%3600)//60)}m"


def _progress_bar(current: int, total: int, prefix: str = "", 
                  suffix: str = "", width: int = 30) -> str:
    """Simple progress bar (standalone, no tqdm dependency)
    简易进度条（不依赖 tqdm）"""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    return f"\r{prefix} |{bar}| {pct:.0f}% {current}/{total} {suffix}"


# ── Checkpoint Manager ──────────────────────────────────────────────────────
# ── 检查点管理器 ────────────────────────────────────────────────────────────
class CheckpointManager:
    """
    General checkpoint manager, shared by all custom training loops
    通用检查点管理器，所有自定义训练循环共用

    Usage:
        ckpt = CheckpointManager(save_dir, keep_last=3)
        Call every N epochs: ckpt.save(model, optimizer, epoch, history, metrics)
        Restore: state = ckpt.load_latest(model, optimizer)  # returns epoch
    用法:
        ckpt = CheckpointManager(save_dir, keep_last=3)
        每 N epoch 调用: ckpt.save(model, optimizer, epoch, history, metrics)
        恢复: state = ckpt.load_latest(model, optimizer)  # 返回 epoch
    """
    
    def __init__(self, save_dir: str, keep_last: int = 5, 
                 save_best: bool = True, best_metric: str = "psnr"):
        self.save_dir = save_dir
        self.keep_last = keep_last
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_value = -float("inf")
        self.saved_ckpts: List[str] = []
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model: nn.Module, optimizer: optim.Optimizer,
             epoch: int, history: dict, metrics: Dict[str, float] = None,
             scheduler: Optional[object] = None) -> str:
        """Save checkpoint with automatic old file rotation. If metric exceeds best value, also save best_model.pt.
        保存检查点，自动轮换旧文件。若指标优于历史最佳，额外保存 best_model.pt。"""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "metrics": metrics or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            ckpt["scheduler_state_dict"] = scheduler.state_dict()
        
        # Periodic save (with I/O retry + atomic write, resilient to NFS transient errors)
        # 定期保存（带 I/O 重试 + 原子写入，应对 NFS 瞬时错误）
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch:04d}.pt")
        tmp_path = path + ".tmp"
        last_err = None
        for _attempt in range(5):
            try:
                torch.save(ckpt, tmp_path)
                if os.path.exists(path):
                    os.remove(path)
                os.rename(tmp_path, path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                import time as _time
                _time.sleep(2.0)
                continue
        if last_err is not None:
            print(f"  [Checkpoint] WARN: failed to save {path} after 5 attempts: {last_err}")
        self.saved_ckpts.append(path)
        
        # Remove old checkpoints (keep most recent N)
        # 删除旧检查点（保留最近 N 个）
        while len(self.saved_ckpts) > self.keep_last:
            old = self.saved_ckpts.pop(0)
            if os.path.exists(old):
                os.remove(old)
        
        # Save best model
        # 保存最佳模型
        if self.save_best and metrics and self.best_metric in metrics:
            val = metrics[self.best_metric]
            if val > self.best_value:
                self.best_value = val
                best_path = os.path.join(self.save_dir, "best_model.pt")
                best_tmp = best_path + ".tmp"
                for _attempt in range(5):
                    try:
                        torch.save(ckpt, best_tmp)
                        if os.path.exists(best_path):
                            os.remove(best_path)
                        os.rename(best_tmp, best_path)
                        break
                    except Exception:
                        import time as _time
                        _time.sleep(2.0)
                        continue
                is_best = True
        
        # Save training history (JSON, for monitoring)
        # 保存历史记录（JSON，用于监控）
        if history:
            hist_path = os.path.join(self.save_dir, "training_history.json")
            try:
                with open(hist_path, "w") as f:
                    json.dump({k: ([float(x) if isinstance(x, (np.floating, np.integer)) else x 
                                     for x in v] if isinstance(v, list) else v)
                               for k, v in history.items()}, f, indent=2)
            except:
                pass
        
        return path
    
    def load(self, path: str, model: nn.Module, 
             optimizer: Optional[optim.Optimizer] = None,
             scheduler: Optional[object] = None) -> Tuple[int, dict]:
        """Restore from specified checkpoint
        从指定检查点恢复"""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt.get("epoch", -1) + 1, ckpt.get("history", {})
    
    def load_latest(self, model: nn.Module, 
                    optimizer: Optional[optim.Optimizer] = None,
                    scheduler: Optional[object] = None) -> Tuple[int, dict]:
        """Load latest checkpoint, returns (start_epoch, history)
        加载最新检查点，返回 (start_epoch, history)"""
        ckpts = sorted([f for f in os.listdir(self.save_dir) 
                        if f.startswith("checkpoint_epoch_") and f.endswith(".pt")])
        if not ckpts:
            # Try best_model
            # 尝试 best_model
            best_path = os.path.join(self.save_dir, "best_model.pt")
            if os.path.exists(best_path):
                return self.load(best_path, model, optimizer, scheduler)
            return 0, {}
        
        latest = ckpts[-1]
        print(f"  [Checkpoint] Resumed from: {latest}")
        print(f"  [Checkpoint] 恢复自: {latest}")
        return self.load(os.path.join(self.save_dir, latest), model, optimizer, scheduler)


# ── TrainingEngine v2 ──────────────────────────────────────────────────────
class TrainingEngine:
    """
    PSR-Net Training Engine v2.0
    PSR-Net 训练引擎 v2.0

    New features:
    新增:
    - Periodic checkpoint saving (save_freq)
    - Batch-level progress bar (show_progress)
    - Resume training (resume_from)
    - Automatic training history JSON export
    - ETA estimation
    """
    
    def __init__(self, model: nn.Module, config, device: str = "cpu"):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs)
        self.history = defaultdict(list)
        self.checkpoint_manager: Optional[CheckpointManager] = None
    
    def _get_lambda_sparse(self, epoch: int) -> float:
        lambda_target = self.config.lambda_sparse
        warmup = getattr(self.config, 'warmup_epochs', 60)
        if epoch < warmup:
            return lambda_target * (epoch / warmup)
        return lambda_target

    def _get_lambda_distill(self) -> float:
        """Auto-read lambda_distill from config; returns 0.0 if not configured."""
        return float(getattr(self.config, 'lambda_distill', 0.0) or 0.0)
    
    def train_epoch(self, train_loader: DataLoader, epoch: int,
                     lambda_distill: float | None = None,
                     show_progress: bool = False) -> Dict[str, float]:
        """Train for one epoch (optional batch-level progress bar)
        训练一个 epoch（可选 batch 级进度条）

        lambda_distill: Mask distillation weight. If None, auto-read from config.lambda_distill.
        lambda_distill: mask 蒸馏权重。若为 None，自动从 config.lambda_distill 读取。
        Set to 0.01~0.1 to effectively prevent M->0 mask collapse.
        设为 0.01~0.1 可有效防止 M->0 掩膜崩溃。
        """
        if lambda_distill is None:
            lambda_distill = self._get_lambda_distill()
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0
        total_batches = len(train_loader)

        lambda_sparse = self._get_lambda_sparse(epoch)

        for batch_idx, batch in enumerate(train_loader):
            dirty, clean, gt_mask = [b.to(self.device) for b in batch]

            residual, mask = self.model(dirty)
            refined = dirty + residual * mask

            loss_l1 = nn.functional.l1_loss(refined, clean)
            loss_sparse = lambda_sparse * mask.mean()
            # Strong mask supervision via BCE to prevent collapse
            loss_distill = lambda_distill * nn.functional.binary_cross_entropy(
                mask.clamp(1e-6, 1-1e-6), gt_mask)
            total_loss = loss_l1 + loss_sparse + loss_distill

            if lambda_distill > 0:
                epoch_losses["distill"] += loss_distill.item()

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["l1"] += loss_l1.item()
            epoch_losses["sparse"] += loss_sparse.item()
            epoch_losses["mask_mean"] += mask.mean().item()
            n_batches += 1
            
            # Batch-level progress display
            # Batch 级进度显示
            if show_progress and (batch_idx % max(1, total_batches // 20) == 0 
                                  or batch_idx == total_batches - 1):
                avg_loss = epoch_losses["total"] / n_batches
                print(_progress_bar(batch_idx + 1, total_batches,
                      prefix=f"  Epoch {epoch:3d}", 
                      suffix=f"Loss={avg_loss:.4f} M={epoch_losses['mask_mean']/n_batches:.4f}"),
                      end="")
        
        if show_progress:
            print()  # newline / 换行
        
        results = {k: v / n_batches for k, v in epoch_losses.items()}
        results["lambda_sparse"] = lambda_sparse
        return results
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics = defaultdict(list)
        for batch in val_loader:
            dirty, clean, gt_mask = [b.to(self.device) for b in batch]
            refined, residual, mask = self.model.refine(dirty)
            metrics["psnr"].append(compute_psnr(refined, clean))
            metrics["ssim"].append(compute_ssim(refined, clean))
            metrics["l1_loss"].append(nn.functional.l1_loss(refined, clean).item())
            metrics["mask_mean"].append(mask.mean().item())
            metrics["mask_contrast"].append(compute_mask_contrast_ratio(mask, gt_mask))
        return {k: float(np.mean(sanitize_metric_array(v))) if sanitize_metric_array(v) else 0.0
                for k, v in metrics.items()}
    
    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              lambda_distill: float | None = None,
              verbose: bool = True,
              show_progress: bool = True,
              val_freq: int = 5,
              save_freq: int = 5,
              save_dir: str = ".",
              resume_from: Optional[str] = None,
              keep_checkpoints: int = 5) -> Dict:
        """
        Full training loop v2.0
        完整训练循环 v2.0

        Args:
            train_loader: Training DataLoader
            train_loader: 训练 DataLoader
            val_loader: Validation DataLoader (optional)
            val_loader: 验证 DataLoader（可选）
            lambda_distill: Distillation loss weight. None=auto-read from config (default 0.0 matching paper); 0.01~0.1=prevent mask collapse
            lambda_distill: 蒸馏损失权重。None=自动从config读取(默认0.0匹配论文)；0.01~0.1=防止掩膜崩溃
            verbose: Whether to print epoch-level logs
            verbose: 是否打印 epoch 级日志
            show_progress: Whether to show batch-level progress bar
            show_progress: 是否显示 batch 级进度条
            val_freq: Validate every N epochs
            val_freq: 每隔 N epoch 验证一次
            save_freq: Save checkpoint every N epochs
            save_freq: 每隔 N epoch 保存检查点
            save_dir: Checkpoint save directory
            save_dir: 检查点保存目录
            resume_from: Resume training (None=no resume, "latest"=latest, or specific path)
            resume_from: 恢复训练（None=不恢复, "latest"=最新, 或指定路径）
            keep_checkpoints: Keep most recent N checkpoints
            keep_checkpoints: 保留最近 N 个检查点

        Returns:
            history: Training history dictionary
            history: 训练历史字典
        """
        # ── Setup checkpoint manager ──
        # ── 设置检查点管理器 ──
        self.checkpoint_manager = CheckpointManager(
            save_dir, keep_last=keep_checkpoints)
        
        start_epoch = 0
        
        # ── Resume training ──
        # ── 断点续训 ──
        if resume_from:
            if resume_from == "latest":
                start_epoch, loaded_history = self.checkpoint_manager.load_latest(
                    self.model, self.optimizer)
            else:
                start_epoch, loaded_history = self.checkpoint_manager.load(
                    resume_from, self.model, self.optimizer)
            
            if loaded_history:
                self.history.update(loaded_history)
            print(f"  [Resume] Resuming from epoch {start_epoch}")
            print(f"  [Resume] 从 epoch {start_epoch} 继续训练")
        
        # ── Main training loop ──
        # ── 训练主循环 ──
        total_epochs = self.config.epochs
        train_start = time.time()
        
        for epoch in range(start_epoch, total_epochs):
            t0 = time.time()
            
            # Train
            # 训练
            train_results = self.train_epoch(
                train_loader, epoch, lambda_distill, show_progress=show_progress)
            
            # Record metrics
            # 记录
            self.history["epoch"].append(epoch)
            self.history["lambda_sparse"].append(train_results["lambda_sparse"])
            self.history["train_loss"].append(train_results["total"])
            self.history["train_l1"].append(train_results["l1"])
            self.history["mask_mean"].append(train_results["mask_mean"])
            
            # Validate
            # 验证
            val_results = {}
            if val_loader and (epoch % val_freq == 0 or epoch == total_epochs - 1):
                val_results = self.validate(val_loader)
                for k, v in val_results.items():
                    self.history.setdefault(f"val_{k}", []).append(v)
                    self.history.setdefault(f"val_epoch_{k}", []).append(epoch)
            
            self.scheduler.step()

            # ── Mask collapse warning ──
            # ── 掩膜崩溃预警 ──
            _mask_mean = train_results.get('mask_mean', 1.0)
            if _mask_mean < 1e-7 and epoch > getattr(self.config, 'warmup_epochs', 60):
                print(f"  \u26a0\ufe0f [Mask Collapse] Epoch {epoch}: mask_mean={_mask_mean:.2e} < 1e-7!")
                print(f"     Mask collapsed to zero. Recommendations:")
                print(f"     \u63a9\u819c\u5df2\u5d29\u7f29\u81f3\u96f6\uff0c\u5efa\u8bae\uff1a")
                print(f"     1) Enable mask distillation: lambda_distill=0.05")
                print(f"     1) \u542f\u7528\u63a9\u819c\u84b8\u998f: lambda_distill=0.05")
                print(f"     2) Reduce sparsity penalty: lambda_sparse=0.01")
                print(f"     2) \u964d\u4f4e\u7a00\u758f\u60e9\u7f5a: lambda_sparse=0.01")
                print(f"     3) Extend warmup: warmup_epochs=80")
                print(f"     3) \u5ef6\u957f warmup: warmup_epochs=80")

            # ── Periodic checkpoint save ──
            # ── 定期保存检查点 ──
            if epoch % save_freq == 0 or epoch == total_epochs - 1:
                all_metrics = {**train_results, **val_results}
                self.checkpoint_manager.save(
                    self.model, self.optimizer, epoch, 
                    dict(self.history), all_metrics)
                
                if verbose:
                    print(f"  [Checkpoint] Epoch {epoch} saved → {save_dir}")
                    print(f"  [Checkpoint] Epoch {epoch} 已保存 → {save_dir}")
            
            # ── Epoch logging ──
            # ── Epoch 日志 ──
            if verbose:
                elapsed = time.time() - t0
                
                # Progress percentage
                # 进度百分比
                done = epoch + 1 - start_epoch
                total = total_epochs - start_epoch
                pct = done / max(total, 1) * 100
                
                # ETA estimation
                # ETA 预估
                elapsed_total = time.time() - train_start
                eta_seconds = (elapsed_total / done) * (total - done) if done > 0 else 0
                
                log_parts = [f"Epoch {epoch:3d}/{total_epochs}"]
                log_parts.append(f"[{pct:3.0f}%]")
                log_parts.append(f"Loss={train_results['total']:.4f}")
                if train_results["lambda_sparse"] > 0:
                    log_parts.append(f"λ_s={train_results['lambda_sparse']:.4f}")
                log_parts.append(f"Mμ={train_results['mask_mean']:.4f}")
                if val_results:
                    log_parts.append(f"PSNR={val_results.get('psnr', 0):.1f}")
                log_parts.append(f"t={elapsed:.1f}s")
                if eta_seconds > 0:
                    log_parts.append(f"ETA={_format_time(eta_seconds)}")
                print(" | ".join(log_parts))
        
        # ── Training complete, final save ──
        # ── 训练完成，最终保存 ──
        final_path = os.path.join(save_dir, "final_model.pt")
        torch.save({
            "epoch": total_epochs - 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": dict(self.history),
            "config": {k: str(v) for k, v in self.config.__dict__.items()},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, final_path)
        
        total_time = time.time() - train_start
        print(f"\n  ✓ Training complete! Total time: {_format_time(total_time)}")
        print(f"\n  ✓ 训练完成！总耗时: {_format_time(total_time)}")
        print(f"  Final model: {final_path}")
        print(f"  最终模型: {final_path}")
        print(f"  Checkpoint directory: {save_dir}")
        print(f"  检查点目录: {save_dir}")
        
        return dict(self.history)


# ── Utility Functions ────────────────────────────────────────────────────────
# ── 便捷函数 ──────────────────────────────────────────────────────────────────
def train_with_config(model_factory: Callable, config, 
                       train_loader: DataLoader,
                       val_loader: Optional[DataLoader] = None,
                       **train_kwargs) -> Tuple[nn.Module, Dict]:
    """Create model + train + return
    创建模型 + 训练 + 返回"""
    model = model_factory().to(config.device)
    engine = TrainingEngine(model, config, config.device)
    history = engine.train(train_loader, val_loader, **train_kwargs)
    return model, history


def run_multi_seed_training(model_factory: Callable, config,
                             train_loader_factory: Callable,
                             seeds: List[int] = None,
                             save_base_dir: str = ".",
                             **train_kwargs) -> Dict:
    """Multi-seed training (A8), each seed saved independently
    多随机种子训练（A8），每个种子独立保存"""
    if seeds is None:
        seeds = [42, 123, 456]
    
    all_histories = {}
    results_per_seed = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Seed {i+1}/{len(seeds)}: seed={seed}")
        print(f"{'='*60}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_loader, val_loader = train_loader_factory(seed=seed)
        
        model = model_factory().to(config.device)
        seed_dir = os.path.join(save_base_dir, f"seed_{seed}")
        
        engine = TrainingEngine(model, config, config.device)
        history = engine.train(train_loader, val_loader, 
                               save_dir=seed_dir, **train_kwargs)
        
        all_histories[f"seed_{seed}"] = history
        
        final_metrics = {}
        for k, v in history.items():
            if k.startswith("val_") and not k.startswith("val_epoch_"):
                final_metrics[k] = v[-1] if v else 0
        
        results_per_seed.append({
            "seed": seed,
            "params": sum(p.numel() for p in model.parameters()),
            **final_metrics,
        })
    
    # Summary / 汇总
    summary = defaultdict(list)
    for r in results_per_seed:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                summary[k].append(v)
    
    stats = {}
    for k, v in summary.items():
        if len(v) > 1:
            stats[f"{k}_mean"] = float(np.mean(v))
            stats[f"{k}_std"] = float(np.std(v))
    
    return {
        "histories": all_histories,
        "per_seed_results": results_per_seed,
        "summary": stats,
    }


# ── Print Training Config Summary ───────────────────────────────────────────
# ── 打印训练配置摘要 ────────────────────────────────────────────────────────
def print_config_summary(config, extra: dict = None):
    """Print training configuration overview at startup
    启动时打印训练配置概览"""
    device_str = str(config.device)
    has_cuda = torch.cuda.is_available()
    
    print(f"\n{'='*60}")
    print(f"  PSR-Net Training Configuration")
    print(f"  PSR-Net 训练配置")
    print(f"{'='*60}")
    print(f"  Experiment: {getattr(config, 'name', 'unnamed')}")
    print(f"  实验: {getattr(config, 'name', 'unnamed')}")
    print(f"  Device: {device_str} {'(CUDA available)' if has_cuda else '(CPU only)'}")
    print(f"  设备: {device_str} {'(CUDA available)' if has_cuda else '(CPU only)'}")
    if has_cuda:
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Image size: {config.image_size}x{config.image_size}")
    print(f"  分辨率: {config.image_size}x{config.image_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.lr}")
    print(f"  λ_s (sparsity): {config.lambda_sparse}")
    print(f"  Warmup: {getattr(config, 'warmup_epochs', 40)} epochs")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    print(f"{'='*60}\n")
