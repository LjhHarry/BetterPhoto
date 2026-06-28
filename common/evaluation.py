"""
Unified evaluation module — supports all metrics required by the paper
(PSNR, SSIM, LPIPS, FID, IoU, mask contrast ratio, L1 improvement rate, pixel fidelity, inference time/memory)
统一评估模块 —— 支持所有论文所需指标
PSNR, SSIM, LPIPS, FID, IoU, 掩膜对比度比率, L1改进率, 像素保真率, 推理时间/显存
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ---- PSNR ----
def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR (Peak Signal-to-Noise Ratio)
    计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


# ---- SSIM ----
def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11, size_average: bool = True) -> float:
    """
    Calculate SSIM (Structural Similarity Index)
    Implemented using a sliding window
    计算 SSIM (Structural Similarity Index)
    使用滑动窗口实现
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Ensure at least a batch dimension
    # 确保至少有 batch 维度
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # Gaussian window
    # 高斯窗口
    channel = pred.size(1)
    window = _gaussian_window(window_size, 1.5).to(pred.device)
    window = window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean().item()) if size_average else float(ssim_map.mean([1,2,3]).cpu().numpy())


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    """Generate Gaussian window
    生成高斯窗口"""
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2*sigma**2))
                          for x in range(window_size)])
    return (gauss / gauss.sum()).unsqueeze(0).unsqueeze(0) * \
           (gauss / gauss.sum()).unsqueeze(1).unsqueeze(0)


# ---- LPIPS (simplified implementation, approximate using pre-trained features) ----
# ---- LPIPS (简化实现，使用预训练特征的近似) ----
def compute_lpips_approx(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Simplified LPIPS (Learned Perceptual Image Patch Similarity)
    Approximates perceptual difference using multi-scale L1 distance.
    Full implementation requires loading AlexNet/VGG pre-trained weights.
    简化版 LPIPS (Learned Perceptual Image Patch Similarity)
    使用多尺度 L1 距离近似感知差异。
    完整实现需加载 AlexNet/VGG 预训练权重。
    """
    # Use multi-scale differences as approximation of perceptual similarity
    # 使用多尺度差异作为感知相似性的近似
    scales = [1, 2, 4]
    total = 0.0
    for s in scales:
        if s > 1:
            p = F.avg_pool2d(pred, kernel_size=s, stride=s)
            t = F.avg_pool2d(target, kernel_size=s, stride=s)
        else:
            p, t = pred, target
        total += F.l1_loss(p, t).item()
    return total / len(scales)


# ---- FID helper functions (simplified approximation — non-standard implementation) ----
# ---- FID 辅助函数 (简化近似 — 非标准实现) ----
def compute_activation_stats_deprecated(images: torch.Tensor, model=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED: Uses flattened pixel values as feature proxy (non-standard implementation).
    Use compute_activation_stats_standard for standard InceptionV3 pool3 features.
    DEPRECATED: 使用展平像素值作为特征代理（非标准实现）。
    请改用 compute_activation_stats_standard 获取标准 InceptionV3 pool3 特征。
    """
    import warnings
    warnings.warn(
        "compute_activation_stats_deprecated is deprecated; "
        "use compute_activation_stats_standard for standard FID.",
        DeprecationWarning, stacklevel=2,
    )
    # Simplified: Use image statistics as feature proxy
    # 简化版：使用图像统计信息作为特征代理
    images_np = images.detach().cpu().numpy()
    if images_np.ndim == 4:
        # Flatten to feature vectors
        # 展平为特征向量
        features = images_np.reshape(images_np.shape[0], -1)
    else:
        features = images_np.reshape(1, -1)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid_approx_deprecated(mu1: np.ndarray, sigma1: np.ndarray,
                                   mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    DEPRECATED: Pixel-space approximate FID (non-standard, not comparable to literature).
    Use compute_fid_standard or compute_fid_standard_from_images instead.
    DEPRECATED: 像素空间近似版 FID（非标准实现，与文献不可比）。
    请改用 compute_fid_standard 或 compute_fid_standard_from_images。
    """
    import warnings
    warnings.warn(
        "compute_fid_approx_deprecated is deprecated; "
        "use compute_fid_standard or compute_fid_standard_from_images for standard FID.",
        DeprecationWarning, stacklevel=2,
    )
    diff = mu1 - mu2
    # Numerical stability handling
    # 数值稳定处理
    covmean = _sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(0, fid))


# Keep old function names as aliases for backward compatibility (with deprecation warning)
# 保留旧函数名作为别名，以兼容现有调用方（带弃用警告）
# compute_fid alias is re-assigned to the standard implementation at the end of the file (if torchvision is available)
# compute_fid 别名在文件末尾重新指向标准实现（若 torchvision 可用）
compute_activation_stats = compute_activation_stats_deprecated
compute_fid_approx = compute_fid_approx_deprecated


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Matrix square root (numerically stable version)

    Uses symmetrization + eigh (eigenvalue decomposition) to compute the matrix square root.
    First symmetrizes the matrix, then computes the square root via eigenvalue decomposition.
    矩阵平方根（数值稳定版）

    使用对称化 + eigh（特征值分解）方法求解矩阵平方根。
    先对矩阵进行对称化，然后通过特征值分解计算平方根。
    """
    # Ensure the input is symmetric (the product of two symmetric matrices is not necessarily symmetric)
    # 确保输入对称（两个对称矩阵的乘积不一定对称）
    if not np.allclose(matrix, matrix.T):
        matrix = (matrix + matrix.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvectors.dot(np.diag(np.sqrt(eigenvalues))).dot(eigenvectors.T)


# ---- Standard FID (InceptionV3 pool3 features, 2048-dimensional) ----
# ---- 标准 FID (InceptionV3 pool3 特征, 2048维) ----
try:
    import torchvision.models as tv_models
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

_INCEPTION_V3: Optional[torch.nn.Module] = None


def _get_inception_v3(device: str = "cpu") -> torch.nn.Module:
    """Lazily load and cache InceptionV3 (fc layer replaced with Identity, outputs 2048-dim pool3 features).
    惰性加载并缓存 InceptionV3（fc 层替换为 Identity，输出 2048 维 pool3 特征）。"""
    global _INCEPTION_V3
    if _INCEPTION_V3 is None:
        if not _HAS_TORCHVISION:
            raise RuntimeError("torchvision is required for standard FID")
        inception = tv_models.inception_v3(weights="DEFAULT", transform_input=False)
        inception.fc = torch.nn.Identity()
        inception.eval()
        for param in inception.parameters():
            param.requires_grad = False
        _INCEPTION_V3 = inception
    return _INCEPTION_V3.to(device)


def _preprocess_for_inception(images: torch.Tensor) -> torch.Tensor:
    """Preprocess for InceptionV3: resize to 299x299, normalize to [-1, 1].
    为 InceptionV3 预处理：resize 到 299x299，归一化到 [-1, 1]。"""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {images.shape[1]}")
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    images = images * 2.0 - 1.0
    return images


def compute_activation_stats_standard(images: torch.Tensor, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of images in feature space using InceptionV3 pool3 features (2048-dim).
    This is the foundation of the standard FID implementation.
    使用 InceptionV3 pool3 特征（2048 维）计算图像在特征空间的均值和协方差。
    这是标准 FID 的实现基础。
    """
    inception = _get_inception_v3(device)
    images = _preprocess_for_inception(images).to(device)

    with torch.no_grad():
        features = inception(images)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.cpu().numpy()

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid_standard(mu1: np.ndarray, sigma1: np.ndarray,
                          mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Standard Frechet Inception Distance computation (requires InceptionV3 features).
    标准 Frechet Inception Distance 计算（需配合 InceptionV3 特征使用）。"""
    diff = mu1 - mu2
    covmean = _sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(0, fid))


def compute_fid_standard_from_images(images1: torch.Tensor, images2: torch.Tensor,
                                      device: str = "cpu") -> float:
    """Standard FID: compute directly from two batches of images.
    标准 FID：直接从两批图像计算。"""
    mu1, sigma1 = compute_activation_stats_standard(images1, device)
    mu2, sigma2 = compute_activation_stats_standard(images2, device)
    return compute_fid_standard(mu1, sigma1, mu2, sigma2)


# ---- Public alias: compute_fid points to standard implementation (falls back to approximate version if torchvision unavailable) ----
# ---- 公开别名：compute_fid 指向标准实现（若 torchvision 不可用则降级到近似版） ----
if _HAS_TORCHVISION:
    compute_fid = compute_fid_standard_from_images
else:
    import warnings
    warnings.warn(
        "torchvision not available; compute_fid falls back to "
        "compute_fid_approx_deprecated (not comparable to literature).",
        RuntimeWarning, stacklevel=2,
    )
    compute_fid = compute_fid_approx_deprecated


# ---- IoU (Intersection over Union) ----
def compute_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                threshold: float = None, eps: float = 1e-8) -> float:
    """
    Compute IoU between predicted mask and ground-truth mask.

    threshold: Binarization threshold. If None, automatically uses max(mask)*0.3 (minimum 0.01),
               adapted for PSR-Net sparse masks (mean typically 0.001~0.01).
    计算预测掩膜与真实掩膜的 IoU。

    threshold: 二值化阈值。若为 None，自动使用 max(mask)*0.3（下限 0.01），
               适配 PSR-Net 稀疏掩膜（均值通常 0.001~0.01）。
    """
    if threshold is None:
        # Adaptive threshold: 30% of the mask maximum, but at least 0.01
        # 自适应阈值：掩膜最大值的 30%，但至少 0.01
        max_val = float(pred_mask.max().item())
        threshold = max(max_val * 0.3, 0.01)

    pred_binary = (pred_mask > threshold).float()
    gt_binary = (gt_mask > threshold).float()
    intersection = (pred_binary * gt_binary).sum().item()
    union = ((pred_binary + gt_binary) > 0).float().sum().item()
    if union == 0:
        # Both are empty
        # 两者都为空
        return 1.0
    return intersection / union


# ---- Mask contrast ratio ----
# ---- 掩膜对比度比率 ----
def compute_mask_contrast_ratio(mask: torch.Tensor, gt_mask: torch.Tensor,
                                 eps: float = 1e-8) -> float:
    """
    Mask contrast ratio = mean activation in defect region / mean activation in clean region

    When clean region activation is near zero (mask almost entirely black), contrast is meaningless and returns None.
    Callers should filter None values before aggregation, or substitute an appropriate default (e.g., 0).
    掩膜对比度比率 = 缺陷区域平均激活 / 完好区域平均激活

    当 clean 区域激活接近零时（掩膜几乎全黑），对比度无意义，返回 None。
    调用方应在聚合前过滤 None 值，或替换为合适的默认值（如 0）。
    """
    defect_region = mask * gt_mask
    clean_region = mask * (1 - gt_mask)

    defect_mean = defect_region.sum() / (gt_mask.sum() + eps)
    clean_mean = clean_region.sum() / ((1 - gt_mask).sum() + eps)

    if clean_mean < eps:
        # Mask too sparse, contrast meaningless
        # 掩膜过稀疏，对比度无意义
        return None
    ratio = float((defect_mean / clean_mean).item())
    return ratio


# ---- L1 improvement rate ----
# ---- L1 改进率 ----
def compute_l1_improvement(dirty: torch.Tensor, refined: torch.Tensor,
                            gt: torch.Tensor) -> float:
    """L1 improvement rate = (|dirty-gt| - |refined-gt|) / |dirty-gt| * 100%

    Returns 0.0 when dirty error is extremely small (near noise level) to avoid spurious large percentages.
    Threshold 1e-6 corresponds to a pixel value difference of approximately 0.0001% (in [0,1] range).
    L1改进率 = (|dirty-gt| - |refined-gt|) / |dirty-gt| * 100%

    当 dirty 误差极小（接近噪声水平）时返回 0.0，避免虚假的大百分比。
    阈值 1e-6 对应像素值差异约 0.0001%（在 [0,1] 范围内）。
    """
    dirty_err = F.l1_loss(dirty, gt).item()
    refined_err = F.l1_loss(refined, gt).item()
    if dirty_err < 1e-6:
        return 0.0
    return float((dirty_err - refined_err) / dirty_err * 100.0)


# ---- Pixel fidelity ----
# ---- 像素保真率 ----
def compute_pixel_fidelity(refined: torch.Tensor, original: torch.Tensor,
                            mask: torch.Tensor, threshold: float = 1e-6) -> float:
    """
    Pixel fidelity = proportion of unchanged pixel values in the M=0 region
    Theoretical value should be 100%
    像素保真率 = M=0区域中像素值未变化的比例
    理论值应为 100%
    """
    # Region where M=0
    # M=0 的区域
    unchanged_region = 1.0 - mask
    diff = torch.abs(refined - original)
    # In the unchanged region, proportion of pixels where difference < threshold
    # 在不变区域中，差异小于阈值的像素比例
    fidelity_map = (diff < threshold).float()
    fidelity = (fidelity_map * unchanged_region).sum() / (unchanged_region.sum() + 1e-8)
    return float(fidelity.item() * 100.0)


# ---- Inference performance ----
# ---- 推理性能 ----
def measure_inference_performance(model: torch.nn.Module,
                                   input_tensor: torch.Tensor,
                                   num_warmup: int = 10,
                                   num_runs: int = 100) -> Dict[str, float]:
    """Measure inference time and GPU memory
    测量推理时间和显存"""
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)

    # Timing
    # 计时
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
        avg_time_ms = np.mean(times)

        # GPU memory
        # 显存
        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                model(input_tensor)
                times.append((time.perf_counter() - t0) * 1000)
        avg_time_ms = np.mean(times)
        mem_allocated = 0.0
        mem_reserved = 0.0

    return {
        "inference_time_ms": avg_time_ms,
        "fps": 1000.0 / avg_time_ms if avg_time_ms > 0 else float("inf"),
        "gpu_memory_mb": mem_allocated,
        "gpu_memory_reserved_mb": mem_reserved,
    }


# ---- Numerical safety handling ----
# ---- 数值安全处理 ----
def sanitize_metrics(metrics_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Filter out inf/NaN metric values, replace with safe defaults.
    Use cases: pre-processing before aggregation, cleaning before JSON serialization.
    过滤 inf/NaN 指标值，替换为 safe defaults.
    适用场景: 聚合前的预处理、JSON 序列化前的清理。
    """
    clean = {}
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            if np.isnan(v):
                clean[k] = 0.0
            elif np.isinf(v):
                clean[k] = 0.0
            else:
                clean[k] = v
        elif isinstance(v, (np.floating, np.integer)):
            fv = float(v)
            if np.isnan(fv) or np.isinf(fv):
                clean[k] = 0.0
            else:
                clean[k] = fv
        else:
            clean[k] = v
    return clean


def sanitize_metric_array(values: List[float]) -> List[float]:
    """
    Remove None/inf/NaN values from the list.
    Used to filter out anomalous samples before aggregation.
    从列表中移除 None/inf/NaN 值。
    用于聚合前过滤异常样本。
    """
    return [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]


def safe_json_value(v):
    """
    JSON-safe serialization: convert inf/NaN to None.
    Note: This function cannot be used directly as the default callback for json.dumps,
    because json.dumps only calls default for non-serializable types, while float('inf')/float('nan')
    are float types (JSON-serializable but produce non-standard JSON).
    Correct usage: pre-process the entire data structure with safe_json_serialize() first, then json.dumps.
    JSON 安全序列化：将 inf/NaN 转为 None。
    注意: 此函数不能直接作为 json.dumps 的 default 回调使用，
    因为 json.dumps 只为不可序列化类型调用 default，而 float('inf')/float('nan')
    是 float 类型（json 可序列化但产生非标准 JSON）。
    正确用法: 先用 safe_json_serialize() 预处理整个数据结构，再 json.dumps。
    """
    if isinstance(v, (float, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
    return v


def safe_json_serialize(obj):
    """
    Recursively clean inf/NaN values in the object, making it safe for JSON serialization.
    Returns a cleaned copy of the object.
    递归清理对象中的 inf/NaN 值，使其可安全 JSON 序列化。
    返回清理后的对象副本。
    """
    if isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer, np.bool_)):
        return float(obj) if not isinstance(obj, np.bool_) else bool(obj)
    elif isinstance(obj, np.complexfloating):
        return None  # JSON does not support complex numbers / JSON 不支持复数
    return obj


# ---- Comprehensive evaluation ----
# ---- 综合评估 ----
def evaluate_all(refined_list: List[torch.Tensor],
                 gt_list: List[torch.Tensor],
                 dirty_list: Optional[List[torch.Tensor]] = None,
                 masks: Optional[List[torch.Tensor]] = None,
                 gt_masks: Optional[List[torch.Tensor]] = None,
                 metrics: List[str] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation function, supports all metrics

    Args:
        refined_list: List of refined images
        gt_list: List of ground-truth images
        dirty_list: List of degraded images (for L1 improvement rate)
        masks: List of predicted masks
        gt_masks: List of ground-truth masks (for IoU and contrast ratio)
        metrics: List of metrics to compute
    综合评估函数，支持所有指标

    Args:
        refined_list: 修复图像列表
        gt_list: 真实图像列表
        dirty_list: 退化图像列表（用于L1改进率）
        masks: 预测掩膜列表
        gt_masks: 真实掩膜列表（用于IoU和对比度比率）
        metrics: 要计算的指标列表
    """
    if metrics is None:
        metrics = ["psnr", "ssim", "l1"]

    results = defaultdict(list)

    for i, (refined, gt) in enumerate(zip(refined_list, gt_list)):
        if "psnr" in metrics:
            results["psnr"].append(compute_psnr(refined, gt))
        if "ssim" in metrics:
            results["ssim"].append(compute_ssim(refined, gt))
        if "l1" in metrics:
            results["l1_loss"].append(F.l1_loss(refined, gt).item())
        if "lpips" in metrics:
            results["lpips_approx"].append(compute_lpips_approx(refined, gt))

        if dirty_list and i < len(dirty_list):
            if "l1_improvement" in metrics:
                results["l1_improvement_pct"].append(
                    compute_l1_improvement(dirty_list[i], refined, gt))

        if masks and gt_masks and i < len(masks) and i < len(gt_masks):
            if "iou" in metrics:
                results["iou"].append(compute_iou(masks[i], gt_masks[i]))
            if "mask_contrast" in metrics:
                results["mask_contrast_ratio"].append(
                    compute_mask_contrast_ratio(masks[i], gt_masks[i]))
            if "mask_mean" in metrics:
                results["mask_mean"].append(masks[i].mean().item())

    # Aggregate (filter inf/NaN)
    # 汇总（过滤 inf/NaN）
    summary = {}
    for k, v in results.items():
        clean_v = sanitize_metric_array(v)
        if clean_v:
            summary[f"{k}_mean"] = float(np.mean(clean_v))
            summary[f"{k}_std"] = float(np.std(clean_v))

    return dict(summary)


def format_results_table(results: Dict[str, float], title: str = "Results") -> str:
    """Format results as a table string
    格式化结果为表格字符串"""
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
    for k, v in sorted(results.items()):
        if isinstance(v, float):
            lines.append(f"  {k:30s}: {v:.6f}")
    lines.append(f"{'='*60}\n")
    return "\n".join(lines)
