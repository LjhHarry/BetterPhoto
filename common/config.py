"""
Unified configuration management -- shared by all PSR-Net experiments.

统一配置管理 —— PSR-Net 所有实验共用
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# --- Path config (auto-adapts for cloud GPU) ---
# ---- 路径配置 (云端 GPU 自动适配) ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP_ROOT = os.path.dirname(BASE_DIR)  # Experiments/

# RedrawingPhotoCreating exists only on local Windows; falls back to None on cloud
# RedrawingPhotoCreating 仅在本地 Windows 存在; 云端回退到 None
REDRAWING_RESOURCES = None
REDRAWING_DATASET = None
REDRAWING_OUTPUTS = None
_REDRAWING_DIR = os.path.join(_EXP_ROOT, "RedrawingPhotoCreating")
if os.path.isdir(_REDRAWING_DIR):
    REDRAWING_RESOURCES = os.path.join(_REDRAWING_DIR, "resourses")
    REDRAWING_DATASET = os.path.join(_REDRAWING_DIR, "dataset")
    REDRAWING_OUTPUTS = os.path.join(_REDRAWING_DIR, "outputs")


@dataclass
class ExperimentConfig:
    """Base config class for individual experiments.

    单个实验的配置基类"""
    name: str = "experiment"
    output_dir: str = ""
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    
    # Data
    # 数据
    image_size: int = 64
    batch_size: int = 16
    # v2.1: 200→1000, prevent overfitting on small samples
    train_samples: int = 1000       # v2.1: 200→1000, 防止小样本过拟合
    test_samples: int = 20

    # Model
    # 模型
    input_channels: int = 3
    base_channels: int = 64

    # Training
    # 训练
    epochs: int = 80
    lr: float = 1e-3
    # v2.1: 0.1→0.03, reduce sparsity penalty to prevent mask collapse
    lambda_sparse: float = 0.03    # v2.1: 0.1→0.03, 降低稀疏惩罚防止掩膜崩溃
    # v2.1: 40→60, extend warmup so model learns inpainting first
    warmup_epochs: int = 60         # v2.1: 40→60, 延长warmup让模型先学会修复
    seed: int = 42

    # Mask distillation (new in v2.1): BCE(mask, gt_mask) direct supervision, prevents M→0 collapse
    # 掩膜蒸馏 (v2.1 新增): BCE(mask, gt_mask) 直接监督，防止M→0崩溃
    # Not described in the paper; set to 0.0 to strictly match Eq.(2)
    # 论文未描述此扩展；设为0.0以严格匹配论文公式(2)
    lambda_distill: float = 0.0

    # Evaluation
    # 评价
    metrics: List[str] = field(default_factory=lambda: ["psnr", "ssim", "l1"])


@dataclass
class LambdaSweepConfig(ExperimentConfig):
    """λ_s continuous sweep config (A5)

    λ_s 连续扫描配置 (A5)"""
    lambda_values: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5])


@dataclass
class HighResConfig(ExperimentConfig):
    """High-resolution experiment config (A4)

    高分辨率实验配置 (A4)"""
    image_size: int = 256
    batch_size: int = 4
    train_samples: int = 10000
    epochs: int = 200
    warmup_epochs: int = 80


@dataclass
class TextureInjectionConfig(ExperimentConfig):
    """Texture injection experiment config (B1/B2)

    纹理注入实验配置 (B1/B2)"""
    image_size: int = 512
    batch_size: int = 2
    train_samples: int = 500
    epochs: int = 100
    warmup_epochs: int = 30
    lambda_sparse: float = 0.05
    base_channels: int = 64


@dataclass
class LocalEditingConfig(ExperimentConfig):
    """Local editing experiment config (B3)

    局部编辑实验配置 (B3)"""
    image_size: int = 256
    batch_size: int = 4
    train_samples: int = 200
    epochs: int = 100
    # Pixel fidelity threshold
    pixel_fidelity_threshold: float = 1e-6  # 像素保真率精度


@dataclass
class PhysicalErrorConfig(ExperimentConfig):
    """Physical error fixing experiment config (B4)

    物理错误修复实验配置 (B4)"""
    image_size: int = 256
    batch_size: int = 4
    train_samples: int = 200
    epochs: int = 100
    # Anomaly area ratio
    anomaly_area_ratio: float = 0.1  # 异常区域占比
    # IoU loss weight
    lambda_iou: float = 1.0          # IoU 损失权重


def get_config(experiment_id: str, **overrides) -> ExperimentConfig:
    """Get config by experiment ID

    根据实验编号获取配置"""
    config_map = {
        "A1": ExperimentConfig(name="A1_adversarial_degradation", epochs=80),
        "A2": ExperimentConfig(name="A2_iterative_redrawing", epochs=100),
        "A3": ExperimentConfig(name="A3_surgical_redrawing", epochs=100, image_size=128),
        "A4": HighResConfig(name="A4_high_resolution"),
        "A5": LambdaSweepConfig(name="A5_lambda_sweep", epochs=60),
        "A6": ExperimentConfig(name="A6_supplementary_metrics", epochs=80),
        "A7": ExperimentConfig(name="A7_method_comparison", epochs=100),
        "A8": ExperimentConfig(name="A8_multi_seed", epochs=80),
        "A9": ExperimentConfig(name="A9_cost_efficiency", epochs=80),
        "B1": TextureInjectionConfig(name="B1_style_refinement"),
        "B2": TextureInjectionConfig(name="B2_high_res_enhancement", image_size=256),
        "B3": LocalEditingConfig(name="B3_zero_trace_editing"),
        "B4": PhysicalErrorConfig(name="B4_physical_error_fix"),
    }
    config = config_map.get(experiment_id, ExperimentConfig())
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    return config
