"""
Model Factory -- Self-contained, zero external dependencies
Defines OverPaintNet and OverPaintNetLarge, no RedrawingPhotoCreating needed

模型工厂 —— 自包含，零外部依赖
定义了 OverPaintNet 和 OverPaintNetLarge，无需 RedrawingPhotoCreating
"""
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# OverPaintNet (Standard version ~1M parameters)
# OverPaintNet (标准版 ~1M 参数)
# ═══════════════════════════════════════════════════════════════════════════

class OverPaintNet(nn.Module):
    """
    Pixel-level selective repainting network (Standard version, ~1M parameters)

    Architecture: Encoder-Decoder
    - Encoder: 4 convolutional layers (with two stride=2 downsamplings)
      When base_channels=64, the channel progression is 64→64→128→256
      Total parameters ~1M (when base_channels=64)
    - Decoder: 256→128→64→32→4 (two transposed convolution upsamplings)
    - Output: first 3 channels = residual R, 4th channel sigmoid = mask M

    Refinement formula: I_refined = I_dirty + R * M

    Note: The paper describes a 3-layer encoder (32,64,128), but the code
    implementation uses 4 layers. The code structure takes precedence.

    像素级选择性重绘网络 (标准版, ~1M 参数)

    架构: 编码器-解码器
    - 编码器: 4 层卷积（含两次 stride=2 下采样）
      当 base_channels=64 时通道为 64→64→128→256
      总参数量约 1M（当 base_channels=64 时）
    - 解码器: 256→128→64→32→4（两次转置卷积上采样）
    - 输出: 前3通道 = 残差 R, 第4通道 sigmoid = 掩膜 M

    修复公式: I_refined = I_dirty + R * M

    注: 论文描述编码器为 3 层 (32,64,128)，但代码实现为 4 层。
    以代码结构为准。
    """

    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 4, 3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Defective image (B, 3, H, W), value range [0, 1]
            x: 缺陷图 (B, 3, H, W), 值域 [0, 1]
        Returns:
            residual: Residual map R (B, 3, H, W)
            mask: Mask M (B, 1, H, W), value range [0, 1]
            residual: 残差图 R (B, 3, H, W)
            mask: 掩膜 M (B, 1, H, W), 值域 [0, 1]
        """
        feat = self.enc(x)
        out = self.dec(feat)
        residual = out[:, :3, :, :]
        mask = torch.sigmoid(out[:, 3:4, :, :])
        return residual, mask

    def refine(self, dirty: torch.Tensor):
        """
        End-to-end refinement
        端到端修复
        Returns: (refined, residual, mask)
        """
        residual, mask = self.forward(dirty)
        refined = dirty + residual * mask
        return refined, residual, mask


# ═══════════════════════════════════════════════════════════════════════════
# OverPaintNetLarge (Large capacity version ~4M parameters, for 256/512 resolution)
# OverPaintNetLarge (大容量版 ~4M 参数, 用于 256/512 分辨率)
# ═══════════════════════════════════════════════════════════════════════════

class OverPaintNetLarge(nn.Module):
    """
    Large capacity version -- three downsamplings, larger receptive field

    Architecture: Encoder-Decoder (with three stride=2 downsamplings)
    - Encoder: 7 convolutional layers, when base_channels=64 the channels are:
      64→64→128→128→256→256→512
    - Decoder: Corresponding 7 upsampling/convolutional layers
    - Total parameters ~4M (when base_channels=64)
    - Suitable for 256/512 resolution

    Note: Compared to the standard OverPaintNet, one extra downsampling and
    intermediate layers are added, providing a larger receptive field but
    higher computational cost.

    大容量版本 —— 三次下采样, 更大感受野

    架构: 编码器-解码器（含三次 stride=2 下采样）
    - 编码器: 7 层卷积，当 base_channels=64 时通道为:
      64→64→128→128→256→256→512
    - 解码器: 对应 7 层上采样/卷积
    - 总参数量约 4M（当 base_channels=64 时）
    - 适用于 256/512 分辨率

    注: 相对于标准版 OverPaintNet，增加了一次下采样和中间层，
    感受野更大，但计算成本更高。
    """

    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 8, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 4, 3, padding=1),
        )

    def forward(self, x):
        feat = self.enc(x)
        out = self.dec(feat)
        residual = out[:, :3, :, :]
        mask = torch.sigmoid(out[:, 3:4, :, :])
        return residual, mask

    def refine(self, dirty):
        residual, mask = self.forward(dirty)
        refined = dirty + residual * mask
        return refined, residual, mask


# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters
    统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(model_type: str = "standard", base_channels: int = 64,
                 input_channels: int = 3, device: str = "cpu") -> nn.Module:
    """Create a model instance
    创建模型实例"""
    if model_type == "large":
        model = OverPaintNetLarge(input_channels=input_channels, base_channels=base_channels)
    else:
        model = OverPaintNet(input_channels=input_channels, base_channels=base_channels)
    return model.to(device)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load model weights (compatible with checkpoint and bare state_dict)
    加载模型权重 (兼容 checkpoint 和纯 state_dict)"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    return model


def save_checkpoint(model: nn.Module, optimizer, epoch: int,
                    history: dict, path: str):
    """Save a complete checkpoint
    保存完整检查点"""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, path)


def get_model_info(model: nn.Module) -> dict:
    """Get model information
    获取模型信息"""
    return {
        "total_params": count_parameters(model),
        "model_class": model.__class__.__name__,
    }
