# PSR-Net: Pixel-level Selective Redrawing Network

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-research--incomplete-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

**Self-Supervised Sparse Regularization for Pixel-level Selective Image Redrawing**

> English | [中文](#中文)

---

## Overview

PSR-Net (Pixel-level Selective Redrawing Network) is a self-supervised image inpainting framework. Unlike traditional methods that require manually annotated defect masks, PSR-Net uses **sparse regularization** to automatically learn which pixels need modification. The core formula is:

```
I_refined = I_d + R ⊙ M
```

where:
- **I_d**: degraded/defective input image
- **R**: residual map (what to change)
- **M**: selection mask ∈ [0,1] (where to change)
- **⊙**: element-wise (Hadamard) product

The key innovation: a sparse regularization term `λ_s · mean(M)` acts as an "Occam's razor" — penalizing excessive modifications, forcing the model to modify **only the pixels that truly need fixing**, without any manual mask annotations.

---

## ⚠️ Current Status — IMPORTANT

**This project is incomplete and no longer under active development by the original author.**

A critical bug known as **Mask Collapse** was discovered: the sparse regularization `λ_s · mean(M)` can incentivize the mask M to collapse to zero, rendering the selective redrawing mechanism ineffective. The model exploits a shortcut — using extremely small M and extremely large R to satisfy the loss function while dodging the sparsity penalty.

### What's Been Tried
- Lowered `λ_s` from 0.1 → 0.03 (helps partially)
- Extended warmup epochs from 40 → 60
- Added mask distillation loss (`λ_distill` with BCE supervision)
- Increased training samples from 200 → 1000

These mitigations are documented in `common/config.py` and `common/training.py`, but the core issue remains unresolved.

### What Works
- The architecture (OverPaintNet, ~1M params) is functional
- The training pipeline (`common/training.py`) runs end-to-end
- Multiple experiment scaffolds exist (A1-A9, B1-B4)
- Synthetic data generation and evaluation metrics are implemented

---

## Future Directions

If you'd like to take over this project, there are two possible paths:

### Route A: Fix Pixel-Level Redrawing
Continue the current approach. Potential fixes to explore:
- **Entropy regularization**: Force M toward binary values (0 or 1) to prevent gray-zone exploitation
- **Tanh clamping on R**: Bound residual values to [-1, 1] to prevent scale ambiguity
- **Total Variation (TV) loss on M**: Encourage smooth, contiguous mask regions
- **Stronger mask supervision**: BCE(mask, gt_mask) with higher weight
- **Adversarial training**: Use a discriminator on mask quality

### Route B: Refactor with Diffusion Models (Recommended)
The original author believes **diffusion models are superior for image inpainting** since pixel-level approaches cannot understand semantic context. Potential approach:
- Use a lightweight network (like PSR-Net's encoder) as an **error detector**
- Use a small diffusion model (e.g., SD 1.5 Inpainting) as the **local redrawer**
- The two-stage pipeline: detect defects → selectively inpaint only 5-10% of pixels
- See experiments B1/B2 for partial diffusion integration

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd PSR-Net/Experiments

# Install core dependencies
pip install -r requirements.txt

# Optional: For diffusion model experiments (B1/B2)
pip install diffusers transformers accelerate
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 2.0, CUDA recommended.

---

## Quick Start

```bash
# Run the simplest experiment (A1 - block occlusion restoration)
python A1/run.py

# Run with custom settings
python A1/run.py --epochs 100 --batch_size 32 --image_size 128
```

---

## Project Structure

```
Experiments/
├── common/                    # Shared modules
│   ├── config.py              # Unified configuration (ExperimentConfig, etc.)
│   ├── training.py            # Training engine (TrainingEngine v2, CheckpointManager)
│   ├── model_factory.py       # Model definitions (OverPaintNet, OverPaintNetLarge)
│   ├── data_utils.py          # Data generation & datasets (synthetic, real, SD)
│   ├── evaluation.py          # Metrics (PSNR, SSIM, LPIPS, FID, IoU, etc.)
│   └── visualization.py       # Plotting utilities
├── A1/ ~ A9/                  # Method experiments (ablation, comparison, metrics)
├── B1/ ~ B4/                  # Application experiments (style, editing, error fix)
├── resources/                 # Cached images and prompts
├── PSR-Net论文.docx           # Original Chinese paper
├── PSR-Net Article.docx       # English translation of the paper
├── requirements.txt           # Python dependencies
├── prompts.txt                # SD image generation prompts
└── README.md                  # This file
```

---

## Experiments Overview

### A-Series: Method Validation

| ID | Experiment | Description |
|----|-----------|-------------|
| A1 | Multi-Degradation | Block, blur, JPEG, adversarial, historical checkpoint degradations |
| A2 | Iterative Redrawing | 2-3 rounds of selective refinement |
| A3 | Two-Stage Surgical Pipeline | Error localizer + local diffusion redrawer |
| A4 | High Resolution | 128/256/512 resolution, real datasets (Places2, CelebA-HQ) |
| A5 | λ_s Sweep | Continuous sweep 0.01-0.5, Pareto frontier analysis |
| A6 | Supplementary Metrics | SSIM, LPIPS, FID, IoU evaluation |
| A7 | Baseline Comparison | PSR-Net vs PartialConv, GatedConv, LaMa |
| A8 | Multi-Seed Statistics | 5 seeds, mean ± std reporting |
| A9 | Cost Efficiency | Inference time & memory measurement |

### B-Series: Application Scenarios

| ID | Experiment | Description |
|----|-----------|-------------|
| B1 | Style Refinement | Texture injection to de-AI-ify generated images |
| B2 | High-Res Enhancement | Selective high-frequency detail injection |
| B3 | Zero-Trace Local Editing | 100% pixel fidelity in non-edited regions |
| B4 | Physical Error Fix | Multi-finger correction via two-stage pipeline |

---

## Key Technical Details

### Model Architecture
- **OverPaintNet** (~1M params): 4-layer CNN encoder, 2× downsampling, for ≤128×128 resolution
- **OverPaintNetLarge** (~4M params): 7-layer encoder, 3× downsampling, for 256/512 resolution

### Loss Function
```
L = |I_refined - I_gt|₁ + λ_s · mean(M)
```

- L1 reconstruction loss for pixel-level accuracy
- Sparse regularization `λ_s · mean(M)` encourages minimal modifications
- Optional mask distillation: `λ_distill · BCE(M, M_gt)` for direct mask supervision

### Data Pipeline
- **Synthetic**: Random gradient backgrounds + block/blur/JPEG degradations
- **SD Image Dataset**: Uses Stable Diffusion v1.5 to generate training images
- **Real Image Dataset**: Loads real photos with synthetic degradations

---

## Known Issues

1. **Mask Collapse** (Critical): M → 0 at high resolutions or with insufficient λ_distill
2. **A3 Measurement Bug**: `full_regen_ms = 0.116ms` appears to be a measurement error
3. **A8 Training Loop**: Uses independent training loop, not `TrainingEngine` — needs synchronization
4. **High-Resolution Instability**: ≥128px masks completely collapse without `OverPaintNetLarge`
5. **Limited Degradation Types**: Currently tested mainly on block occlusion

---

## Citation

If you use this code or ideas in your research, please cite:

```bibtex
@misc{psrnet2026,
  title={PSR-Net: Pixel-level Selective Redrawing Network with Self-Supervised Sparse Regularization},
  author={Li, Jiahao},
  year={2026},
  howpublished={\url{<github-repo-url>}},
}
```

---

## Contributing

Contributions are welcome! Since this project is incomplete and the original author cannot continue development, anyone interested is encouraged to:

1. **Fork** the repository
2. **Choose a direction** (Route A or Route B above)
3. **Start with priority experiments** — see the rerun priorities in the paper
4. **Submit a PR** with improvements

Please feel free to open issues for questions or discussions.


---

# 中文

## PSR-Net：基于自监督稀疏正则化的像素级选择性重绘网络

## 概述

PSR-Net（像素级选择性重绘网络）是一个自监督图像修复框架。与传统方法需要人工标注缺陷掩膜不同，PSR-Net 使用**稀疏正则化**自动学习哪些像素需要修改。核心公式为：

```
I_refined = I_d + R ⊙ M
```

其中：
- **I_d**：退化/有缺陷的输入图像
- **R**：残差图（修改什么）
- **M**：选择掩膜 ∈ [0,1]（修改哪里）
- **⊙**：逐元素（Hadamard）乘法

核心创新：稀疏正则化项 `λ_s · mean(M)` 充当"奥卡姆剃刀"——惩罚过度修改，迫使模型**只修改真正需要修复的像素**，无需任何人工掩膜标注。

---

## ⚠️ 当前状态 — 重要

**此项目尚未完成，原作者已停止开发。**

发现了一个名为**掩膜崩溃（Mask Collapse）**的关键bug：稀疏正则化 `λ_s · mean(M)` 会激励掩膜 M 崩溃至零，使选择性重绘机制失效。模型走捷径——用极小 M 和极大 R 来满足损失函数同时规避稀疏惩罚。

### 已尝试的缓解措施
- 将 `λ_s` 从 0.1 降至 0.03（部分改善）
- 将 warmup 轮次从 40 延长至 60
- 添加掩膜蒸馏损失（`λ_distill` + BCE 监督）
- 将训练样本从 200 增加到 1000

这些措施记录在 `common/config.py` 和 `common/training.py` 中，但核心问题仍未解决。

### 哪些部分可用
- 网络架构（OverPaintNet，约100万参数）可正常运行
- 训练管线（`common/training.py`）可端到端运行
- 多个实验框架已就绪（A1-A9, B1-B4）
- 合成数据生成与评估指标已实现

---

## 未来方向

接手此项目可选择两个方向：

### 路线A：修复像素级重绘
继续当前路线。可探索的修复方案：
- **熵正则化**：迫使 M 趋向二值（0或1），防止灰区钻空子
- **Tanh 值域钳制**：将残差 R 限制在 [-1, 1]，防止尺度模糊
- **TV 损失**：对 M 施加 Total Variation 损失，鼓励平滑连续的掩膜区域
- **更强的掩膜监督**：提高 BCE(mask, gt_mask) 权重
- **对抗训练**：用判别器评估掩膜质量

### 路线B：使用扩散模型重构（推荐）
原作者认为**扩散模型在图像修复中更优**，因为像素级方法无法理解语义上下文。建议方案：
- 使用轻量网络（如 PSR-Net 的编码器）作为**错误检测器**
- 使用小型扩散模型（如 SD 1.5 Inpainting）作为**局部重绘器**
- 两阶段管线：检测缺陷 → 仅对 5-10% 像素进行选择性修复
- 参考实验 B1/B2 的部分扩散模型集成

---

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd PSR-Net/Experiments

# 安装核心依赖
pip install -r requirements.txt

# 可选：用于扩散模型实验（B1/B2）
pip install diffusers transformers accelerate
```

**环境要求**：Python ≥ 3.8, PyTorch ≥ 2.0, 推荐 CUDA。

---

## 快速开始

```bash
# 运行最简单的实验（A1 - 块遮挡修复）
python A1/run.py

# 自定义参数运行
python A1/run.py --epochs 100 --batch_size 32 --image_size 128
```

---

## 项目结构

```
Experiments/
├── common/                    # 共享模块
│   ├── config.py              # 统一配置（ExperimentConfig 等）
│   ├── training.py            # 训练引擎（TrainingEngine v2, CheckpointManager）
│   ├── model_factory.py       # 模型定义（OverPaintNet, OverPaintNetLarge）
│   ├── data_utils.py          # 数据生成与数据集（合成、真实、SD）
│   ├── evaluation.py          # 评估指标（PSNR, SSIM, LPIPS, FID, IoU 等）
│   └── visualization.py       # 可视化工具
├── A1/ ~ A9/                  # 方法实验（消融、对比、指标）
├── B1/ ~ B4/                  # 应用实验（风格、编辑、纠错）
├── resources/                 # 缓存图像和提示词
├── PSR-Net论文.docx           # 原始中文论文
├── PSR-Net Article.docx       # 英文翻译版论文
├── requirements.txt           # Python 依赖
├── prompts.txt                # SD 图像生成提示词
└── README.md                  # 本文件
```

---

## 实验概览

### A 系列：方法验证

| 编号 | 实验 | 说明 |
|------|------|------|
| A1 | 多退化类型 | 块遮挡、模糊、JPEG、对抗性、历史检查点退化 |
| A2 | 迭代式重绘 | 2-3 轮选择性精修 |
| A3 | 两阶段手术式管线 | 错误定位器 + 局部扩散重绘器 |
| A4 | 高分辨率 | 128/256/512 分辨率，真实数据集（Places2, CelebA-HQ） |
| A5 | λ_s 扫描 | 连续扫描 0.01-0.5，Pareto 前沿分析 |
| A6 | 补充指标 | SSIM、LPIPS、FID、IoU 评估 |
| A7 | 基线对比 | PSR-Net vs PartialConv、GatedConv、LaMa |
| A8 | 多种子统计 | 5 个种子，均值 ± 标准差报告 |
| A9 | 成本效率 | 推理时间与显存测量 |

### B 系列：应用场景

| 编号 | 实验 | 说明 |
|------|------|------|
| B1 | 风格精修 | 纹理注入以消除 AI 生成图像的"塑料感" |
| B2 | 高分辨率增强 | 选择性高频细节注入 |
| B3 | 零痕迹局部编辑 | 非编辑区域 100% 像素保真 |
| B4 | 物理错误修复 | 通过两阶段管线修正多指等错误 |

---

## 关键技术细节

### 模型架构
- **OverPaintNet**（约100万参数）：4 层 CNN 编码器，2 次下采样，适用于 ≤128×128 分辨率
- **OverPaintNetLarge**（约400万参数）：7 层编码器，3 次下采样，适用于 256/512 分辨率

### 损失函数
```
L = |I_refined - I_gt|₁ + λ_s · mean(M)
```

- L1 重建损失保证像素级精度
- 稀疏正则化 `λ_s · mean(M)` 鼓励最少修改
- 可选掩膜蒸馏：`λ_distill · BCE(M, M_gt)` 提供直接掩膜监督

### 数据管线
- **合成数据**：随机渐变背景 + 块/模糊/JPEG 退化
- **SD 图像数据集**：使用 Stable Diffusion v1.5 生成训练图像
- **真实图像数据集**：加载真实照片并施加合成退化

---

## 已知问题

1. **掩膜崩溃**（严重）：高分辨率或 λ_distill 不足时 M → 0
2. **A3 测量 Bug**：`full_regen_ms = 0.116ms` 疑似基准测量错误
3. **A8 训练循环**：使用独立训练循环，未使用 `TrainingEngine`，需同步
4. **高分辨率不稳定**：≥128px 掩膜在无 `OverPaintNetLarge` 时完全崩溃
5. **退化类型有限**：目前主要在块遮挡上测试

---

## 引用

如果在研究中使用本代码或思路，请引用：

```bibtex
@misc{psrnet2026,
  title={PSR-Net: Pixel-level Selective Redrawing Network with Self-Supervised Sparse Regularization},
  author={Li, Jiahao},
  year={2026},
  howpublished={\url{<github-repo-url>}},
}
```

---

## 贡献

欢迎贡献！由于原作者无法继续开发，鼓励任何人：

1. **Fork** 本仓库
2. **选择方向**（上述路线 A 或路线 B）
3. **从优先实验开始**——详见论文中的重跑优先级
4. **提交 PR** 提出改进

欢迎提交 Issue 进行提问或讨论。
