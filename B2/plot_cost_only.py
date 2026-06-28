"""
B2: Standalone plot cost_analysis.png (no-label version)
B2: 单独绘制 cost_analysis.png（无标签版）
Only load data from results.json, no experiment runs
仅从 results.json 加载数据，不跑实验
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Load existing data
# 加载已有数据
with open(os.path.join(out_dir, "results.json")) as f:
    b2 = json.load(f)

cost_data = b2["cost_analysis"]

# ── Plot ───────────────────────────────────────────────
# ── 绘图 ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(cost_data.keys())
qualities = [cost_data[n].get("quality", 0) for n in names]
costs = [cost_data[n].get("cost", 0) for n in names]
mask_ratios = [cost_data[n].get("mask_ratio", 0) for n in names]

colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

# 左图: Quality vs Cost
for n, q, c, col in zip(names, qualities, costs, colors):
    axes[0].scatter(c, q, s=200, color=col, edgecolors="black", zorder=5)
    # No text labels added — leave for manual PPT annotation
    # 不添加文字标签 — 留给 PPT 手动标注

axes[0].set_xlabel("Relative Cost (lower is cheaper)", fontsize=11)
axes[0].set_ylabel("PSNR (dB)", fontsize=11)
axes[0].set_title("Quality vs Cost", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].margins(0.25)

# 右图: Quality vs Mask Activation Ratio
for n, q, r, col in zip(names, qualities, mask_ratios, colors):
    axes[1].scatter(r, q, s=200, color=col, edgecolors="black", zorder=5)
    # No text labels added — leave for manual PPT annotation
    # 不添加文字标签 — 留给 PPT 手动标注

axes[1].set_xlabel("Mask Activation Ratio (%)", fontsize=11)
axes[1].set_ylabel("PSNR (dB)", fontsize=11)
axes[1].set_title("Quality vs Selectivity", fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].margins(0.25)

plt.subplots_adjust(right=0.85)
plt.tight_layout()

save_path = os.path.join(out_dir, "cost_analysis.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {save_path}")
