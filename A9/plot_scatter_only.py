"""
A9: Standalone plot quality_cost_scatter.png (no-label version)
A9: 单独绘制 quality_cost_scatter.png（无标签版）
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
    results = json.load(f)

# ── Plot ───────────────────────────────────────────────
# ── 绘图 ──────────────────────────────────────────────
methods_order = [
    "PSR-Net 64", "PSR-Net 128", "PSR-Net 256", "PSR-Net 512",
    "Full Regen 256", "Full Regen 512",
    "PSR-Net + Upscale 512",
]

colors = plt.cm.tab10(np.linspace(0, 1, len(methods_order)))

fig, ax = plt.subplots(figsize=(10, 7))

for method, color in zip(methods_order, colors):
    if method not in results:
        continue
    m = results[method]
    time_ms = m.get("inference_time_ms", 0)
    mem = m.get("gpu_memory_mb", 0)
    psnr = m.get("psnr", 0)
    size = max(mem * 2, 30)

    ax.scatter(time_ms, psnr, s=size, c=[color],
               alpha=0.7, edgecolors="black", linewidth=0.5)
    # No text labels added — leave for manual PPT annotation
    # 不添加任何文字标签 — 留给 PPT 手动标注

ax.set_xlabel("Inference Time (ms)", fontsize=12)
ax.set_ylabel("PSNR (dB)", fontsize=12)
ax.set_title("Quality-Cost Trade-off (size = GPU memory)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.margins(0.25)
plt.subplots_adjust(right=0.85)
plt.tight_layout()

save_path = os.path.join(out_dir, "quality_cost_scatter.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {save_path}")
