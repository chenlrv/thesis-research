"""Re-render the strict 6-slice map with per-slice label counts on each panel.
Reads the saved strict labels (no recompute). Overwrites method_a_all_slices_spatial_strict.png.
Run: conda run -n thesis_research python agents/annotation/plot_strict_with_counts.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "D:/thesis-research/agents/outputs/annotation/final"
df = pd.read_csv(f"{OUT}/method_a_all_slices_labels_strict.csv")

TUMOR = {1, 2, 5, 6}
focal = ["Microglia", "MDM", "BAM", "Astrocyte"]
order = ["Microglia", "MDM", "BAM", "Astrocyte", "Myeloid_unresolved", "unassigned"]
colors = {"Microglia": "#1f77b4", "MDM": "#d62728", "BAM": "#ff7f0e", "Astrocyte": "#2ca02c"}

fig, axes = plt.subplots(2, 3, figsize=(24, 14), dpi=130)
for i, s in enumerate(range(1, 7)):
    d = df[df["slice"] == s]
    n = len(d)
    lab = d["method_a_core"].to_numpy(str)
    x = d["CenterX_global_px"].to_numpy(float)
    y = d["CenterY_global_px"].to_numpy(float)
    ax = axes.ravel()[i]

    oth = ~np.isin(lab, focal)
    ax.scatter(x[oth], y[oth], s=0.6, c="#dddddd", alpha=0.45, linewidths=0)
    for cls in focal:
        m = lab == cls
        ax.scatter(x[m], y[m], s=2.5, c=colors[cls], alpha=0.85, linewidths=0, label=cls)

    vc = d["method_a_core"].value_counts()
    lines = [f"{'n cells':<19}{n:>7,}"]
    for cls in order:
        c = int(vc.get(cls, 0))
        lines.append(f"{cls:<19}{c:>7,} ({c/n*100:4.1f}%)")
    ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    ax.set_title(f"slice {s}  ({'TUMOR' if s in TUMOR else 'CONTROL'})", fontsize=12)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.legend(loc="lower right", fontsize=8, markerscale=3, framealpha=0.9)

plt.suptitle("Method A (strict MDM/BAM gate) — all 6 slices, with per-slice label counts", fontsize=15)
plt.tight_layout()
plt.savefig(f"{OUT}/method_a_all_slices_spatial_strict.png", bbox_inches="tight")
print("WROTE", f"{OUT}/method_a_all_slices_spatial_strict.png")
