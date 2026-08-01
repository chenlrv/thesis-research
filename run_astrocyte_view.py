"""Clearer render of the raw-count astrocyte gate (reads astrocyte_cells_noise.csv).
Bigger markers + a zoom on the dense upper-left region so the structure is visible.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "D:/thesis-research/astrocyte_gate_slice3/astrocyte_cells_noise.csv"
OUT = "D:/thesis-research/astrocyte_gate_slice3/astrocyte_view.png"

df = pd.read_csv(CSV)
x = df["x"].to_numpy(); y = -df["y"].to_numpy()
astro = df["astrocyte"].to_numpy().astype(bool)
both = df["both"].to_numpy().astype(bool)
mid = astro & ~both

fig, axes = plt.subplots(2, 1, figsize=(15, 12), dpi=170,
                         gridspec_kw={"height_ratios": [1, 1]})

# full tissue
ax = axes[0]
ax.scatter(x, y, s=2, c="0.86", linewidths=0, rasterized=True)
ax.scatter(x[mid], y[mid], s=6, c="#41ab5d", linewidths=0, rasterized=True,
           label=f"astrocyte single-marker ({int(mid.sum())})")
ax.scatter(x[both], y[both], s=14, c="#00441b", linewidths=0, rasterized=True,
           label=f"high-conf both GFAP+Sparcl1 ({int(both.sum())})")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(markerscale=2, loc="lower right", fontsize=11)
ax.set_title(f"slice_3 astrocytes (raw-count scaled-noise gate)  "
             f"n={int(astro.sum())}, {100*astro.mean():.1f}%", fontsize=13)

# zoom: upper-left quadrant (the dense region)
ax = axes[1]
xlo, xhi = x.min(), np.quantile(x, 0.42)
ylo, yhi = np.quantile(y, 0.55), y.max()
sel = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
ax.scatter(x[sel], y[sel], s=5, c="0.86", linewidths=0, rasterized=True)
a = sel & mid; b = sel & both
ax.scatter(x[a], y[a], s=16, c="#41ab5d", linewidths=0, rasterized=True)
ax.scatter(x[b], y[b], s=34, c="#00441b", linewidths=0, rasterized=True)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("zoom: dense upper-left region", fontsize=13)

plt.tight_layout(); plt.savefig(OUT, dpi=170, bbox_inches="tight"); plt.close()
print("saved", OUT)
