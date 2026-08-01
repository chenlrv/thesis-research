"""Re-draw the per-cluster spatial maps, each cluster in its OWN color
(non-cluster cells grey). Also a combined grid of all clusters. Reads the saved
cluster_assignments.csv (x, y, leiden) -- no re-clustering.

Usage: python run_recolor_cluster_spatial.py [slice_id]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "3"
OUT_DIR = f"D:/thesis-research/recluster_slice{SLICE_ID}"
ASSIGN_CSV = f"{OUT_DIR}/cluster_assignments.csv"
SPATIAL_DIR = f"{OUT_DIR}/per_cluster_spatial_colored"


def main():
    os.makedirs(SPATIAL_DIR, exist_ok=True)
    a = pd.read_csv(ASSIGN_CSV, index_col=0)
    x = a["x"].to_numpy()
    y = -a["y"].to_numpy()                       # flip Y to match slide
    lab = a["leiden"].astype(str).to_numpy()
    cats = sorted(set(lab), key=lambda s: int(s) if s.isdigit() else 1e9)
    n = len(cats)

    cmap = colormaps["tab10"].resampled(max(n, 1))
    color = {c: cmap(i) for i, c in enumerate(cats)}

    # ---- one PNG per cluster, each in its own color ----
    for c in cats:
        m = lab == c
        fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
        ax.scatter(x[~m], y[~m], s=1.2, c="lightgrey", linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=2.5, color=color[c], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice_{SLICE_ID} — cluster {c}  (n={int(m.sum())}, "
                     f"{100*m.mean():.1f}%)")
        plt.savefig(f"{SPATIAL_DIR}/cluster_{c}_spatial.png", dpi=180,
                    bbox_inches="tight")
        plt.close()

    # ---- combined grid: all clusters, each in its own color ----
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow), dpi=160)
    for ax, c in zip(axes.ravel(), cats):
        m = lab == c
        ax.scatter(x[~m], y[~m], s=0.6, c="lightgrey", linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=1.6, color=color[c], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cluster {c} (n={int(m.sum())}, {100*m.mean():.1f}%)", fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"slice_{SLICE_ID} non-tumor — clusters in space (each its own color)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/per_cluster_spatial_grid.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"saved {n} colored per-cluster maps -> {SPATIAL_DIR}")
    print(f"saved combined grid -> {OUT_DIR}/per_cluster_spatial_grid.png")


if __name__ == "__main__":
    main()
