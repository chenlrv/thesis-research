"""Scatter plot: GFP vs Cx3cr1 correlation — all 6 slices, two figures.

Only cells expressing at least one of the two genes (>= 1 raw count).
Non-tumor cells only.
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import pearsonr

SLICES = {i: {
    "raw":   f"D:/thesis-research/resources/cache/slice_{i}_adata.h5ad",
    "tumor": f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{i}_adata.h5ad",
} for i in range(1, 7)}
OUTPUT_1  = "D:/thesis-research/gfp_cx3cr1_correlation_slices1-3.png"
OUTPUT_2  = "D:/thesis-research/gfp_cx3cr1_correlation_slices4-6.png"
TUMOR_COL = "pred_tumor_XGBoost"


def get_expr(adata, gene):
    panel = {g.lower(): g for g in adata.var_names}
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found in panel")
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)


def load_slice(paths, slice_id):
    print(f"Loading Slice {slice_id} ...")
    adata = ad.read_h5ad(paths["raw"])
    adata_tumor = ad.read_h5ad(paths["tumor"])
    col = TUMOR_COL
    if col not in adata_tumor.obs.columns:
        col = next(c for c in adata_tumor.obs.columns if c.startswith("pred_tumor_"))
    adata.obs[col] = adata_tumor.obs[col].reindex(adata.obs_names).fillna(0).astype(int)
    non_tumor = adata.obs[col].to_numpy() == 0

    gfp = get_expr(adata, "GFP")[non_tumor]
    cx3 = get_expr(adata, "Cx3cr1")[non_tumor]

    keep = (gfp >= 1) | (cx3 >= 1)
    return gfp[keep], cx3[keep]


def draw_panel(ax, gfp, cx3, slice_id):
    r, _ = pearsonr(gfp, cx3)

    coords, counts = np.unique(np.column_stack([gfp, cx3]), axis=0, return_counts=True)
    gx, cy = coords[:, 0], coords[:, 1]

    ax.scatter(gx, cy, s=counts / counts.max() * 1200,
               alpha=0.6, color="steelblue", linewidths=0.4, edgecolors="white")

    for xi, yi, cnt in zip(gx, cy, counts):
        if yi <= 14.5:
            ax.text(xi, yi, f"{cnt:,}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="black")

    ax.text(0.97, 0.97, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    for ref in [10, 100, 1000]:
        ax.scatter([], [], s=ref / counts.max() * 1200,
                   color="steelblue", alpha=0.6, label=f"n={ref:,}")
    ax.legend(title="Cells per dot", fontsize=7, title_fontsize=7,
              loc="lower right", framealpha=0.8)

    ax.set_title(f"Slice {slice_id}  (n = {gfp.size:,})", fontsize=10)
    ax.set_xlabel("GFP (raw counts)")
    ax.set_ylabel("Cx3cr1 (raw counts)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xlim(-0.5, gfp.max() + 0.5)
    ax.set_ylim(-0.5, 15)


def plot_group(slice_ids, output):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.suptitle("GFP vs Cx3cr1 (non-tumor cells expressing ≥ 1 count)", fontsize=13)
    for ax, slice_id in zip(axes, slice_ids):
        gfp, cx3 = load_slice(SLICES[slice_id], slice_id)
        draw_panel(ax, gfp, cx3, slice_id)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {output}")


plot_group([1, 2, 3], OUTPUT_1)
plot_group([4, 5, 6], OUTPUT_2)
