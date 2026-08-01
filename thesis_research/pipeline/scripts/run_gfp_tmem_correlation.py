"""Scatter plot: GFP vs TMEM119 correlation — slices 3 and 4.

Only non-tumor cells expressing >= 1 count of either gene.
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
} for i in (3, 4)}
OUTPUT    = "D:/thesis-research/gfp_tmem_correlation_slices3-4.png"
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

    gfp  = get_expr(adata, "GFP")[non_tumor]
    tmem = get_expr(adata, "TMEM119")[non_tumor]

    keep = (gfp >= 1) | (tmem >= 1)
    return gfp[keep], tmem[keep]


def draw_panel(ax, gfp, tmem, slice_id):
    r, _ = pearsonr(gfp, tmem)

    coords, counts = np.unique(np.column_stack([gfp, tmem]), axis=0, return_counts=True)
    gx, ty = coords[:, 0], coords[:, 1]

    ax.scatter(gx, ty, s=counts / counts.max() * 1200,
               alpha=0.6, color="steelblue", linewidths=0.4, edgecolors="white")

    for xi, yi, cnt in zip(gx, ty, counts):
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
    ax.set_ylabel("TMEM119 (raw counts)")
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.set_xlim(-0.5, gfp.max() + 0.5)
    ax.set_ylim(-0.5, 15)

    print(f"  Slice {slice_id}: n={gfp.size:,}  r={r:.3f}")


fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
fig.suptitle("GFP vs TMEM119 (non-tumor cells expressing ≥ 1 count)", fontsize=13)
for ax, sid in zip(axes, (3, 4)):
    gfp, tmem = load_slice(SLICES[sid], sid)
    draw_panel(ax, gfp, tmem, sid)
plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"Saved: {OUTPUT}")
