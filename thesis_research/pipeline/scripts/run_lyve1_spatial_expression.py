"""Spatial plots of Lyve1 expression across all 6 slices.

Cells with Lyve1 >= 1 raw count are plotted and colored by expression level.
Non-expressing non-tumor cells shown as light gray background; tumor cells excluded.
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

SLICES = {i: {
    "raw":   f"D:/thesis-research/resources/cache/slice_{i}_adata.h5ad",
    "tumor": f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{i}_adata.h5ad",
} for i in range(1, 7)}
OUTPUT    = "D:/thesis-research/lyve1_spatial_expression.png"
TUMOR_COL = "pred_tumor_XGBoost"
GENE      = "Lyve1"


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
    tumor_mask = adata.obs[col].to_numpy() == 1
    expr = get_expr(adata, GENE)
    x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
    y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)
    return expr, tumor_mask, x, y


def draw_panel(ax, expr, tumor_mask, x, y, slice_id, vmax=None):
    non_tumor = ~tumor_mask
    lyve1_mask = (expr >= 1) & non_tumor
    bg_mask = non_tumor & ~lyve1_mask

    ax.scatter(x[bg_mask], y[bg_mask],
               c="#E0E0E0", s=0.4, alpha=0.15, linewidths=0, rasterized=True)

    sc = ax.scatter(x[lyve1_mask], y[lyve1_mask],
                    c=expr[lyve1_mask], cmap="YlOrRd",
                    s=2.5, alpha=0.95, linewidths=0, rasterized=True,
                    vmin=1, vmax=vmax if vmax is not None else (expr[lyve1_mask].max() if lyve1_mask.any() else 1))

    cb = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.02)
    cb.set_label("Raw counts", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_aspect("equal")
    ax.axis("off")
    total_non_tumor = non_tumor.sum()
    ax.set_title(f"Slice {slice_id}  —  {lyve1_mask.sum():,} / {total_non_tumor:,}", fontsize=10)

    print(f"  Slice {slice_id}: {lyve1_mask.sum():,} Lyve1+ non-tumor cells  "
          f"(max expr = {expr[lyve1_mask].max() if lyve1_mask.any() else 0})")


print("First pass: finding global expression range ...")
slices_data = {}
for slice_id in range(1, 7):
    expr, tumor_mask, x, y = load_slice(SLICES[slice_id], slice_id)
    slices_data[slice_id] = (expr, tumor_mask, x, y)

global_max = max(
    expr[(expr >= 1) & ~tumor_mask].max()
    for expr, tumor_mask, _, _ in slices_data.values()
    if ((expr >= 1) & ~tumor_mask).any()
)
print(f"Global Lyve1 max across all slices: {global_max}\n")

fig, axes = plt.subplots(2, 3, figsize=(27, 18), dpi=150)
fig.suptitle(f"{GENE} spatial expression (raw counts ≥ 1, non-tumor cells)", fontsize=14)

for ax, slice_id in zip(axes.ravel(), range(1, 7)):
    expr, tumor_mask, x, y = slices_data[slice_id]
    draw_panel(ax, expr, tumor_mask, x, y, slice_id, vmax=global_max)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
