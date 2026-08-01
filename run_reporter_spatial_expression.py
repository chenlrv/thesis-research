"""Spatial plots colored by reporter co-expression using raw integer counts.
Panels show progressively stricter tdTomato+GFP thresholds to find the
combination that gives biologically expected MDM distribution (near tumor).
"""
import sys
sys.path.insert(0, "D:/thesis-research")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

RAW_PATH   = "D:/thesis-research/resources/cache/slice_1_adata.h5ad"
TUMOR_PATH = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUTPUT     = "D:/thesis-research/reporter_spatial_expression.png"
TUMOR_COL  = "pred_tumor_XGBoost"

print("Loading raw counts adata ...")
adata = ad.read_h5ad(RAW_PATH)
print(f"  {adata.n_obs:,} cells  |  dtype: {adata.X.dtype}")

print("Merging tumor predictions ...")
adata_tumor = ad.read_h5ad(TUMOR_PATH)
if TUMOR_COL not in adata_tumor.obs.columns:
    pred_cols = [c for c in adata_tumor.obs.columns if c.startswith("pred_tumor_")]
    TUMOR_COL = pred_cols[0]
adata.obs[TUMOR_COL] = adata_tumor.obs[TUMOR_COL].reindex(adata.obs_names).fillna(0).astype(int)
tumor_mask = adata.obs[TUMOR_COL].to_numpy() == 1
print(f"  Tumor cells: {tumor_mask.sum():,}")

panel = {g.lower(): g for g in adata.var_names}

def get_expr(gene):
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found in panel")
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)

gfp_expr = get_expr("GFP")
tdt_expr = get_expr("tdTomato")

x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)
non_tumor = ~tumor_mask

# ── Masks — progressively stricter co-expression thresholds ──────────────
masks = [
    ((tdt_expr >= 1) & (gfp_expr >= 1) & non_tumor, "tdT ≥ 1 AND GFP ≥ 1"),
    ((tdt_expr >= 2) & (gfp_expr >= 1) & non_tumor, "tdT ≥ 2 AND GFP ≥ 1"),
    ((tdt_expr >= 2) & (gfp_expr >= 2) & non_tumor, "tdT ≥ 2 AND GFP ≥ 2"),
    ((tdt_expr >= 3) & (gfp_expr >= 1) & non_tumor, "tdT ≥ 3 AND GFP ≥ 1"),
    ((tdt_expr >= 3) & (gfp_expr >= 2) & non_tumor, "tdT ≥ 3 AND GFP ≥ 2"),
]

print("\nCo-expression counts (non-tumor cells):")
for mask, label in masks:
    print(f"  {label:30s}  {mask.sum():,}")

# ── Figure — 2 rows: top row = tumor context, bottom = MDM candidates ────
fig, axes = plt.subplots(2, 3, figsize=(27, 18), dpi=150)
fig.suptitle("Reporter co-expression spatial — Slice 1 (raw counts, tumor excluded)", fontsize=13)

BG_COLOR = "#E0E0E0"
TUMOR_COLOR = "#E74C3C"

def draw(ax, mask, color, title):
    # all non-tumor cells as background
    bg = non_tumor & ~mask
    ax.scatter(x[bg], y[bg], c=BG_COLOR, s=0.4, alpha=0.15, linewidths=0, rasterized=True)
    # tumor cells for spatial reference
    ax.scatter(x[tumor_mask], y[tumor_mask],
               c=TUMOR_COLOR, s=0.8, alpha=0.4, linewidths=0, rasterized=True, label=f"Tumor n={tumor_mask.sum():,}")
    # highlighted cells
    ax.scatter(x[mask], y[mask],
               c="#2ECC71", s=3.0, alpha=0.9, linewidths=0, rasterized=True,
               label=f"MDM candidate n={mask.sum():,}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\nn={mask.sum():,}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.8, markerscale=3)

# top-left: tumor alone for reference
ax = axes[0, 0]
ax.scatter(x[non_tumor], y[non_tumor], c=BG_COLOR, s=0.4, alpha=0.15, linewidths=0, rasterized=True)
ax.scatter(x[tumor_mask], y[tumor_mask], c=TUMOR_COLOR, s=0.8, alpha=0.6, linewidths=0, rasterized=True)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title(f"Tumor cells (reference)\nn={tumor_mask.sum():,}", fontsize=10)

# remaining 5 panels
panel_axes = [axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2]]
for ax, (mask, label) in zip(panel_axes, masks):
    draw(ax, mask, "#2ECC71", label)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
