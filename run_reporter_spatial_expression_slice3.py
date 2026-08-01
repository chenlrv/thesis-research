"""Spatial plots colored by reporter expression using raw integer counts — Slice 3.
Four panels:
  1. GFP expression
  2. tdTomato — all expressing (>= 1 raw count)
  3. tdTomato AND GFP (both >= 1 raw count)
  4. tdTomato AND Cx3cr1 (both >= 1 raw count)
"""
import sys
sys.path.insert(0, "D:/thesis-research")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

RAW_PATH   = "D:/thesis-research/resources/cache/slice_3_adata.h5ad"
TUMOR_PATH = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"
OUTPUT     = "D:/thesis-research/reporter_spatial_expression_slice3.png"
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
cx3_expr = get_expr("Cx3cr1")

x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)

non_tumor = ~tumor_mask

# ── Masks ─────────────────────────────────────────────────────────────────
gfp_mask     = (gfp_expr >= 1) & non_tumor
tdt_mask     = (tdt_expr >= 1) & non_tumor
tdt_gfp_mask = (tdt_expr >= 1) & (gfp_expr >= 1) & non_tumor
tdt_cx3_mask = (tdt_expr >= 1) & (cx3_expr >= 1) & non_tumor

print(f"\nGFP >= 1 (non-tumor):              {gfp_mask.sum():,}")
print(f"tdTomato >= 1 (non-tumor):         {tdt_mask.sum():,}")
print(f"tdTomato >= 1 AND GFP >= 1:        {tdt_gfp_mask.sum():,}")
print(f"tdTomato >= 1 AND Cx3cr1 >= 1:     {tdt_cx3_mask.sum():,}")

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(36, 11), dpi=150)
fig.suptitle("Reporter gene spatial expression — Slice 3  (tumor cells shown as gray)", fontsize=13)

BG_COLOR    = "#E0E0E0"
TUMOR_COLOR = "#AAAAAA"

def draw(ax, expr, mask, cmap, title):
    ax.scatter(x[~mask], y[~mask],
               c=BG_COLOR, s=0.5, alpha=0.2, linewidths=0, rasterized=True)
    ax.scatter(x[tumor_mask], y[tumor_mask],
               c=TUMOR_COLOR, s=0.5, alpha=0.3, linewidths=0, rasterized=True)
    sc = ax.scatter(x[mask], y[mask],
                    c=expr[mask], cmap=cmap,
                    s=2.5, alpha=0.95, linewidths=0, rasterized=True,
                    vmin=expr[mask].min(), vmax=expr[mask].max())
    cb = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.02)
    cb.set_label("Expression level", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)

draw(axes[0], gfp_expr, gfp_mask,     "YlGn",
     f"GFP\nn = {gfp_mask.sum():,} (non-tumor)")

draw(axes[1], tdt_expr, tdt_mask,     "YlOrRd",
     f"tdTomato — all expressing\nn = {tdt_mask.sum():,} (non-tumor)")

draw(axes[2], tdt_expr, tdt_gfp_mask, "YlOrRd",
     f"tdTomato AND GFP\nn = {tdt_gfp_mask.sum():,} (non-tumor)")

draw(axes[3], tdt_expr, tdt_cx3_mask, "YlOrRd",
     f"tdTomato AND Cx3cr1\nn = {tdt_cx3_mask.sum():,} (non-tumor)")

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
