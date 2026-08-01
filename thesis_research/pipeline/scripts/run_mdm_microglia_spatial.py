"""Spatial plot of MDMs and microglia on slice 1.

Microglia = GFP+ TMEM119+
MDMs      = GFP+ TMEM119−
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

RAW_PATH   = "/resources/cache/slice_1_adata.h5ad"
TUMOR_PATH = "/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUTPUT     = "D:/thesis-research/mdm_microglia_spatial.png"
TUMOR_COL  = "pred_tumor_XGBoost"

print("Loading adata ...")
adata = ad.read_h5ad(RAW_PATH)

adata_tumor = ad.read_h5ad(TUMOR_PATH)
if TUMOR_COL not in adata_tumor.obs.columns:
    TUMOR_COL = next(c for c in adata_tumor.obs.columns if c.startswith("pred_tumor_"))
adata.obs[TUMOR_COL] = adata_tumor.obs[TUMOR_COL].reindex(adata.obs_names).fillna(0).astype(int)
tumor_mask = adata.obs[TUMOR_COL].to_numpy() == 1

panel = {g.lower(): g for g in adata.var_names}

def get_expr(gene):
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found in panel")
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)

gfp  = get_expr("GFP")
tmem = get_expr("TMEM119")

x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)

microglia = (gfp >= 1) & (tmem >= 1) & ~tumor_mask
mdm       = (gfp >= 1) & (tmem == 0) & ~tumor_mask
other     = ~microglia & ~mdm & ~tumor_mask

print(f"Microglia: {microglia.sum():,}")
print(f"MDMs:      {mdm.sum():,}")
print(f"Tumor:     {tumor_mask.sum():,}")

fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
fig.suptitle("MDMs vs Microglia — Slice 1\nMicroglia = GFP+ TMEM119+  |  MDMs = GFP+ TMEM119−", fontsize=12)

BG      = "#DDDDDD"
TUMOR   = "#E74C3C"
MG_COL  = "#4393c3"
MDM_COL = "#d6604d"

def base_scatter(ax):
    ax.scatter(x[other],      y[other],      c=BG,    s=0.3, alpha=0.15, linewidths=0, rasterized=True)
    ax.scatter(x[tumor_mask], y[tumor_mask], c=TUMOR, s=0.8, alpha=0.4,  linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")

# Panel 1: microglia only
ax = axes[0]
base_scatter(ax)
ax.scatter(x[microglia], y[microglia], c=MG_COL, s=3, alpha=0.9, linewidths=0, rasterized=True)
ax.set_title(f"Microglia — GFP+ TMEM119+\nn={microglia.sum():,}", fontsize=10)

# Panel 2: MDMs only
ax = axes[1]
base_scatter(ax)
ax.scatter(x[mdm], y[mdm], c=MDM_COL, s=3, alpha=0.9, linewidths=0, rasterized=True)
ax.set_title(f"MDMs — GFP+ TMEM119−\nn={mdm.sum():,}", fontsize=10)

# Panel 3: both together
ax = axes[2]
base_scatter(ax)
ax.scatter(x[microglia], y[microglia], c=MG_COL,  s=3, alpha=0.8, linewidths=0, rasterized=True, label=f"Microglia n={microglia.sum():,}")
ax.scatter(x[mdm],       y[mdm],       c=MDM_COL, s=3, alpha=0.8, linewidths=0, rasterized=True, label=f"MDMs n={mdm.sum():,}")
ax.set_title("Both populations", fontsize=10)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9, markerscale=4)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"Saved: {OUTPUT}")
