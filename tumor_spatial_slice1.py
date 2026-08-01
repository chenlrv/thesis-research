"""
Slice 1 (L321): cell-level spatial plot of the tumor population.
Non-tumor cells in light grey, tumor cells (pred_tumor_XGBoost) in red, drawn on top.
Orientation matches the pipeline spatial plots (CenterX/Y_global_px, no y-invert).
"""
import os
import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "fov_qc"
os.makedirs(OUTDIR, exist_ok=True)

a = ad.read_h5ad("resources/cache/with_tumor_prediction/slice_1_adata.h5ad")
o = a.obs
tumor = o["pred_tumor_XGBoost"].astype(bool).values
x = o["CenterX_global_px"].to_numpy(float)
y = o["CenterY_global_px"].to_numpy(float)
n_t = int(tumor.sum())

fig, ax = plt.subplots(figsize=(12, 11))
ax.scatter(x[~tumor], y[~tumor], s=1.2, c="#d9d9d9", linewidths=0, label=f"non-tumor ({(~tumor).sum():,})")
ax.scatter(x[tumor], y[tumor], s=3.5, c="#e6194B", linewidths=0, label=f"tumor ({n_t:,})")
ax.set_aspect("equal")
ax.set_xlabel("global X (px)"); ax.set_ylabel("global Y (px)")
ax.set_title(f"Slice 1 — tumor cell population (spatial)\n"
             f"{n_t:,} tumor / {a.n_obs:,} cells ({100*tumor.mean():.1f}%)")
ax.legend(markerscale=6, loc="upper right", framealpha=0.9)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/slice1_tumor_cells_spatial.png", dpi=200); plt.close()
print(f"wrote {OUTDIR}/slice1_tumor_cells_spatial.png  ({n_t} tumor cells)")
