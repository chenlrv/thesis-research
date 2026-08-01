"""Tumor cells expressing GFP > 0, plotted spatially — all 6 slices, 2 figures."""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

CACHE_DIR  = "/resources/cache"
TUMOR_DIR  = f"{CACHE_DIR}/with_tumor_prediction"
TUMOR_COL  = "pred_tumor_XGBoost"

BG       = "#D0D0D0"
TUMOR_BG = "#F4A460"
GFP_COL  = "#2ECC71"

SLIDES = {
    1: [1, 2, 3],
    2: [4, 5, 6],
}
OUTPUTS = {
    1: "D:/thesis-research/tumor_gfp_expression_slide1.png",
    2: "D:/thesis-research/tumor_gfp_expression_slide2.png",
}


def load_slice(slice_id):
    print(f"  Loading slice {slice_id} ...")
    adata = ad.read_h5ad(f"{CACHE_DIR}/slice_{slice_id}_adata.h5ad")
    adata_tumor = ad.read_h5ad(f"{TUMOR_DIR}/slice_{slice_id}_adata.h5ad")

    col = TUMOR_COL
    if col not in adata_tumor.obs.columns:
        col = next(c for c in adata_tumor.obs.columns if c.startswith("pred_tumor_"))
    adata.obs[col] = adata_tumor.obs[col].reindex(adata.obs_names).fillna(0).astype(int)

    tumor_mask = adata.obs[col].to_numpy() == 1

    panel = {g.lower(): g for g in adata.var_names}
    gfp_key = panel.get("gfp")
    if gfp_key is None:
        raise KeyError(f"'GFP' not found in panel for slice {slice_id}")

    gfp_all = adata[:, gfp_key].X
    gfp_all = (gfp_all.toarray().ravel() if issparse(gfp_all) else np.asarray(gfp_all).ravel()).astype(int)

    x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
    y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)

    return dict(
        x=x, y=y,
        tumor_mask=tumor_mask,
        gfp_pos=gfp_all[tumor_mask] > 0,
    )


def draw_slice(ax, data, slice_id):
    x, y = data["x"], data["y"]
    tumor_mask = data["tumor_mask"]
    gfp_pos    = data["gfp_pos"]

    non_tumor = ~tumor_mask
    x_tumor = x[tumor_mask]
    y_tumor = y[tumor_mask]

    n_tumor   = tumor_mask.sum()
    n_gfp_pos = gfp_pos.sum()
    n_gfp_neg = n_tumor - n_gfp_pos
    pct = 100 * n_gfp_pos / n_tumor if n_tumor else 0

    ax.scatter(x[non_tumor], y[non_tumor],
               c=BG, s=0.3, alpha=0.12, linewidths=0, rasterized=True)
    ax.scatter(x_tumor[~gfp_pos], y_tumor[~gfp_pos],
               c=TUMOR_BG, s=1.5, alpha=0.5, linewidths=0, rasterized=True,
               label=f"GFP=0  n={n_gfp_neg:,}")
    ax.scatter(x_tumor[gfp_pos], y_tumor[gfp_pos],
               c=GFP_COL, s=4.0, alpha=0.9, linewidths=0, rasterized=True,
               label=f"GFP>0  n={n_gfp_pos:,} ({pct:.1f}%)")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Slice {slice_id}", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85, markerscale=4)

    print(f"    Tumor {n_tumor:,}  |  GFP>0 {n_gfp_pos:,} ({pct:.1f}%)")


for slide_id, slices in SLIDES.items():
    print(f"\nSlide {slide_id}: slices {slices}")
    fig, axes = plt.subplots(1, 3, figsize=(27, 9), dpi=150)
    fig.suptitle(f"Tumor cells expressing GFP > 0 — Slide {slide_id} (slices {slices[0]}–{slices[-1]})",
                 fontsize=13)

    for ax, sid in zip(axes, slices):
        data = load_slice(sid)
        draw_slice(ax, data, sid)

    plt.tight_layout()
    out = OUTPUTS[slide_id]
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {out}")
