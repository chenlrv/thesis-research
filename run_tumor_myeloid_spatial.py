"""Spatial plot: tumor cells (red), brain-resident myeloid (yellow), MDM (green)."""
import sys
sys.path.insert(0, "D:/thesis-research")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from thesis_research.pipeline.cell_type_annotation.myeloid.myeloid_typing import gate_by_reporters

SLICE_PATH  = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUTPUT_PLOT = "D:/thesis-research/tumor_myeloid_spatial.png"

print("Loading adata ...")
adata = ad.read_h5ad(SLICE_PATH)
print(f"  {adata.n_obs:,} cells x {adata.n_vars} genes")
print("  obs columns:", list(adata.obs.columns))

# ── Stage 1: reporter gating ───────────────────────────────────────────────
print("\nRunning reporter gating ...")
adata = gate_by_reporters(adata)

# ── Resolve final cell class (tumor takes priority) ────────────────────────
TUMOR_COL = "pred_tumor_XGBoost"
if TUMOR_COL not in adata.obs.columns:
    pred_cols = [c for c in adata.obs.columns if c.startswith("pred_tumor_")]
    if not pred_cols:
        raise KeyError(
            f"No tumor prediction column found. Available: {list(adata.obs.columns)}"
        )
    TUMOR_COL = pred_cols[0]
    print(f"'{TUMOR_COL}' not found — using '{TUMOR_COL}' instead")

tumor_mask = adata.obs[TUMOR_COL].astype(int) == 1
print(f"Using '{TUMOR_COL}' — {tumor_mask.sum():,} tumor cells")

reporter = adata.obs["myeloid_reporter_label"].to_numpy(dtype=str)

cell_class = np.where(
    tumor_mask,                           "Tumor",
    np.where(reporter == "MDM",           "MDM",
    np.where(reporter == "brain_resident_myeloid", "Brain_Myeloid",
                                          "Other"))
)
adata.obs["cell_class"] = cell_class

counts = {c: int((cell_class == c).sum()) for c in ["Tumor", "MDM", "Brain_Myeloid", "Other"]}
total  = adata.n_obs
print("\nCell class counts:")
for k, v in counts.items():
    print(f"  {k}: {v:,}  ({v/total*100:.1f}%)")

# ── Spatial plot ───────────────────────────────────────────────────────────
PALETTE = {
    "Other":         "#CCCCCC",
    "Brain_Myeloid": "#F1C40F",   # yellow
    "MDM":           "#2ECC71",   # green
    "Tumor":         "#E74C3C",   # red
}
DRAW_ORDER = ["Other", "Brain_Myeloid", "MDM", "Tumor"]

x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(14, 12), dpi=150)

for cls in DRAW_ORDER:
    mask = cell_class == cls
    if not mask.any():
        continue
    is_bg = cls == "Other"
    ax.scatter(
        x[mask], y[mask],
        c=PALETTE[cls],
        s=0.8 if is_bg else 2.0,
        alpha=0.3 if is_bg else 0.9,
        linewidths=0,
        rasterized=True,
    )

ax.set_aspect("equal")
ax.axis("off")
ax.set_title(
    f"Slice 1 — Tumor / Myeloid spatial map\n"
    f"n = {total:,} total cells",
    fontsize=12,
)

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=8,
           color=PALETTE[c],
           label=f"{c}   n={counts[c]:,}  ({counts[c]/total*100:.1f}%)")
    for c in DRAW_ORDER
]
ax.legend(handles=legend_handles, loc="upper left", frameon=True,
          fontsize=9, edgecolor="0.7", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, bbox_inches="tight", dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_PLOT}")
