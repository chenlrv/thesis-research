"""Run myeloid annotation on slice 1 and save spatial plot."""
import sys, os
sys.path.insert(0, "D:/thesis-research")

import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from thesis_research.pipeline.cell_type_annotation.myeloid.myeloid_typing import (
    annotate_myeloid_cells,
    CLASS_NAMES,
)

SLICE_PATH     = "D:/thesis-research/resources/cache/slice_1_adata.h5ad"
ANNOTATED_PATH = "D:/thesis-research/resources/cache/slice_1_myeloid_annotated.h5ad"
OUTPUT_PLOT    = "D:/thesis-research/myeloid_slice1_annotation.png"

# ── Maximally distinct palette ─────────────────────────────────────────────
# Chosen so each type is a different hue and clearly separable
PALETTE = {
    "Other":                 "#AAAAAA",   # medium gray   — visible background
    "Astrocyte":             "#2ECC71",   # emerald green
    "MDM":                   "#E74C3C",   # red
    "BAM":                   "#8E44AD",   # violet
    "Microglia_Homeostatic": "#2980B9",   # ocean blue
    "Microglia_DAM":         "#F39C12",   # amber / orange-gold
}

# Draw order: Other first (background), then specific types on top
DRAW_ORDER = ["Other", "Astrocyte", "BAM", "Microglia_Homeostatic",
              "Microglia_DAM", "MDM"]

# ── Pipeline: run or load from cache ──────────────────────────────────────
if os.path.exists(ANNOTATED_PATH):
    print(f"Loading cached annotated adata from {ANNOTATED_PATH} ...")
    adata = ad.read_h5ad(ANNOTATED_PATH)
    print(f"  {adata.n_obs:,} cells")
else:
    print("Loading slice 1 ...")
    adata = ad.read_h5ad(SLICE_PATH)
    print(f"  {adata.n_obs:,} cells x {adata.n_vars} genes")

    print("Normalise + log1p + scale + PCA(50) ...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    print("Running myeloid annotation pipeline ...")
    adata = annotate_myeloid_cells(adata, gat_epochs=400)

    print(f"Saving annotated adata to {ANNOTATED_PATH} ...")
    adata.write_h5ad(ANNOTATED_PATH)

# ── Data ──────────────────────────────────────────────────────────────────
ct    = adata.obs["myeloid_cell_type"].to_numpy(dtype=str)
x     = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
y     = adata.obs["CenterY_global_px"].to_numpy(dtype=float)
probs = adata.obs["myeloid_cell_type_prob"].to_numpy(dtype=float)

counts       = {c: int((ct == c).sum()) for c in CLASS_NAMES + ["Other"]}
total        = adata.n_obs
myeloid_cts  = [c for c in CLASS_NAMES if c != "Other"]

# ── Figure: 3 panels ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 10), dpi=150)
gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1.2, 1.2], wspace=0.05)
ax1 = fig.add_subplot(gs[0])   # all cell types
ax2 = fig.add_subplot(gs[1])   # myeloid / astrocyte only (no Other)
ax3 = fig.add_subplot(gs[2])   # GAT confidence on myeloid cells

# ─── Panel 1: all cell types ─────────────────────────────────────────────
for cell_type in DRAW_ORDER:
    mask = ct == cell_type
    if not mask.any():
        continue
    is_other = cell_type == "Other"
    ax1.scatter(
        x[mask], y[mask],
        c=PALETTE[cell_type],
        s=1.2 if is_other else 2.0,
        alpha=0.45 if is_other else 0.9,
        linewidths=0,
        rasterized=True,
    )

ax1.set_title(
    f"All cells — slice 1\n"
    f"n = {total:,} total  |  {sum(counts[c] for c in myeloid_cts):,} myeloid/glial",
    fontsize=11,
)
ax1.set_aspect("equal")
ax1.axis("off")

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=7,
           color=PALETTE[c],
           label=f"{c}   n={counts.get(c,0):,}  ({counts.get(c,0)/total*100:.1f}%)")
    for c in DRAW_ORDER
]
ax1.legend(handles=legend_handles, loc="upper left", frameon=True,
           fontsize=7.5, edgecolor="0.7", framealpha=0.9,
           handlelength=1, borderpad=0.8)

# ─── Panel 2: myeloid / astrocyte only (Other = faint bg) ────────────────
other_mask = ct == "Other"
ax2.scatter(x[other_mask], y[other_mask],
            c="#E8E8E8", s=0.6, alpha=0.3, linewidths=0, rasterized=True)

for cell_type in ["Astrocyte", "BAM", "Microglia_Homeostatic",
                  "Microglia_DAM", "MDM"]:
    mask = ct == cell_type
    if not mask.any():
        continue
    ax2.scatter(
        x[mask], y[mask],
        c=PALETTE[cell_type],
        s=2.5,
        alpha=0.92,
        linewidths=0,
        rasterized=True,
    )

ax2.set_title("Myeloid / glial cells only\n(Other shown as faint background)", fontsize=11)
ax2.set_aspect("equal")
ax2.axis("off")

leg2_handles = [
    Line2D([0], [0], marker="o", linestyle="", markersize=7,
           color=PALETTE[c], label=f"{c}  n={counts.get(c,0):,}")
    for c in ["Astrocyte", "BAM", "Microglia_Homeostatic", "Microglia_DAM", "MDM"]
]
ax2.legend(handles=leg2_handles, loc="upper left", frameon=True,
           fontsize=7.5, edgecolor="0.7", framealpha=0.9)

# ─── Panel 3: GAT confidence (myeloid + astrocyte cells only) ────────────
non_other = ct != "Other"
ax3.scatter(x[other_mask], y[other_mask],
            c="#E8E8E8", s=0.6, alpha=0.3, linewidths=0, rasterized=True)
sc_obj = ax3.scatter(
    x[non_other], y[non_other],
    c=probs[non_other], cmap="plasma",
    s=2.5, alpha=0.92, linewidths=0, rasterized=True,
    vmin=0.4, vmax=1.0,
)
cb = plt.colorbar(sc_obj, ax=ax3, shrink=0.45, pad=0.02)
cb.set_label("GAT confidence", fontsize=9)
cb.ax.tick_params(labelsize=8)
ax3.set_title("GAT classification confidence\n(myeloid / glial cells only)", fontsize=11)
ax3.set_aspect("equal")
ax3.axis("off")

plt.savefig(OUTPUT_PLOT, bbox_inches="tight", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_PLOT}")

# ── Classification explanation ─────────────────────────────────────────────
print()
print("=" * 65)
print("HOW EACH CELL TYPE WAS CLASSIFIED")
print("=" * 65)

explanation = """
MDM  (Monocyte-Derived Macrophage)  n={MDM:,}
  Stage 1 — Reporter gating (ground truth):
    GFP high  AND  tdTomato high
    GFP marks all CX3CR1+ myeloid lineage; tdTomato marks cells
    that originated from CCR2+ monocytes in the blood.
    Co-expression = cell infiltrated from blood = MDM.
  Anchor cells: 214  ->  GAT expanded to {MDM:,}

BAM  (Border-Associated Macrophage)  n={BAM:,}
  Stage 1 — Reporter: GFP high, tdTomato low (brain-resident myeloid)
  Stage 2 — Marker score >= 0.4 for:
    Lyve1, Mrc1, Cd163, Clec12a, Clec7a, Marco, Itgax, Mertk
  Stage 3 — Vascular proximity >= 0.5  (close to Pecam1/Cdh5/Kdr/Tek/Vwf cells)
  BAMs live at perivascular, meningeal, and choroid plexus borders,
  so spatial closeness to endothelial cells is a strong prior.
  Anchor cells: 46  ->  GAT expanded to {BAM:,}

Microglia_Homeostatic  n={Microglia_Homeostatic:,}
  Stage 1 — Reporter: GFP high, tdTomato low (brain-resident myeloid)
  Stage 2 — Marker score >= 0.35 for:
    Cx3cr1, TMEM119, Aif1, C1qa, C1qb, C1qc, Tyrobp, Csf1r
  Stage 3 — NOT vascular (proximity < 0.2)  +  low BAM score (< 0.4)
  Resting, surveilling microglia distributed throughout parenchyma.
  Anchor cells: 110  ->  GAT expanded to {Microglia_Homeostatic:,}

Microglia_DAM  (Disease-Associated Microglia)  n={Microglia_DAM:,}
  Stage 1 — Reporter: GFP high, tdTomato low (brain-resident myeloid)
  Stage 2 — Marker score >= 0.35 for:
    Cst7, Gpnmb, Spp1, Apoe, Trem2, Lgals3, Cd9, Cd63
  Stage 3 — NOT vascular (proximity < 0.2)  +  low BAM score (< 0.4)
  Activated microglia responding to tumor/damage signals.
  Transcriptionally close to MDMs — the hardest distinction.
  Anchor cells: 149  ->  GAT expanded to {Microglia_DAM:,}

Astrocyte  n={Astrocyte:,}
  Stage 1 — Reporter: GFP low  (NOT myeloid lineage)
  Stage 2 — Marker score >= 0.4 for:
    GFAP, S100b, Sox9, Vim, Sparcl1, Glul
  Anchor cells: 11,935  ->  GAT refined to {Astrocyte:,}
  Note: includes reactive astrocytes which upregulate GFAP/Vim.

Other  n={Other:,}  ({other_pct:.1f}% of all cells)
  All cells that did not meet any anchor threshold AND were not
  in the GAT subset. Predominantly neurons, oligodendrocytes,
  OPCs, and endothelial cells.
  These are NOT classified by this pipeline — use CellTypist
  or SingleR on the full dataset for complete annotation.

GAT role (Stage 5):
  The GAT built a hybrid graph: spatial k-NN (k=6) + expression
  k-NN (k=10). High-confidence anchor cells (above) form the
  training labels. The GAT propagates labels to ambiguous myeloid
  cells using both their gene expression AND the identity of their
  spatial neighbours.
  Key example: an ambiguous GFP+ cell surrounded by BAM anchors
  near a vessel -> GAT predicts BAM even if its own marker score
  is borderline.
  Training accuracy on anchors: ~73% (expected — DAM/MDM boundary
  is genuinely ambiguous transcriptionally).
""".format(**counts, other_pct=counts.get("Other", 0) / total * 100)

print(explanation)
