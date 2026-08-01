"""Check raw tdTomato expression corrected for per-cell background using negative probes.

Reads QC-filtered cells and negative probe counts from slice_1_adata_with_neg.h5ad.
"""
import sys
sys.path.insert(0, "D:/thesis-research")

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import issparse

SLICE_ADATA = "D:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
OUTPUT      = "D:/thesis-research/raw_reporter_check.png"

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading slice 1 adata (with neg probes) ...")
adata = ad.read_h5ad(SLICE_ADATA)
print(f"  {adata.n_obs:,} cells  |  {adata.n_vars} genes  |  dtype: {adata.X.dtype}")
print(f"  Genes (first 20): {list(adata.var_names[:20])}")

# ── Identify columns ──────────────────────────────────────────────────────
panel    = {g.lower(): g for g in adata.var_names}
neg_vars = [g for g in adata.var_names if g.lower().startswith("neg")]
print(f"\nNegative probe columns ({len(neg_vars)}): {neg_vars}")

if not neg_vars:
    raise RuntimeError("No negative probe genes found (expected prefix 'Neg')")

def get_expr(gene):
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found. Available: {list(adata.var_names)[:20]}")
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)

tdt_expr = get_expr("tdTomato")
gfp_expr = get_expr("GFP")

# ── Per-cell background from negative probes ─────────────────────────────
neg_X = adata[:, neg_vars].X
neg_counts = neg_X.toarray() if issparse(neg_X) else np.asarray(neg_X)
per_cell_bg = neg_counts.astype(float).mean(axis=1)

print(f"\nPer-cell background (mean of {len(neg_vars)} neg probes):")
print(f"  mean:   {per_cell_bg.mean():.3f}")
print(f"  median: {np.median(per_cell_bg):.3f}")
print(f"  max:    {per_cell_bg.max():.3f}")

# ── Compare filtering strategies ─────────────────────────────────────────
tdt_above_bg = tdt_expr > per_cell_bg

print(f"\ntdTomato filtering strategies:")
print(f"  >= 1 raw count (uncorrected):        {(tdt_expr >= 1).sum():,}")
print(f"  >= 2 raw counts (uncorrected):       {(tdt_expr >= 2).sum():,}")
print(f"  >= 3 raw counts (uncorrected):       {(tdt_expr >= 3).sum():,}")
print(f"  > per-cell background (corrected):   {tdt_above_bg.sum():,}")
print(f"  > background AND GFP >= 1:           {(tdt_above_bg & (gfp_expr >= 1)).sum():,}")

print(f"\nCo-expression strategies (tdT AND GFP):")
print(f"  tdT >= 1 AND GFP >= 1:               {((tdt_expr >= 1) & (gfp_expr >= 1)).sum():,}")
print(f"  tdT >= 2 AND GFP >= 1:               {((tdt_expr >= 2) & (gfp_expr >= 1)).sum():,}")
print(f"  tdT >= 2 AND GFP >= 2:               {((tdt_expr >= 2) & (gfp_expr >= 2)).sum():,}")
print(f"  tdT >= 3 AND GFP >= 1:               {((tdt_expr >= 3) & (gfp_expr >= 1)).sum():,}")
print(f"  tdT >= 3 AND GFP >= 2:               {((tdt_expr >= 3) & (gfp_expr >= 2)).sum():,}")

# ── Plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("tdTomato raw counts vs per-cell background — Slice 1 (QC-filtered)", fontsize=12)

ax = axes[0]
pos = tdt_expr[tdt_expr > 0]
ax.hist(pos, bins=np.arange(0.5, pos.max() + 1.5, 1), color="#E74C3C", alpha=0.7)
ax.set_title(f"tdTomato raw counts\n(expressing cells, n={len(pos):,})")
ax.set_xlabel("Raw count")
ax.set_ylabel("Number of cells")

ax = axes[1]
ax.hist(per_cell_bg, bins=50, color="#3498DB", alpha=0.7)
ax.axvline(np.median(per_cell_bg), color="black", linestyle="--",
           label=f"median = {np.median(per_cell_bg):.3f}")
ax.set_title(f"Per-cell background\n(mean of {len(neg_vars)} negative probes)")
ax.set_xlabel("Mean negative probe count")
ax.set_ylabel("Number of cells")
ax.legend(fontsize=9)

ax = axes[2]
expressing = tdt_expr > 0
ax.scatter(per_cell_bg[expressing], tdt_expr[expressing],
           c="#E74C3C", s=1.5, alpha=0.3, rasterized=True)
lim = max(per_cell_bg.max(), tdt_expr.max())
ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="tdt = background")
ax.set_xlabel("Per-cell background (mean neg probe count)")
ax.set_ylabel("tdTomato raw count")
ax.set_title("Points above diagonal = signal > noise")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
