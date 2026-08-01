"""Histogram of GFP and tdTomato expression levels across all cells."""
import sys
sys.path.insert(0, "D:/thesis-research")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

SLICE_PATH = "D:/thesis-research/resources/cache/slice_1_adata.h5ad"
OUTPUT     = "D:/thesis-research/reporter_histograms.png"

print("Loading adata ...")
adata = ad.read_h5ad(SLICE_PATH)
print(f"  {adata.n_obs:,} cells")

panel = {g.lower(): g for g in adata.var_names}

def get_expr(gene: str) -> np.ndarray:
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not in panel. Available: {list(adata.var_names)[:10]} ...")
    x = adata[:, key].X
    return x.toarray().ravel() if issparse(x) else np.asarray(x).ravel()

gfp_expr = get_expr("GFP")
tdt_expr = get_expr("tdTomato")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Reporter gene expression — Slice 1 (raw counts)", fontsize=13)

GENES = [("GFP",      gfp_expr, "#2ECC71"),
         ("tdTomato", tdt_expr, "#E74C3C")]

for row, (name, expr, color) in enumerate(GENES):
    n_total = len(expr)
    n_zero  = int((expr == 0).sum())
    n_pos   = int((expr > 0).sum())

    # ── Left: full distribution (including zeros) ─────────────────────────
    ax = axes[row, 0]
    bins = np.arange(-0.5, expr.max() + 1.5, 1)
    ax.hist(expr, bins=bins, color=color, alpha=0.65, edgecolor="none")
    ax.set_title(f"{name} — all cells (n={n_total:,})\nzero: {n_zero:,}  |  expressing: {n_pos:,}")
    ax.set_xlabel("Raw count")
    ax.set_ylabel("Number of cells")

    # ── Right: expressing cells only (zoom in) ────────────────────────────
    ax = axes[row, 1]
    pos = expr[expr > 0]
    ax.hist(pos, bins=np.arange(0.5, pos.max() + 1.5, 1), color=color, alpha=0.65, edgecolor="none")
    ax.set_title(f"{name} — expressing cells only (n={n_pos:,})")
    ax.set_xlabel("Raw count")
    ax.set_ylabel("Number of cells")

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"Saved: {OUTPUT}")
