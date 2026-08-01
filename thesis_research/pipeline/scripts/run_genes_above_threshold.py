"""Per-slice panel QC: how many genes reach a given peak expression?

For each slice and each threshold T, count the number of genes whose
maximum per-cell raw count across all cells is >= T.

Thresholds: 1, 5, 10, 15, ..., up to the highest raw count observed
across the entire panel (rounded up to the next multiple of 5).
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

SLICES = {i: f"D:/thesis-research/resources/cache/slice_{i}_adata.h5ad"
          for i in range(1, 7)}
OUTPUT = "D:/thesis-research/genes_above_threshold_slices1-6.png"
STEP   = 5


def gene_max_counts(adata):
    """Per-gene max raw count across all cells (1-D int array, len = n_genes)."""
    X = adata.X
    if issparse(X):
        # sparse: max per column
        return np.asarray(X.max(axis=0).todense()).ravel().astype(int)
    return np.asarray(X).max(axis=0).astype(int)


def load_slice(path, slice_id):
    print(f"Loading Slice {slice_id} ...")
    adata = ad.read_h5ad(path)
    maxes = gene_max_counts(adata)
    print(f"  Slice {slice_id}: n_genes = {len(maxes):,}  panel max = {maxes.max()}")
    return maxes, adata.var_names.to_numpy()


per_slice = {sid: load_slice(p, sid) for sid, p in SLICES.items()}

global_max = max(m.max() for m, _ in per_slice.values())
top = int(np.ceil(global_max / STEP) * STEP)
# Fine-grained in the low-count region (1..5), step=5 thereafter
thresholds = np.array([1, 2, 3, 4] + list(range(STEP, top + 1, STEP)))

fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150, sharex=True, sharey=True)
fig.suptitle(
    f"Genes reaching peak raw count >= T  (all cells, per slice)\n"
    f"x-axis up to panel-wide max = {global_max}",
    fontsize=12,
)

for ax, (sid, (maxes, gene_names)) in zip(axes.ravel(), per_slice.items()):
    counts = np.array([(maxes >= t).sum() for t in thresholds])
    ax.plot(thresholds, counts, marker="o", color="steelblue", linewidth=1.6, markersize=4)
    ax.fill_between(thresholds, counts, alpha=0.15, color="steelblue")
    ax.set_title(f"Slice {sid}  (n_genes = {len(maxes):,}, max = {maxes.max()})",
                 fontsize=10)
    ax.set_xlabel("Threshold T (raw counts)")
    ax.set_ylabel("# genes with max(per-cell count) >= T")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_yscale("log")
    ax.set_xticks(thresholds)
    ax.tick_params(axis="x", labelsize=7)

    # annotate the top few genes by per-slice max
    top_idx = np.argsort(maxes)[::-1][:5]
    label = "Top genes (max counts):\n" + "\n".join(
        f"  {gene_names[i]}: {maxes[i]}" for i in top_idx
    )
    ax.text(0.97, 0.97, label, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85,
                      edgecolor="lightgray"))

    print(f"  Slice {sid}: " +
          "  ".join(f">={t}:{c}" for t, c in zip(thresholds[:8], counts[:8])) +
          ("  ..." if len(thresholds) > 8 else ""))

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
