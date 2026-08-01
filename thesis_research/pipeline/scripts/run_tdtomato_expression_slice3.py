"""tdTomato raw-count distribution in slice 3 (non-tumor cells).

Reports the % of non-tumor cells at each count level and plots a
log-scale histogram so the long tail is visible.
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

RAW_PATH   = "D:/thesis-research/resources/cache/slice_3_adata.h5ad"
TUMOR_PATH = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"
OUTPUT     = "D:/thesis-research/tdtomato_expression_slice3.png"
TUMOR_COL  = "pred_tumor_XGBoost"


def get_expr(adata, gene):
    panel = {g.lower(): g for g in adata.var_names}
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found in panel")
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)


print("Loading Slice 3 ...")
adata = ad.read_h5ad(RAW_PATH)
adata_tumor = ad.read_h5ad(TUMOR_PATH)
col = TUMOR_COL
if col not in adata_tumor.obs.columns:
    col = next(c for c in adata_tumor.obs.columns if c.startswith("pred_tumor_"))
adata.obs[col] = adata_tumor.obs[col].reindex(adata.obs_names).fillna(0).astype(int)
non_tumor = adata.obs[col].to_numpy() == 0

tdtom = get_expr(adata, "tdTomato")[non_tumor]
n = tdtom.size

print(f"\nSlice 3 — non-tumor cells: n = {n:,}")
print(f"  tdTomato max:    {tdtom.max()}")
print(f"  tdTomato mean:   {tdtom.mean():.3f}")
print(f"  tdTomato median: {np.median(tdtom):.1f}")
print(f"  tdTomato > 0:    {(tdtom > 0).sum():,}  ({100*(tdtom > 0).mean():.2f}%)")
print(f"  tdTomato >= 2:   {(tdtom >= 2).sum():,}  ({100*(tdtom >= 2).mean():.2f}%)")
print(f"  tdTomato >= 5:   {(tdtom >= 5).sum():,}  ({100*(tdtom >= 5).mean():.2f}%)")

print("\nPer-count breakdown:")
print(f"{'count':>6}  {'n_cells':>10}  {'% of non-tumor':>16}")
vals, counts_ = np.unique(tdtom, return_counts=True)
for v, c in zip(vals, counts_):
    print(f"{int(v):>6}  {c:>10,}  {100*c/n:>15.3f}%")

x_max = max(int(tdtom.max()), 10)
bins = np.arange(-0.5, x_max + 1.5, 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=150)
ax.hist(tdtom, bins=bins, color="#d6604d", alpha=0.85,
        histtype="stepfilled", edgecolor="black", linewidth=0.4)
ax.set_yscale("log")
ax.set_xlim(-0.5, x_max + 0.5)
ax.set_xlabel("tdTomato (raw counts)")
ax.set_ylabel("Number of cells (log scale)")
ax.set_title(
    f"tdTomato expression — Slice 3 non-tumor cells (n = {n:,})\n"
    f"max = {tdtom.max()}   |   mean = {tdtom.mean():.2f}   |   "
    f">0: {100*(tdtom > 0).mean():.1f}%",
    fontsize=10,
)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
