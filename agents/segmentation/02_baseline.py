"""
Step 2: BASELINE metrics on the CURRENT segmentation, restricted to the chosen
slice-1 FOVs, non-tumor cells. Uses the shared seg_metrics logic.

Run:
  conda run -n thesis_research python agents/segmentation/02_baseline.py
"""
import os
import json
import numpy as np
import scipy.sparse as sp
import anndata as ad

import sys
sys.path.insert(0, r"D:\thesis-research\agents\segmentation")
from seg_metrics import compute_all, fmt_report

OUT = r"D:\thesis-research\agents\outputs\segmentation"
WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
WITH_TUMOR = r"d:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"

with open(os.path.join(OUT, "chosen_fovs.txt")) as f:
    CHOSEN = [int(x) for x in f.read().strip().split(",")]
print("chosen FOVs:", CHOSEN)

a = ad.read_h5ad(WITH_NEG)
# attach tumor prediction by cell_global_id
at = ad.read_h5ad(WITH_TUMOR)
tmap = dict(zip(at.obs["cell_global_id"].astype(str), at.obs["pred_tumor_XGBoost"].to_numpy()))
a.obs["pred_tumor"] = a.obs["cell_global_id"].astype(str).map(tmap).fillna(False).astype(bool)
del at

fov = a.obs["fov"].astype(int).to_numpy()
in_fov = np.isin(fov, CHOSEN)
nontumor = ~a.obs["pred_tumor"].to_numpy()
sel = in_fov & nontumor
print(f"cells in chosen FOVs: {in_fov.sum():,}; non-tumor of those: {sel.sum():,}")

sub = a[sel].copy()
# Drop negative-probe / falsecode columns: keep only real genes for purity totals?
# For probe correlations we just index by name; total counts should reflect RNA.
# Identify negative probe columns by name prefix.
neg_mask = np.array([vn.lower().startswith(("negprb", "negprobe", "falsecode", "blank"))
                     for vn in sub.var_names])
print(f"negative/falsecode var columns: {int(neg_mask.sum())} of {sub.n_vars}")
genes_keep = sub.var_names[~neg_mask].tolist()
subg = sub[:, genes_keep]
X = subg.X
mat = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
mat = mat.astype(np.float64)
genes = list(genes_keep)

# whole-selection metrics
res_all = compute_all(mat, genes, label=f"BASELINE all chosen FOVs (non-tumor)")
print(fmt_report(res_all))

# per-FOV metrics + runtime note (baseline has negligible 'segmentation' runtime)
per_fov = {}
fov_sub = fov[sel]
for fv in CHOSEN:
    fm = fov_sub == fv
    if fm.sum() < 50:
        continue
    r = compute_all(mat[fm], genes, label=f"BASELINE FOV {fv}")
    per_fov[str(fv)] = r

out = {"all": res_all, "per_fov": per_fov, "chosen_fovs": CHOSEN}
with open(os.path.join(OUT, "baseline_metrics.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote baseline_metrics.json")
print(f"\nBaseline cell counts per FOV (non-tumor): "
      + ", ".join(f"{fv}:{int((fov_sub==fv).sum())}" for fv in CHOSEN))
