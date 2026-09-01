"""
Shared prep for the annotation benchmark (slice 1).
- Loads with_tumor_prediction/slice_1 (raw int counts, all obs, spatial).
- Restricts to non-tumor (pred_tumor_XGBoost == False).
- Pulls 11 Negative* control probes from slice_1_adata_with_neg.h5ad (row-aligned by cell_global_id)
  -> per-cell mean negprobe = RNA-detection background (NOT the 184 SystemControl).
- Adds decontx_contamination from decontx file (row-aligned).
- Computes log1p layer + X_pca (needed by the GAT pipeline) on the non-tumor subset.
- Writes resources/cache/annotation_bench/slice1_nontumor.h5ad  (raw counts in .X,
  layers['log1p'], obsm['X_pca'], obs['neg_mean'], obs['decontx_contamination']).
"""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

OUT_DIR = "D:/thesis-research/resources/cache/annotation_bench"
os.makedirs(OUT_DIR, exist_ok=True)
BASE = "D:/thesis-research/resources/cache/"

print("Loading with_tumor_prediction/slice_1 ...")
a = ad.read_h5ad(BASE + "with_tumor_prediction/slice_1_adata.h5ad")
print("  full:", a.shape)

# --- pull negatives (row-aligned by cell_global_id) ---
print("Loading with_neg for Negative* probes ...")
wn = ad.read_h5ad(BASE + "slice_1_adata_with_neg.h5ad")
neg_genes = [g for g in wn.var_names if g.lower().startswith("negative")]
print("  Negative probes:", len(neg_genes), neg_genes)
# align by cell_global_id
assert "cell_global_id" in a.obs and "cell_global_id" in wn.obs
neg_X = wn[:, neg_genes].X
neg_X = neg_X.toarray() if sp.issparse(neg_X) else np.asarray(neg_X)
neg_df = pd.DataFrame({"neg_total": neg_X.sum(1), "neg_mean": neg_X.mean(1)},
                      index=wn.obs["cell_global_id"].values)
gid = a.obs["cell_global_id"].values
a.obs["neg_total"] = neg_df["neg_total"].reindex(gid).values
a.obs["neg_mean"] = neg_df["neg_mean"].reindex(gid).values
print("  neg_mean median/mean:", float(np.median(a.obs["neg_mean"])), float(np.mean(a.obs["neg_mean"])))
n_neg = len(neg_genes)
a.uns["n_neg_probes"] = n_neg

# --- decontx contamination ---
print("Loading decontx contamination ...")
dx = ad.read_h5ad(BASE + "decontx/slice_1_decontx.h5ad")
# decontx file is row-aligned (same n_obs, same order presumably) - match on index
if dx.n_obs == a.n_obs and (dx.obs.index == a.obs.index).all():
    a.obs["decontx_contamination"] = dx.obs["decontx_contamination"].values
    print("  matched decontx by obs index (identical order)")
else:
    # fall back: align on cell_global_id if present, else position
    print("  WARNING decontx index mismatch; aligning by position")
    a.obs["decontx_contamination"] = dx.obs["decontx_contamination"].values

# --- restrict to non-tumor ---
nt = a[~a.obs["pred_tumor_XGBoost"].astype(bool)].copy()
print("non-tumor:", nt.shape)

# keep raw counts as a layer
nt.layers["counts"] = nt.X.copy()

# --- standard processing for PCA (the GAT needs X_pca) ---
sc.pp.normalize_total(nt, target_sum=1e4)
sc.pp.log1p(nt)
nt.layers["log1p"] = nt.X.copy()
sc.pp.scale(nt, max_value=10)
sc.tl.pca(nt, n_comps=50, svd_solver="arpack")
# restore raw counts to .X (gating + InSituType need raw counts)
nt.X = nt.layers["counts"].copy()

outpath = OUT_DIR + "/slice1_nontumor.h5ad"
nt.write_h5ad(outpath)
print("WROTE", outpath, nt.shape)
print("obsm:", list(nt.obsm.keys()), "layers:", list(nt.layers.keys()))
print("n_neg_probes:", nt.uns.get("n_neg_probes"))
