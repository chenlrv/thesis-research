"""Inspect how cells/fovs are keyed in the slice-1 h5ads so we can join the tx
file's `cell` / (fov, cell_ID) to the tumor prediction and gene panel."""
import anndata as ad
import numpy as np

WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
WITH_TUMOR = r"d:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"

a = ad.read_h5ad(WITH_NEG)
print("=== with_neg obs columns ===")
print(list(a.obs.columns))
print("obs_names[:5]:", list(a.obs_names[:5]))
for c in ["fov", "cell_ID", "cell", "cell_global_id"]:
    if c in a.obs.columns:
        print(f"  {c} sample:", a.obs[c].iloc[:5].tolist())
print("n_vars:", a.n_vars)
# marker presence
markers = ["GFP","tdTomato","Cx3cr1","Lyve1","Mrc1","Cd163","TMEM119",
           "C1qa","C1qb","Hexb","Ccr2","Plac8","Vim","S100a6"]
present = [m for m in markers if m in a.var_names]
print("markers present in panel:", present)
missing = [m for m in markers if m not in a.var_names]
print("markers MISSING:", missing)

at = ad.read_h5ad(WITH_TUMOR)
print("\n=== with_tumor obs columns ===")
print(list(at.obs.columns))
print("obs_names[:5]:", list(at.obs_names[:5]))
for c in ["fov", "cell_ID", "cell", "cell_global_id", "pred_tumor_XGBoost"]:
    if c in at.obs.columns:
        print(f"  {c} sample:", at.obs[c].iloc[:5].tolist())
print("tumor frac:", float(at.obs["pred_tumor_XGBoost"].mean()))
