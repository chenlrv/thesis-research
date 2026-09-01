import anndata as ad, numpy as np
BASE = "D:/thesis-research/resources/cache/"
a = ad.read_h5ad(BASE + "with_tumor_prediction/slice_1_adata.h5ad")
dx = ad.read_h5ad(BASE + "decontx/slice_1_decontx.h5ad")
print("a n_obs", a.n_obs, "dx n_obs", dx.n_obs)
print("index identical:", bool((a.obs.index == dx.obs.index).all()))
# cross-check tumor labels agree position-wise
at = a.obs["pred_tumor_XGBoost"].astype(bool).values
dt = dx.obs["pred_tumor_XGBoost"].astype(bool).values
print("tumor label agreement (positional):", float((at == dt).mean()))
# cross-check coords agree
ax = a.obs["CenterX_global_px"].values.astype(float)
dxx = dx.obs["CenterX_global_px"].values.astype(float)
print("CenterX max abs diff (positional):", float(np.max(np.abs(ax - dxx))))
