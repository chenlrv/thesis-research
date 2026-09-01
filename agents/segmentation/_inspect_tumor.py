import anndata as ad
import numpy as np
P = r"d:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
a = ad.read_h5ad(P)
print("shape:", a.shape)
tumor_cols = [c for c in a.obs.columns if "tumor" in c.lower() or "pred" in c.lower()]
print("tumor/pred cols:", tumor_cols)
for c in tumor_cols:
    print(f"  {c}: dtype={a.obs[c].dtype}, sample={a.obs[c].iloc[:3].tolist()}, ",
          f"vc={a.obs[c].value_counts(dropna=False).head().to_dict() if a.obs[c].dtype.name in ('category','object','bool') else 'num'}")
idcols = [c for c in a.obs.columns if "global_id" in c.lower() or c in ("cell_ID","cell","fov")]
print("id cols:", idcols)
print("obs_names sample:", list(a.obs_names[:3]))
