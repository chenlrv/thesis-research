import anndata as ad
import numpy as np
P = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
a = ad.read_h5ad(P)
print("shape:", a.shape)
print("obs cols:", list(a.obs.columns))
print("var head:", list(a.var_names[:10]))
for g in ["GFP","tdTomato","Cx3cr1","Lyve1","Mrc1","Cd163","TMEM119","C1qa","Ccr2","Plac8","Ly6c2","Trem2"]:
    print(f"  {g}: {'present' if g in a.var_names else 'MISSING'}")
print("obsm:", list(a.obsm.keys()))
# fov column?
for c in a.obs.columns:
    if "fov" in c.lower() or "tumor" in c.lower() or "cell_id" in c.lower().replace("_",""):
        print("  obs candidate:", c, "| sample:", a.obs[c].iloc[:3].tolist())
print("X dtype:", a.X.dtype, "max:", a.X.max())
