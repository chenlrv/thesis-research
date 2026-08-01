import numpy as np
import scanpy as sc
import scipy.sparse as sp

ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
low = {g.lower(): g for g in ad.var_names}
obs = ad.obs

# locate reporters
print("== reporter availability ==")
for q in ["gfp", "tdtomato", "tdtom", "tomato"]:
    hits = [v for k, v in low.items() if q in k]
    if hits:
        print(f"  var match '{q}': {hits}")
print("  obs has Mean.G:", "Mean.G" in obs.columns, "| Max.G:", "Max.G" in obs.columns)
print("  obs has Mean.DAPI:", "Mean.DAPI" in obs.columns)

# pan-myeloid gate (exclude microglia-specific so MDM/BAM also gate in)
PANMY = ["Csf1r","C1qa","C1qb","C1qc","Aif1","Tyrobp","Itgam","Ptprc","Cd68","Trem2"]
cols = [ad.var_names.get_loc(g) for g in PANMY if g in ad.var_names]
sub = ad[:, cols].X
sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
myeloid = (sub >= 1).sum(1) >= 3
print(f"\nmyeloid compartment: {myeloid.sum():,} / {ad.n_obs:,} ({100*myeloid.mean():.1f}%) "
      f"[>=3 of {len(cols)} pan-myeloid markers]")
np.save("D:/thesis-research/_myeloid_mask_slice1.npy", myeloid)
