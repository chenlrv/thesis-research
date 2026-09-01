"""Inspect structure of slice_1 with_neg + decontx h5ads for probe-validation."""
import anndata as ad
import numpy as np
import scipy.sparse as sp

WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
DECONTX = r"d:/thesis-research/resources/cache/decontx/slice_1_decontx.h5ad"

for label, path in (("WITH_NEG", WITH_NEG), ("DECONTX", DECONTX)):
    print("=" * 70)
    print(label, path)
    print("=" * 70)
    a = ad.read_h5ad(path)
    print(f"shape: {a.shape}  X dtype={a.X.dtype}  sparse={sp.issparse(a.X)}")
    print(f"X min/max: {a.X.min()} / {a.X.max()}")
    print(f"obs columns: {list(a.obs.columns)}")
    print(f"layers: {list(a.layers.keys())}")
    print(f"obsm: {list(a.obsm.keys())}")
    # var names containing neg/control hints
    vn = list(a.var_names)
    print(f"n_var = {len(vn)}")
    hints = [v for v in vn if any(k.lower() in v.lower()
             for k in ("neg", "syscontrol", "systemcontrol", "control", "falsecode", "blank"))]
    print(f"control-like var_names ({len(hints)}): {hints[:60]}")
    # custom probes presence
    custom = ["Ccl2", "Cxcl13", "GFAP", "Lyve1", "TMEM119", "Trem2", "GFP", "tdTomato"]
    refs = ["Cx3cr1", "Pecam1", "Snap25", "Mrc1", "Cd163", "P2ry12", "Aif1", "Csf1r"]
    print("custom present:", {g: (g in a.var_names) for g in custom})
    print("ref present:", {g: (g in a.var_names) for g in refs})
    print()
