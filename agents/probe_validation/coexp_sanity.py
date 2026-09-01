"""Sanity: unconditioned Pearson (all non-tumor cells, no positivity mask) for the
key pairs, to separate the real anti-correlation signal from union-mask conditioning bias."""
import numpy as np, anndata as ad, scipy.sparse as sp, pandas as pd

adx = ad.read_h5ad(r"d:/thesis-research/resources/cache/decontx/slice_1_decontx.h5ad")
tv = adx.obs["pred_tumor_XGBoost"]
nt = ~((tv.to_numpy() > 0.5) if np.issubdtype(tv.dtype, np.number) else tv.to_numpy().astype(bool))
sub = adx[nt]

def c(g):
    x = sub[:, g].X
    return np.asarray(x.todense()).ravel().astype(float) if sp.issparse(x) else np.asarray(x).ravel().astype(float)

def pear(x, y):
    xm = x - x.mean(); ym = y - y.mean()
    sx = np.sqrt((xm*xm).sum()); sy = np.sqrt((ym*ym).sum())
    return float((xm*ym).sum()/(sx*sy)) if sx and sy else float("nan")

genes = {k: c(k) for k in ["GFP","tdTomato","Cx3cr1","TMEM119","Csf1r","Lyve1","Mrc1","Cd163"]}
pairs = [("GFP","Cx3cr1"),("tdTomato","Cx3cr1"),("TMEM119","Cx3cr1"),
         ("GFP","tdTomato"),("Csf1r","Cx3cr1"),("TMEM119","Csf1r"),
         ("Mrc1","Cd163"),("Lyve1","Mrc1")]
print("UNCONDITIONED Pearson (ALL non-tumor cells, no mask):")
for a_, b_ in pairs:
    full = pear(genes[a_], genes[b_])
    # union-mask version for contrast
    m = (genes[a_] > 0) | (genes[b_] > 0)
    masked = pear(genes[a_][m], genes[b_][m])
    print(f"  {a_:9s}~{b_:9s}  full={full:+.3f}   union-mask={masked:+.3f}  (n_mask={int(m.sum())})")
