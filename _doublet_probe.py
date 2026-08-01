import numpy as np
import scanpy as sc

ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
X = ad.X
import scipy.sparse as sp
if sp.issparse(X):
    X = X.tocsc()
vn = list(ad.var_names)
idx = {g: i for i, g in enumerate(vn)}

def detected(genes, k):
    """boolean: cell detects >= k of these marker genes (>=1 raw count each)"""
    cols = [idx[g] for g in genes if g in idx]
    sub = ad[:, cols].X
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    return (sub >= 1).sum(1) >= k, cols

MICRO = ["Csf1r","C1qa","C1qb","C1qc","Cx3cr1","TMEM119","Aif1","Trem2","Tyrobp","Itgam","Ptprc","Cd68"]
ENDO  = ["Pecam1","Flt1","Vwf","Cdh5","Kdr"]
ASTRO = ["GFAP","S100b"]
NEUR  = ["Meg3"]

micro, _ = detected(MICRO, 4)   # >=4 of 12 myeloid markers
endo,  _ = detected(ENDO,  3)   # >=3 of 5 endothelial markers
astro, _ = detected(ASTRO, 2)   # both GFAP and S100b
neur,  _ = detected(NEUR,  1)   # Meg3 (single-gene, weak)
n = ad.n_obs

def report(name, a, b):
    pa, pb = a.mean(), b.mean()
    both = (a & b).mean()
    exp = pa * pb                       # null: independent
    enr = both / exp if exp > 0 else np.nan
    # of the rarer lineage, what frac is double-positive
    frac_of_a = (a & b).sum() / max(a.sum(), 1)
    frac_of_b = (a & b).sum() / max(b.sum(), 1)
    print(f"{name}")
    print(f"   {name.split(' x ')[0]:>10}+ = {pa*100:5.2f}%   {name.split(' x ')[1]:>10}+ = {pb*100:5.2f}%")
    print(f"   double+      = {both*100:5.2f}%   (chance/null = {exp*100:.2f}%  -> enrichment {enr:.1f}x)")
    print(f"   of {name.split(' x ')[0]}+ cells, {frac_of_a*100:.1f}% are also {name.split(' x ')[1]}+ ; "
          f"reverse {frac_of_b*100:.1f}%\n")

print(f"slice_1  n={n:,}\n")
report("Microglia x Endo", micro, endo)
report("Microglia x Astro", micro, astro)
report("Microglia x Neuron", micro, neur)
