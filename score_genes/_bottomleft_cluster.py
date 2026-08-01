"""What are the concentrated bottom-left cells on slice 3 that light up as BOTH
BAM and Microglia? Locate the dense myeloid clump (DBSCAN on the myeloid pool),
then characterize it vs the rest of the tissue:
  - cell density, library size (segmentation/ambient artifact tells)
  - top genes enriched in the clump (what actually defines it)
  - key structural markers (choroid Ttr, vessel, blood, immune)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

SL = "3"
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT = "D:/thesis-research/score_genes_slice3_v2/codetect"
PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LIN = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
BAMg = ["Pf4", "Maf", "Mrc1", "Cd163", "Cd5l"]


def _dec(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def _X(h5):
    n = h5["X"]; e = str(n.attrs.get("encoding-type", "")); s = tuple(n.attrs["shape"])
    a = (n["data"][...], n["indices"][...], n["indptr"][...])
    return (csc_matrix(a, shape=s) if "csc" in e else csr_matrix(a, shape=s)).tocsr()


def _var(h5):
    v = h5["var"]; k = v.attrs.get("_index", "_index")
    return list(_dec(v[k.decode() if isinstance(k, bytes) else k][...]))


def _num(h5, c):
    nd = h5["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...]).astype(float)
        return ct[np.clip(cd, 0, None)]
    return nd[...].astype(float)


def _bool(h5, c):
    nd = h5["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...])
        return np.isin(np.where(cd >= 0, ct[np.clip(cd, 0, None)], "False").astype(str),
                       ["True", "1", "1.0", "TRUE", "true"])
    ar = nd[...]
    return (np.isin(_dec(ar).astype(str), ["True", "1"]) if ar.dtype.kind in ("S", "O")
            else ar.astype(bool))


with h5py.File(TMPL.format(SL), "r") as h5:
    X = _X(h5); var = np.array(_var(h5))
    cx = _num(h5, "CenterX_global_px"); cy = _num(h5, "CenterY_global_px")
    tum = _bool(h5, "pred_tumor_XGBoost")
Xk = X[~tum]; del X
cxk, cyk = cx[~tum], cy[~tum]
lib = np.asarray(Xk.sum(1)).ravel()
ngenes = np.diff(Xk.indptr)
vi = {g: list(var).index(g) for g in set(PAN + LIN + BAMg)}
det = lambda gs: np.vstack([np.asarray(Xk[:, vi[g]].todense()).ravel() > 0 for g in gs]).sum(0)
pool = (det(PAN) >= 3) & (det(LIN) >= 1)
bam = pool & (det(BAMg) >= 2)

# locate the dense BAM clump via a 2D-histogram peak (robust to coord scale)
bx, by = cxk[bam], cyk[bam]
print(f"coord range x[{cxk.min():.0f},{cxk.max():.0f}] y[{cyk.min():.0f},{cyk.max():.0f}]")
nb = 40
H, xe, ye = np.histogram2d(bx, by, bins=nb)
pi, pj = np.unravel_index(np.argmax(H), H.shape)
pcx, pcy = (xe[pi] + xe[pi + 1]) / 2, (ye[pj] + ye[pj + 1]) / 2
R = 1.5 * max((xe[1] - xe[0]), (ye[1] - ye[0]))   # ~1.5 bins around the peak
print(f"BAM density peak at ({pcx:.0f},{pcy:.0f}), {int(H.max())} BAM in peak bin, R={R:.0f}")
x0, x1 = pcx - R, pcx + R
y0, y1 = pcy - R, pcy + R
region = (cxk >= x0) & (cxk <= x1) & (cyk >= y0) & (cyk <= y1)
rest = ~region
area = (x1 - x0) * (y1 - y0)

print(f"clump bbox x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}]  cells={int(region.sum()):,}")
print(f"  cell DENSITY  region={region.sum()/area*1e6:.1f}  rest={rest.sum()/(np.ptp(cxk)*np.ptp(cyk))*1e6:.1f}  per 1e6 px^2")
print(f"  median library  region={np.median(lib[region]):.0f}  rest={np.median(lib[rest]):.0f}")
print(f"  median #genes   region={np.median(ngenes[region]):.0f}  rest={np.median(ngenes[rest]):.0f}")
print(f"  myeloid pool%   region={100*pool[region].mean():.0f}%  rest={100*pool[rest].mean():.0f}%")
print(f"  BAM co-det%     region={100*bam[region].mean():.0f}%  rest={100*bam[rest].mean():.0f}%")

# top genes enriched in the clump (normalized mean fold-change)
Xn = Xk.multiply(1.0 / np.maximum(lib, 1)[:, None]).tocsr() * 1e4
mi = np.asarray(Xn[region].mean(0)).ravel()
mo = np.asarray(Xn[rest].mean(0)).ravel()
lfc = np.log2((mi + 1) / (mo + 1))
order = np.argsort(-lfc)[:20]
print("\ntop 20 genes enriched in the clump (norm mean region vs rest):")
print(f"  {'gene':10} {'region':>8} {'rest':>8} {'log2FC':>7} {'det%reg':>7}")
for j in order:
    dpct = 100 * (np.asarray(Xk[region, j].todense()).ravel() > 0).mean()
    print(f"  {var[j]:10} {mi[j]:>8.2f} {mo[j]:>8.2f} {lfc[j]:>7.2f} {dpct:>6.0f}%")

fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
ax.scatter(cxk, cyk, s=0.5, c="#ddd", linewidths=0, rasterized=True)
ax.scatter(cxk[region], cyk[region], s=3, c="#d62728", linewidths=0, rasterized=True,
           label=f"clump ({int(region.sum())})")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.legend()
ax.set_title("slice 3 bottom-left clump (characterized)")
fig.savefig(f"{OUT}/bottomleft_clump.png", bbox_inches="tight"); plt.close(fig)
print(f"\nsaved -> {OUT}/bottomleft_clump.png")
