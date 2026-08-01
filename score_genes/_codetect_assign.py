"""Full hierarchical CO-DETECTION assignment (high-confidence, no classifier)
on the clean inclusive myeloid pool. Reports counts for all 3 subtypes so we can
see whether positive co-detection alone yields a few-k each + stays low in control.

BAM  = >=2 of {Pf4,Maf,Mrc1,Cd163,Cd5l}
MDM  = Ccr2 AND >=1 of {Plac8,Vcan,Cd14,Ccr1,Fpr1}, and NOT Pf4/Maf+   (specific)
Micro= >=k of {Adgrg1,Hpgds,Gpr183} and not BAM/MDM   (report k=1 and k=2)
order: BAM > MDM > Microglia > unresolved
"""
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
MICRO = ["Adgrg1", "Hpgds", "Gpr183"]
BAM = ["Pf4", "Maf", "Mrc1", "Cd163", "Cd5l"]
BAM_STABLE = ["Pf4", "Maf"]
MDM_SUP = ["Plac8", "Vcan", "Cd14", "Ccr1", "Fpr1"]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ALLG = set(PAN + LINEAGE + MICRO + BAM + MDM_SUP + ["Ccr2"])


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


def _bool(h5, c):
    nd = h5["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...])
        return np.isin(np.where(cd >= 0, ct[np.clip(cd, 0, None)], "False").astype(str),
                       ["True", "1", "1.0", "TRUE", "true"])
    ar = nd[...]
    return (np.isin(_dec(ar).astype(str), ["True", "1"]) if ar.dtype.kind in ("S", "O")
            else ar.astype(bool))


for sl in (sys.argv[1:] or ["1", "2", "3"]):
    with h5py.File(TMPL.format(sl), "r") as h5:
        X = _X(h5); var = _var(h5); tum = _bool(h5, "pred_tumor_XGBoost")
    Xk = X[~tum]; del X
    raw = {g: np.asarray(Xk[:, var.index(g)].todense()).ravel() for g in ALLG if g in var}

    def c(genes):
        return np.vstack([raw[g] > 0 for g in genes]).sum(0)

    pool = (c(PAN) >= 3) & (c(LINEAGE) >= 1)
    bam = pool & (c(BAM) >= 2)
    mdm = pool & ~bam & (raw["Ccr2"] > 0) & (c(MDM_SUP) >= 1) & (c(BAM_STABLE) == 0)
    for kmi in (1, 2):
        micro = pool & ~bam & ~mdm & (c(MICRO) >= kmi)
        unres = pool & ~bam & ~mdm & ~micro
        role = "CONTROL" if sl == "3" else "tumor"
        print(f"slice {sl} ({role}) pool={int(pool.sum()):,}  micro_k={kmi}:  "
              f"Microglia={int(micro.sum()):,}  BAM={int(bam.sum()):,}  "
              f"MDM={int(mdm.sum()):,}  unresolved={int(unres.sum()):,}")
