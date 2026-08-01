"""High-confidence MDM/BAM by positive CO-DETECTION of specific markers, drawn
from a clean inclusive myeloid pool (detect>=3 pan-myeloid AND >=1 lineage gene).
Compares tumor (1) vs control (3): does co-detection give MORE than the current
soft gate (163 MDM / 401 BAM on slice 1) while staying LOW in control?"""
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
MDM_ANCH = ["Ccr2", "Plac8"]
MDM_SUP = ["Vcan", "Cd14", "Ccr1", "Fpr1"]
BAM = ["Pf4", "Maf", "Mrc1", "Cd163", "Cd5l"]
BAM_STABLE = ["Pf4", "Maf"]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ALL = PAN + MDM_ANCH + MDM_SUP + BAM


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


def cnt(raw, genes, mask):
    return np.vstack([raw[g][mask] > 0 for g in genes]).sum(0)


import pandas as pd


def report(name, raw, p, nt):
    ccr2 = raw["Ccr2"][p] > 0
    mdm_sup = cnt(raw, MDM_ANCH[1:] + MDM_SUP, p) >= 1        # Plac8 or a support gene
    bstab = cnt(raw, BAM_STABLE, p) >= 1
    mdm_hc = ccr2 & mdm_sup & ~bstab
    mdm_2anch = cnt(raw, MDM_ANCH, p) >= 2                    # Ccr2 AND Plac8 (strictest)
    bam_hc = cnt(raw, BAM, p) >= 2
    bam_anch = (cnt(raw, BAM_STABLE, p) >= 1) & (cnt(raw, ["Mrc1", "Cd163", "Cd5l"], p) >= 1)
    print(f"  [{name}] pool {len(p):,} ({100*len(p)/nt:.1f}%)  "
          f"MDM Ccr2+sup={int(mdm_hc.sum()):,} (Ccr2&Plac8={int(mdm_2anch.sum()):,})  "
          f"BAM>=2/5={int(bam_hc.sum()):,} ((Pf4/Maf)&sup={int(bam_anch.sum()):,})")


for sl in (sys.argv[1:] or ["1", "3"]):
    with h5py.File(TMPL.format(sl), "r") as h5:
        X = _X(h5); var = _var(h5); tum = _bool(h5, "pred_tumor_XGBoost")
    Xk = X[~tum]; del X
    raw = {g: np.asarray(Xk[:, var.index(g)].todense()).ravel() for g in ALL if g in var}
    nt = Xk.shape[0]

    incl = np.where((cnt(raw, PAN, slice(None)) >= 3) & (cnt(raw, LINEAGE, slice(None)) >= 1))[0]
    cs = pd.read_csv(f"D:/thesis-research/score_genes_slice{sl}_v2/cell_scores.csv")
    strict = np.where(cs["celltype_v2"].to_numpy() == "Myeloid")[0]

    role = "CONTROL" if sl == "3" else "tumor"
    print(f"\n=== slice {sl} ({role}) ===  non-tumor {nt:,}")
    report("STRICT Stage-1 pool", raw, strict, nt)
    report("inclusive pool     ", raw, incl, nt)
