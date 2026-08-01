"""Diagnose the ambiguous bin: are the myeloid-candidate ambiguous cells real
myeloid, and what are they tied AGAINST (which type, which genes = the ambient
confounder)?  Slices 1 (tumor) and 3 (control)."""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

TUMOR_SLICES = {1, 2, 5, 6}
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
FDR, RATIO = 0.05, 1.5
COVK = {"Astrocytes": 2, "Neurons": 2, "Myeloid": 2, "Ependymal": 2, "Vascular": 4}
NEUR = ["Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1", "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]


def M(sl):
    return {"Astrocytes": ["Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9"],
            "Neurons": NEUR,
            "Myeloid": ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"],
            "Ependymal": (["Ttr", "Adgrv1", "Cd24a"] if sl in TUMOR_SLICES
                          else ["Ttr", "Adgrv1", "Cd24a", "Krt8", "Krt18", "Krt19", "Cldn4", "Epcam"]),
            "Vascular": ["Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Tie1", "Esam", "Slc2a1",
                         "Clec14a", "Adgrl4", "Eng", "Icam2", "Ramp2", "Vwf", "Rgs5",
                         "Pdgfrb", "Notch3", "Vtn"]}


def _dec(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def _X(h):
    n = h["X"]; e = str(n.attrs.get("encoding-type", "")); s = tuple(n.attrs["shape"])
    a = (n["data"][...], n["indices"][...], n["indptr"][...])
    return (csc_matrix(a, shape=s) if "csc" in e else csr_matrix(a, shape=s)).tocsr()


def _var(h):
    v = h["var"]; k = v.attrs.get("_index", "_index")
    return list(_dec(v[k.decode() if isinstance(k, bytes) else k][...]))


def _bool(h, c):
    nd = h["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...])
        return np.isin(np.where(cd >= 0, ct[np.clip(cd, 0, None)], "False").astype(str),
                       ["True", "1", "1.0", "TRUE", "true"])
    ar = nd[...]
    return (np.isin(_dec(ar).astype(str), ["True", "1"]) if ar.dtype.kind in ("S", "O") else ar.astype(bool))


def fdr_thr(s, fdr=0.05):
    s = np.asarray(s, float)
    for t in np.unique(np.sort(s[s > 0])):
        if int((s <= -t).sum()) / max(int((s >= t).sum()), 1) <= fdr:
            return float(t)
    return np.inf


def sg(a, genes, nm):
    g = [x for x in genes if x in a.var_names]
    sc.tl.score_genes(a, gene_list=g, score_name=nm, ctrl_size=50, n_bins=25, random_state=0)
    return a.obs[nm].to_numpy()


for sl in (int(x) for x in (sys.argv[1:] or [1, 3])):
    mk = M(sl); labels = list(mk)
    with h5py.File(TMPL.format(sl), "r") as h:
        X = _X(h); var = _var(h); tum = _bool(h, "pred_tumor_XGBoost")
    Xk = X[~tum].tocsr(); del X
    vi = {g: var.index(g) for g in set(sum(mk.values(), [])) if g in var}
    cov = {l: np.asarray((Xk[:, [vi[g] for g in mk[l] if g in vi]] > 0).sum(1)).ravel() for l in labels}
    rawL = {g: np.asarray(Xk[:, var.index(g)].todense()).ravel() for g in set(NEUR + LINEAGE) if g in var}

    a = ad.AnnData(X=Xk.astype(np.float32)); a.var_names = pd.Index(var); a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    S = pd.DataFrame({l: sg(a, mk[l], f"s_{l}") for l in labels})[labels]

    raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, 0)), 0)
    mad = np.where(mad > 0, mad, raw.std(0)); mad = np.where(mad > 0, mad, 1.0)
    scd = raw / mad
    order = np.argsort(-scd, 1); lab = np.asarray(labels)
    top_l, sec_l = lab[order[:, 0]], lab[order[:, 1]]
    top = np.take_along_axis(scd, order[:, :1], 1)[:, 0]
    sec = np.take_along_axis(scd, order[:, 1:2], 1)[:, 0]
    margin = (top > 0) & ((sec <= 0) | (top >= RATIO * sec))
    thr = {l: fdr_thr(S[l].to_numpy(), FDR) for l in labels}
    top_raw = raw[np.arange(raw.shape[0]), order[:, 0]]
    fdrp = top_raw >= np.array([thr[l] for l in top_l])
    ambiguous = fdrp & ~margin

    inpair = (top_l == "Myeloid") | (sec_l == "Myeloid")
    myc = ambiguous & inpair
    role = "CONTROL" if sl not in TUMOR_SLICES else "tumor"
    print(f"\n=== slice {sl} ({role}) ===  ambiguous={int(ambiguous.sum()):,}, "
          f"Myeloid in top-2 of ambiguous={int(myc.sum()):,}")
    print("  Myeloid is TOP (won, <1.5x):", int((myc & (top_l == 'Myeloid')).sum()),
          " | Myeloid is SECOND (lost):", int((myc & (sec_l == 'Myeloid')).sum()))
    other = np.where(top_l[myc] == "Myeloid", sec_l[myc], top_l[myc])
    print("  tied AGAINST:", pd.Series(other).value_counts().to_dict())

    # of the myeloid-candidate ambiguous, are they real myeloid? lineage vs neuron breadth
    ndet = lambda genes, m: np.vstack([rawL[g][m] > 0 for g in genes if g in rawL]).sum(0).mean()
    conf_my = top_l == "Myeloid"
    conf_my = conf_my & fdrp & margin
    print(f"  #lineage detected  myc={ndet(LINEAGE,myc):.2f}  confMyeloid={ndet(LINEAGE,conf_my):.2f}")
    # which neuron genes drive the tie (detection% in myc vs whole)
    print("  neuron-gene detection% in myeloid-candidate-ambiguous (drivers of the tie):")
    for g in NEUR:
        if g in rawL:
            print(f"    {g:8} {100*(rawL[g][myc]>0).mean():>5.0f}%  (all cells {100*(rawL[g]>0).mean():>4.0f}%)")
