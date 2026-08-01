"""Two-track annotation: score_genes winner-take-all, then rescue ONLY the
'ambiguous' cells by re-breaking the tie with the over-ambient gene Meg3 dropped
from the Neuron module. 'unknown' (spillover, incl. the clump) is left untouched.

Track A (all markers)  -> confident / ambiguous / low_cov / unknown   (= celltype_v2)
Track B (Neurons w/o Meg3) -> re-decision, applied ONLY where Track A == ambiguous.
final = A where A is confident/unknown/low_cov ; = B where A == ambiguous.

Reports the rescue breakdown + a Myeloid map (rescued cells highlighted, tumor black).
Slices from argv (default 1 3). Output -> score_genes_slice{n}_v2/rescue/
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

TUMOR_SLICES = {1, 2, 5, 6}
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
FDR, RATIO = 0.05, 1.5
COVK = {"Astrocytes": 2, "Neurons": 2, "Myeloid": 2, "Ependymal": 2, "Vascular": 4}
_EP_FULL = ["Ttr", "Adgrv1", "Cd24a", "Krt8", "Krt18", "Krt19", "Cldn4", "Epcam"]
_EP_SAFE = ["Ttr", "Adgrv1", "Cd24a"]
NEUR_FULL = ["Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1", "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst"]
AMBIENT = ["Meg3"]                                   # dropped in the rescue track


def markers(sl):
    return {
        "Astrocytes": ["Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9"],
        "Neurons": NEUR_FULL,
        "Myeloid": ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"],
        "Ependymal": _EP_SAFE if sl in TUMOR_SLICES else _EP_FULL,
        "Vascular": ["Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Tie1", "Esam", "Slc2a1",
                     "Clec14a", "Adgrl4", "Eng", "Icam2", "Ramp2", "Vwf",
                     "Rgs5", "Pdgfrb", "Notch3", "Vtn"],
    }


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


def fdr_thr(scores, fdr=0.05):
    scores = np.asarray(scores, float)
    for t in np.unique(np.sort(scores[scores > 0])):
        if int(np.sum(scores <= -t)) / max(int(np.sum(scores >= t)), 1) <= fdr:
            return float(t)
    return np.inf


def decide(S, cov, labels):
    """FDR + 1.5x MAD-margin + coverage -> array of {label, ambiguous, low_cov, unknown}."""
    thr = {l: fdr_thr(S[l].to_numpy(), FDR) for l in labels}
    raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, 0)), 0)
    mad = np.where(mad > 0, mad, raw.std(0)); mad = np.where(mad > 0, mad, 1.0)
    sc_ = raw / mad
    order = np.argsort(-sc_, 1); lab = np.asarray(labels)
    top_l = lab[order[:, 0]]
    top = np.take_along_axis(sc_, order[:, :1], 1)[:, 0]
    second = np.take_along_axis(sc_, order[:, 1:2], 1)[:, 0]
    margin = (top > 0) & ((second <= 0) | (top >= RATIO * second))
    top_raw = raw[np.arange(raw.shape[0]), order[:, 0]]
    fdrp = top_raw >= np.array([thr[l] for l in top_l])
    cov_top = np.array([cov[top_l[i]][i] for i in range(len(top_l))])
    covp = cov_top >= np.array([COVK[l] for l in top_l])
    out = np.full(len(top_l), "unknown", dtype=object)
    out[fdrp & ~margin] = "ambiguous"
    out[fdrp & margin & ~covp] = "low_markers_coverage"
    ok = fdrp & margin & covp
    out[ok] = top_l[ok]
    return out


def sg(a, genes, nm):
    g = [x for x in genes if x in a.var_names]
    sc.tl.score_genes(a, gene_list=g, score_name=nm, ctrl_size=50, n_bins=25, random_state=0)
    return a.obs[nm].to_numpy()


def run(sl):
    out = f"D:/thesis-research/score_genes_slice{sl}_v2/rescue"; os.makedirs(out, exist_ok=True)
    M = markers(sl); labels = list(M)
    with h5py.File(TMPL.format(sl), "r") as h5:
        X = _X(h5); var = _var(h5)
        cx = _num(h5, "CenterX_global_px"); cy = _num(h5, "CenterY_global_px")
        tum = _bool(h5, "pred_tumor_XGBoost")
    keep = ~tum; Xk = X[keep].tocsr(); del X
    cxk, cyk = cx[keep], cy[keep]
    vi = {g: var.index(g) for g in set(sum(M.values(), [])) if g in var}
    cov = {l: np.asarray((Xk[:, [vi[g] for g in M[l] if g in vi]] > 0).sum(1)).ravel() for l in labels}

    a = ad.AnnData(X=Xk.astype(np.float32)); a.var_names = pd.Index(var); a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    S = pd.DataFrame({l: sg(a, M[l], f"s_{l}") for l in labels})[labels]
    A = decide(S, cov, labels)                       # Track A (with Meg3)

    # Track B: Neurons without Meg3
    Sb = S.copy()
    Sb["Neurons"] = sg(a, [g for g in NEUR_FULL if g not in AMBIENT], "s_N_noMeg3")
    B = decide(Sb, cov, labels)

    amb = A == "ambiguous"
    final = A.copy()
    final[amb] = B[amb]                               # rescue only the ambiguous bin

    role = "CONTROL" if sl not in TUMOR_SLICES else "tumor"
    resc = amb & (B != "ambiguous")
    print(f"\n=== slice {sl} ({role}) : {int(keep.sum()):,} non-tumor ===")
    print(f"  Track-A ambiguous: {int(amb.sum()):,}")
    print(f"  rescued: {int(resc.sum()):,}  -> " + str(pd.Series(B[resc]).value_counts().to_dict()))
    print(f"  still ambiguous: {int((amb & (B=='ambiguous')).sum()):,}")
    print(f"  Myeloid: A={int((A=='Myeloid').sum()):,}  ->  final={int((final=='Myeloid').sum()):,} "
          f"(+{int((resc & (B=='Myeloid')).sum()):,})")
    # clump check (slice 3): do any rescued-myeloid land in the spillover clump?
    if sl == 3:
        cl = (cxk >= 21776) & (cxk <= 29740) & (cyk >= -2660) & (cyk <= 5303)
        print(f"  clump rescued-to-Myeloid: {int((resc & (B=='Myeloid') & cl).sum())} (should be ~0)")

    pd.DataFrame({"x": cxk, "y": cyk, "trackA": A, "final": final}).to_csv(f"{out}/labels.csv", index=False)

    # map: final Myeloid, rescued highlighted, tumor black
    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    ax.scatter(cxk, cyk, s=0.5, c="#eee", linewidths=0, rasterized=True)
    if tum.sum():
        ax.scatter(cx[tum], cy[tum], s=1.2, c="black", linewidths=0, rasterized=True, label=f"tumor ({int(tum.sum()):,})")
    base = (final == "Myeloid") & ~resc
    rm = resc & (B == "Myeloid")
    ax.scatter(cxk[base], cyk[base], s=4, c="#00a087", linewidths=0, rasterized=True,
               label=f"Myeloid (Track-A conf, {int(base.sum()):,})")
    ax.scatter(cxk[rm], cyk[rm], s=6, c="#ff7f0e", linewidths=0, rasterized=True,
               label=f"Myeloid (rescued, {int(rm.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {sl} ({role}) Myeloid: score_genes + Meg3-ambiguous rescue")
    ax.legend(loc="lower right", markerscale=3, fontsize=8)
    fig.savefig(f"{out}/slice{sl}_myeloid_rescue.png", bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out}")


for s in (int(x) for x in (sys.argv[1:] or [1, 3])):
    run(s)
