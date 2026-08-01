"""Ambiguous-bin rescue using the low-ambient LINEAGE myeloid genes as the
tie-breaker (generalizes over all competitors, not just Meg3/Neurons).

Track A: score_genes winner-take-all -> confident / ambiguous / unknown.
Rescue: ambiguous cell with Myeloid in its top-2 AND >=k lineage genes detected
        (Csf1r/Aif1/Tyrobp/Fcer1g) -> Myeloid.  'unknown' (clump) untouched.
Reports k=1 and k=2, checks the slice-3 clump, and maps the result (rescued
highlighted, tumor black).  Output -> score_genes_slice{n}_v2/rescue/
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
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
KRESCUE = 2                         # require >=2 lineage genes to rescue to Myeloid


def M(sl):
    return {"Astrocytes": ["Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9"],
            "Neurons": ["Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1", "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst"],
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


def _num(h, c):
    nd = h["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...]).astype(float); return ct[np.clip(cd, 0, None)]
    return nd[...].astype(float)


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


def run(sl):
    out = f"D:/thesis-research/score_genes_slice{sl}_v2/rescue"; os.makedirs(out, exist_ok=True)
    mk = M(sl); labels = list(mk)
    with h5py.File(TMPL.format(sl), "r") as h:
        X = _X(h); var = _var(h)
        cx = _num(h, "CenterX_global_px"); cy = _num(h, "CenterY_global_px"); tum = _bool(h, "pred_tumor_XGBoost")
    Xk = X[~tum].tocsr(); del X
    cxk, cyk = cx[~tum], cy[~tum]
    vi = {g: var.index(g) for g in set(sum(mk.values(), [])) if g in var}
    cov = {l: np.asarray((Xk[:, [vi[g] for g in mk[l] if g in vi]] > 0).sum(1)).ravel() for l in labels}
    linedet = np.vstack([np.asarray(Xk[:, var.index(g)].todense()).ravel() > 0 for g in LINEAGE if g in var]).sum(0)

    a = ad.AnnData(X=Xk.astype(np.float32)); a.var_names = pd.Index(var); a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    S = pd.DataFrame({l: sg(a, mk[l], f"s_{l}") for l in labels})[labels]

    raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, 0)), 0)
    mad = np.where(mad > 0, mad, raw.std(0)); mad = np.where(mad > 0, mad, 1.0)
    scd = raw / mad; order = np.argsort(-scd, 1); lab = np.asarray(labels)
    top_l, sec_l = lab[order[:, 0]], lab[order[:, 1]]
    top = np.take_along_axis(scd, order[:, :1], 1)[:, 0]; sec = np.take_along_axis(scd, order[:, 1:2], 1)[:, 0]
    margin = (top > 0) & ((sec <= 0) | (top >= RATIO * sec))
    thr = {l: fdr_thr(S[l].to_numpy(), FDR) for l in labels}
    top_raw = raw[np.arange(raw.shape[0]), order[:, 0]]
    fdrp = top_raw >= np.array([thr[l] for l in top_l])
    cov_top = np.array([cov[top_l[i]][i] for i in range(len(top_l))])
    covp = cov_top >= np.array([COVK[l] for l in top_l])
    A = np.full(len(top_l), "unknown", dtype=object)
    A[fdrp & ~margin] = "ambiguous"
    A[fdrp & margin & ~covp] = "low_markers_coverage"
    ok = fdrp & margin & covp; A[ok] = top_l[ok]

    myc = (A == "ambiguous") & ((top_l == "Myeloid") | (sec_l == "Myeloid"))
    role = "CONTROL" if sl not in TUMOR_SLICES else "tumor"
    print(f"\n=== slice {sl} ({role}) ===  Myeloid A={int((A=='Myeloid').sum()):,}, "
          f"myeloid-candidate ambiguous={int(myc.sum()):,}")
    for k in (1, 2):
        r = myc & (linedet >= k)
        print(f"  rescue k>={k} lineage: +{int(r.sum()):,} -> Myeloid final={int((A=='Myeloid').sum())+int(r.sum()):,}")

    resc = myc & (linedet >= KRESCUE)
    final = A.copy(); final[resc] = "Myeloid"
    if sl == 3:
        cl = (cxk >= 21776) & (cxk <= 29740) & (cyk >= -2660) & (cyk <= 5303)
        print(f"  clump rescued->Myeloid (k>={KRESCUE}): {int((resc & cl).sum())} (should be ~0)")
    pd.DataFrame({"x": cxk, "y": cyk, "trackA": A, "final": final}).to_csv(f"{out}/labels_lineage.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    ax.scatter(cxk, cyk, s=0.5, c="#eee", linewidths=0, rasterized=True)
    if tum.sum():
        ax.scatter(cx[tum], cy[tum], s=1.2, c="black", linewidths=0, rasterized=True, label=f"tumor ({int(tum.sum()):,})")
    base = (A == "Myeloid")
    ax.scatter(cxk[base], cyk[base], s=4, c="#00a087", linewidths=0, rasterized=True,
               label=f"Myeloid Track-A ({int(base.sum()):,})")
    ax.scatter(cxk[resc], cyk[resc], s=6, c="#ff7f0e", linewidths=0, rasterized=True,
               label=f"Myeloid rescued ({int(resc.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {sl} ({role}) Myeloid: score_genes + lineage ambiguous-rescue (k>={KRESCUE})")
    ax.legend(loc="lower right", markerscale=3, fontsize=8)
    fig.savefig(f"{out}/slice{sl}_myeloid_lineage_rescue.png", bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> {out}")


for s in (int(x) for x in (sys.argv[1:] or [1, 3])):
    run(s)
