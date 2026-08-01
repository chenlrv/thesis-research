"""FINAL committed annotation method (L321: slices 1,2,3).

Backbone (score_genes winner-take-all, FDR-FIRST):
  types = Astrocytes, Neurons, Myeloid(=LINEAGE Csf1r/Aif1/Tyrobp/Fcer1g, C1q dropped),
          Ependymal(Ttr sink -> relabeled), Vascular
  per cell: candidates = types passing own mirrored-FDR (raw>=thr);
            none -> unknown; winner = top MAD-scaled AMONG candidates;
            margin = winner >= 1.5x second-among-candidates (auto-pass if 1 candidate & >0);
            coverage gate on RAW counts (>=k detected); else ambiguous/low.
Recover: ambiguous with Myeloid in top-2 (among candidates) AND >=2 lineage genes -> Myeloid.
NK veto: drop myeloid cells that are Nkg7/Klrb1c+ AND Csf1r=0 AND Aif1=0.
Choroid: Ttr>=5 raw gate (replaces scored Ependymal; Ependymal label -> unknown).
Subtype myeloid (raw co-detection, order BAM>MDM>Micro):
  BAM   = >=2 of {Pf4,Maf,Mrc1,Cd163,Cd5l}
  MDM   = Ccr2 & >=1 of {Plac8,Vcan,Cd14,Ccr1,Fpr1} & Pf4/Maf-negative
  Micro = (>=1 homeostatic {Adgrg1,Hpgds,Gpr183} OR C1q-high >=2 of {C1qa,C1qb,C1qc}) & not BAM/MDM
  else  = Myeloid_unresolved
Output -> score_genes_slice{n}_v2/final_v3/  (per-type + combined maps, tumor black; labels csv)
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

SLICES = [1, 2, 3]
TUMOR = {1, 2, 5, 6}
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
RATIO, TTR_THR, KRESCUE = 1.5, 5, 2
COVK = {"Astrocytes": 2, "Neurons": 2, "Myeloid": 2, "Ependymal": 2, "Vascular": 4, "Lymphoid": 2}
LIN = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LYMPHg = ["Cd3e", "Cd3d", "Cd8a", "Cd8b1", "Cd6", "Il7r", "Ms4a1", "Cd79a", "Nkg7", "Klrb1c", "Ccl19"]
MICRO_HOM = ["Adgrg1", "Hpgds", "Gpr183"]; C1Q = ["C1qa", "C1qb", "C1qc"]
BAMg = ["Pf4", "Maf", "Mrc1", "Cd163", "Cd5l"]; BSTAB = ["Pf4", "Maf"]
MDMsup = ["Plac8", "Vcan", "Cd14", "Ccr1", "Fpr1"]
BIO = [("Astrocytes", "#1f77b4"), ("Neurons", "#e377c2"), ("Microglia", "#17becf"),
       ("BAM", "#d62728"), ("MDM", "#00a087"), ("Myeloid_unresolved", "#8c564b"),
       ("Vascular", "#2ca02c"), ("Choroid", "#9467bd"), ("Lymphoid", "#ff7f0e")]
DISP = {"Myeloid_unresolved": "Myeloid (unresolved)"}


def base(sl):
    return {"Astrocytes": ["Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9"],
            "Neurons": ["Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1", "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst"],
            "Myeloid": LIN,
            "Ependymal": (["Ttr", "Adgrv1", "Cd24a"] if sl in TUMOR
                          else ["Ttr", "Adgrv1", "Cd24a", "Krt8", "Krt18", "Krt19", "Cldn4", "Epcam"]),
            "Vascular": ["Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Tie1", "Esam", "Slc2a1",
                         "Clec14a", "Adgrl4", "Eng", "Icam2", "Ramp2", "Vwf", "Rgs5", "Pdgfrb", "Notch3", "Vtn"],
            "Lymphoid": LYMPHg}


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


def _bl(h, c):
    nd = h["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...])
        return np.isin(np.where(cd >= 0, ct[np.clip(cd, 0, None)], "False").astype(str), ["True", "1", "1.0", "TRUE", "true"])
    ar = nd[...]
    return (np.isin(_dec(ar).astype(str), ["True", "1"]) if ar.dtype.kind in ("S", "O") else ar.astype(bool))


def ft(s, f=0.05):
    s = np.asarray(s, float)
    for t in np.unique(np.sort(s[s > 0])):
        if int((s <= -t).sum()) / max(int((s >= t).sum()), 1) <= f:
            return float(t)
    return np.inf


def sg(a, g, n):
    sc.tl.score_genes(a, [x for x in g if x in a.var_names], score_name=n, ctrl_size=50, n_bins=25, random_state=0)
    return a.obs[n].to_numpy()


def decide(S, cov, L, lindet):
    """FDR-FIRST winner-take-all -> array of {label, ambiguous, low, unknown} + top2 among candidates."""
    raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, 0)), 0); mad = np.where(mad > 0, mad, raw.std(0)); mad = np.where(mad > 0, mad, 1.0)
    scaled = raw / mad
    thr = np.array([ft(S[l].to_numpy()) for l in L])
    passing = raw >= thr                                   # per-cell per-type FDR pass
    npass = passing.sum(1)
    masked = np.where(passing, scaled, -np.inf)            # rank only among FDR-passing types
    order = np.argsort(-masked, 1); lab = np.array(L)
    topl = lab[order[:, 0]]; secl = lab[order[:, 1]]
    top = np.take_along_axis(masked, order[:, :1], 1)[:, 0]
    sec = np.take_along_axis(masked, order[:, 1:2], 1)[:, 0]
    margin = (top > 0) & ((sec <= 0) | np.isneginf(sec) | (top >= RATIO * sec))
    covt = np.array([cov[topl[i]][i] for i in range(len(topl))])
    covp = covt >= np.array([COVK[l] for l in topl])
    out = np.full(len(topl), "unknown", dtype=object)
    has = npass >= 1
    out[has & ~margin] = "ambiguous"
    out[has & margin & ~covp] = "low"
    ok = has & margin & covp; out[ok] = topl[ok]
    # lineage rescue: ambiguous with Myeloid in top-2 among candidates & >=2 lineage
    myin = (topl == "Myeloid") | ((secl == "Myeloid") & (npass >= 2))
    out[(out == "ambiguous") & myin & (lindet >= KRESCUE)] = "Myeloid"
    return out


def run(sl, src=None, outdir="final_v3"):
    out = f"D:/thesis-research/score_genes_slice{sl}_v2/{outdir}"; os.makedirs(out, exist_ok=True)
    bm = base(sl); L = list(bm)
    with h5py.File(src or TMPL.format(sl), "r") as h:
        X = _X(h); var = _var(h)
        cx = _num(h, "CenterX_global_px"); cy = _num(h, "CenterY_global_px"); tum = _bl(h, "pred_tumor_XGBoost")
    Xk = X[~tum].tocsr(); del X
    cxk, cyk = cx[~tum], cy[~tum]
    gi = lambda g: np.asarray(Xk[:, var.index(g)].todense()).ravel()
    c = lambda gs: np.vstack([gi(g) > 0 for g in gs if g in var]).sum(0)
    lindet = c(LIN)
    cov = {l: np.asarray((Xk[:, [var.index(g) for g in bm[l] if g in var]] > 0).sum(1)).ravel() for l in L}

    a = ad.AnnData(X=Xk.astype(np.float32)); a.var_names = pd.Index(var); a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    S = pd.DataFrame({l: sg(a, bm[l], f"s_{l}") for l in L})[L]
    lab = decide(S, cov, L, lindet)

    # NK veto on myeloid
    nksusp = ((gi("Nkg7") > 0) | (gi("Klrb1c") > 0)) & (gi("Csf1r") == 0) & (gi("Aif1") == 0)
    nk_removed = int(((lab == "Myeloid") & nksusp).sum())
    lab[(lab == "Myeloid") & nksusp] = "unknown"
    myeloid = lab == "Myeloid"

    # subtype (raw co-detection; order BAM > MDM > Micro)
    bam = myeloid & (c(BAMg) >= 2)
    mdm = myeloid & ~bam & (gi("Ccr2") > 0) & (c(MDMsup) >= 1) & (c(BSTAB) == 0)
    micro = myeloid & ~bam & ~mdm & ((c(MICRO_HOM) >= 1) | (c(C1Q) >= 2))
    final = lab.astype(object).copy()
    final[micro] = "Microglia"; final[bam] = "BAM"; final[mdm] = "MDM"
    final[myeloid & ~(bam | mdm | micro)] = "Myeloid_unresolved"
    # choroid replaces scored Ependymal
    final[final == "Ependymal"] = "unknown"
    final[gi("Ttr") >= TTR_THR] = "Choroid"
    final[np.isin(final, ["ambiguous", "low", "unknown"])] = "unassigned"

    tot = len(final) + int(tum.sum())
    print(f"\n=== slice {sl} ({'CONTROL' if sl not in TUMOR else 'tumor'}) ===  {tot:,} cells; myeloid={int(myeloid.sum()):,}; NK-vetoed={nk_removed}")
    vc = pd.Series(final).value_counts()
    for k, _ in BIO:
        if k in vc: print(f"   {DISP.get(k, k):22s} {vc[k]:>7,}  {100*vc[k]/tot:5.1f}%")
    print(f"   {'unassigned':22s} {int((final == 'unassigned').sum()):>7,}")
    pd.DataFrame({"x": cxk, "y": cyk, "celltype_final": final}).to_csv(f"{out}/annotation.csv", index=False)
    plots(sl, cxk, cyk, final, cx[tum], cy[tum], tot, out)


def plots(sl, x, y, final, tx, ty, tot, out):
    tl = f"tumor ({len(tx):,})"
    for k, col in BIO:
        m = final == k
        fig, ax = plt.subplots(figsize=(9, 7), dpi=155)
        ax.scatter(x, y, s=0.6, c="#e3e3e3", linewidths=0, rasterized=True, label="other")
        if len(tx): ax.scatter(tx, ty, s=1.2, c="black", linewidths=0, rasterized=True, label=tl)
        ax.scatter(x[m], y[m], s=1.2, c=col, linewidths=0, rasterized=True, label=DISP.get(k, k))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {sl} — {DISP.get(k, k)}  (n={int(m.sum()):,}, {100*m.sum()/tot:.1f}%)")
        ax.legend(loc="lower right", markerscale=10, fontsize=8)
        fig.savefig(f"{out}/slice{sl}_{k}.png", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=165)
    m0 = final == "unassigned"
    ax.scatter(x[m0], y[m0], s=0.6, c="#ececec", linewidths=0, rasterized=True, label=f"unassigned ({int(m0.sum()):,})")
    if len(tx): ax.scatter(tx, ty, s=1.2, c="black", linewidths=0, rasterized=True, label=tl)
    for k, col in BIO:
        m = final == k
        if m.any(): ax.scatter(x[m], y[m], s=1.2, c=col, linewidths=0, rasterized=True, label=f"{DISP.get(k, k)} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {sl} — final annotation (committed method)")
    ax.legend(loc="lower right", markerscale=10, fontsize=8)
    fig.savefig(f"{out}/slice{sl}_ALL.png", bbox_inches="tight"); plt.close(fig)
    print(f"   saved -> {out}")


if __name__ == "__main__":
    for s in (int(x) for x in (sys.argv[1:] or SLICES)):
        run(s)
