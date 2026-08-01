"""Stage-2 myeloid subtyping with the STAGE-1 decision rule (no control calibration).

Per slice (1,2,3), independently:
  pool = clean inclusive myeloid (detect >=3 pan-myeloid AND >=1 lineage gene)
  score_genes(Microglia/BAM/MDM) on the pool (norm+log1p)
  -> mirrored-FDR threshold per subtype (FDR<=0.05, negative-tail decoy)
  -> MAD-scaled margin: top >= 1.5x second
  -> coverage gate: winning subtype must express >= k of its markers
Outcomes (like Stage-1): unresolved (FDR fail) / ambiguous (margin fail) /
low_markers_coverage (cov fail) / Microglia|BAM|MDM.

No slice-3 control calibration -- each slice self-calibrated by its own FDR.
Pool: pass "strict" as argv[1] to use the Stage-1 celltype_v2=="Myeloid" set instead.

Output -> score_genes_slice{n}_v2/myeloid_stage2_fdr.png|labels.csv,
          score_genes_slice_all/myeloid_stage2_fdr_counts.csv
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
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
POOL_MODE = sys.argv[1] if len(sys.argv) > 1 else "inclusive"
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ALLOUT = "D:/thesis-research/score_genes_slice_all"
TUMOR_COL = "pred_tumor_XGBoost"
FDR, RATIO = 0.05, 1.5

PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
MARKERS = {
    "Microglia": ["Adgrg1", "Hpgds", "Gpr183"],
    "BAM":       ["Mrc1", "Cd163", "Pf4", "Maf", "Cd5l"],
    "MDM":       ["Ccr2", "Plac8", "Cd14", "Ccr1", "Fpr1", "Crip1",
                  "S100a8", "S100a9", "Vcan", "Fn1", "Itgal"],
}
KCOV = {"Microglia": 1, "BAM": 2, "MDM": 2}   # "# expressed genes" threshold per subtype
SUBS = list(MARKERS)
COL = {"Microglia": "#17becf", "BAM": "#d62728", "MDM": "#00a087"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


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


def mirrored_fdr_threshold(scores, fdr=0.05):
    scores = np.asarray(scores, float)
    for t in np.unique(np.sort(scores[scores > 0])):
        if int(np.sum(scores <= -t)) / max(int(np.sum(scores >= t)), 1) <= fdr:
            return float(t)
    return np.inf


def scaled_margin_calls(S, thr, ratio):
    labels = list(S.columns); raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, 0)), 0)
    mad = np.where(mad > 0, mad, raw.std(0)); mad = np.where(mad > 0, mad, 1.0)
    sc = raw / mad
    order = np.argsort(-sc, 1); lab = np.asarray(labels)
    top_l = lab[order[:, 0]]
    top = np.take_along_axis(sc, order[:, :1], 1)[:, 0]
    second = np.take_along_axis(sc, order[:, 1:2], 1)[:, 0]
    margin = (top > 0) & ((second <= 0) | (top >= ratio * second))
    top_raw = raw[np.arange(raw.shape[0]), order[:, 0]]
    fdrp = top_raw >= np.array([thr[l] for l in top_l])
    return top_l, fdrp, margin


def sg(adata, genes, name):
    g = [x for x in genes if x in adata.var_names]
    sc.tl.score_genes(adata, gene_list=g, score_name=name, ctrl_size=50, n_bins=25, random_state=0)
    return adata.obs[name].to_numpy()


def run(n):
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _X(h5); var = _var(h5)
        cx = _num(h5, "CenterX_global_px"); cy = _num(h5, "CenterY_global_px")
        tumor = _bool(h5, TUMOR_COL)
    keep = ~tumor
    Xk = X[keep].tocsr(); del X
    cxk, cyk = cx[keep], cy[keep]
    vi = {g: var.index(g) for g in set(PAN + LINEAGE + sum(MARKERS.values(), [])) if g in var}

    def det(genes):
        return np.asarray((Xk[:, [vi[g] for g in genes if g in vi]] > 0).sum(1)).ravel()

    if POOL_MODE == "strict":
        cs = pd.read_csv(f"D:/thesis-research/score_genes_slice{n}_v2/cell_scores.csv")
        pool = (cs["celltype_v2"].to_numpy() == "Myeloid")
    else:
        pool = (det(PAN) >= 3) & (det(LINEAGE) >= 1)
    pidx = np.where(pool)[0]
    cov = {s: np.asarray((Xk[pool][:, [vi[g] for g in MARKERS[s] if g in vi]] > 0).sum(1)).ravel()
           for s in SUBS}

    a = ad.AnnData(X=Xk[pool].astype(np.float32)); a.var_names = pd.Index(var); a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    S = pd.DataFrame({s: sg(a, MARKERS[s], f"s_{s}") for s in SUBS})[SUBS]
    thr = {s: mirrored_fdr_threshold(S[s].to_numpy(), FDR) for s in SUBS}
    top_l, fdrp, margin = scaled_margin_calls(S, thr, RATIO)
    cov_top = np.array([cov[top_l[i]][i] for i in range(len(top_l))])
    k_top = np.array([KCOV[l] for l in top_l])
    covp = cov_top >= k_top

    lab = np.full(len(pidx), "unresolved", dtype=object)
    lab[fdrp & ~margin] = "ambiguous"
    lab[fdrp & margin & ~covp] = "low_markers_coverage"
    ok = fdrp & margin & covp
    lab[ok] = top_l[ok]

    cnt = {s: int((lab == s).sum()) for s in SUBS}
    role = " (CONTROL)" if n == 3 else ""
    print(f"\nslice {n}{role}: pool {len(pidx):,}  FDR thr " +
          " ".join(f"{s}={thr[s]:.2f}" for s in SUBS))
    print(f"   Microglia={cnt['Microglia']}  BAM={cnt['BAM']}  MDM={cnt['MDM']}  "
          f"ambiguous={int((lab=='ambiguous').sum())}  low_cov={int((lab=='low_markers_coverage').sum())}  "
          f"unresolved={int((lab=='unresolved').sum())}")

    pd.DataFrame({"cx": cxk[pool], "cy": cyk[pool], "subtype": lab}).to_csv(
        f"D:/thesis-research/score_genes_slice{n}_v2/myeloid_stage2_fdr_labels.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    ax.scatter(cxk, cyk, s=0.5, c="#eee", linewidths=0, rasterized=True)
    if tumor.sum():
        ax.scatter(cx[tumor], cy[tumor], s=1.2, c="black", linewidths=0, rasterized=True,
                   label=f"tumor ({int(tumor.sum()):,})")
    for s in SUBS:
        m = lab == s
        if m.any():
            ax.scatter(cxk[pool][m], cyk[pool][m], s=5, c=COL[s], linewidths=0,
                       rasterized=True, label=f"{s} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {n}{role} myeloid subtypes — FDR+margin+coverage ({POOL_MODE} pool)")
    ax.legend(loc="lower right", markerscale=3, fontsize=8)
    fig.savefig(f"D:/thesis-research/score_genes_slice{n}_v2/myeloid_stage2_fdr.png",
                bbox_inches="tight")
    plt.close(fig)
    return {"slice": n, "pool": len(pidx), **cnt}


def main():
    print(f"POOL_MODE = {POOL_MODE}  | coverage k = {KCOV}")
    rows = [run(n) for n in SLICES]
    pd.DataFrame(rows).to_csv(f"{ALLOUT}/myeloid_stage2_fdr_counts.csv", index=False)
    print(f"\nsaved -> {ALLOUT}/myeloid_stage2_fdr_counts.csv")


if __name__ == "__main__":
    main()
