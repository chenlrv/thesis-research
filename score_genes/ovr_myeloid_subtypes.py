"""Apply the BAM and Microglia/MDM subtyping to the Myeloid cells called by the
OvR-abstain classifier (ovr_nontumor_predictions.csv, final_label == 'Myeloid').

Stage 1 - BAM: score_genes(Mrc1/Cd163/Pf4) -> 2-component GMM split -> is_bam,
              validated perivascular (distance to the OvR Vascular cells).
Stage 2 - Microglia vs MDM on the non-BAM remainder: score_genes(TMEM119/Selplg)
              and (Ccr2/Plac8) -> FDR 0.05 + scaled-margin 1.5 gate.

Output -> score_genes_slice1_merged/classify/ovr_myeloid_*.{png,csv}
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)

OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
BAM_MARKERS = ["Mrc1", "Cd163", "Pf4"]
MARKERS = {"Microglia": ["TMEM119", "Selplg"], "MDM": ["Ccr2", "Plac8"]}
AUDIT = ["Mrc1", "Cd163", "Pf4", "TMEM119", "Selplg", "Cx3cr1", "Ccr2", "Plac8", "Cd14"]
FDR_CUTOFF, MARGIN_RATIO = 0.05, 1.5
COL = {"BAM": "#d62728", "Microglia": "#17becf", "MDM": "#00a087", "unknown": "#cccccc"}
sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gmm_high_component(score):
    s = np.asarray(score, float).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(s)
    hi = int(np.argmax(gm.means_.ravel()))
    is_high = gm.predict(s) == hi
    return is_high, float(s[is_high].min())


def mean_expr(adata, mask, genes):
    genes = [g for g in genes if g in adata.var_names]
    M = adata[mask, genes].X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    return {g: float(M[:, i].mean()) for i, g in enumerate(genes)}


def main():
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]
    adata.obs["x"], adata.obs["y"] = cxnt, cynt

    ovr = pd.read_csv(OVR)
    assert np.allclose(ovr["x"].to_numpy(), cxnt, atol=1e-3)
    final = ovr["final_label"].to_numpy()
    mye_mask = final == "Myeloid"
    vasc_xy = np.c_[cxnt[final == "Vascular"], cynt[final == "Vascular"]]
    mye = adata[mye_mask].copy()
    mye_xy = np.c_[mye.obs["x"].to_numpy(), mye.obs["y"].to_numpy()]
    print(f"OvR Myeloid cells: {mye.n_obs:,}   (vessel scaffold = "
          f"{len(vasc_xy):,} OvR Vascular cells)")

    # ---- Stage 1: BAM score + GMM split ----
    bg = [g for g in BAM_MARKERS if g in mye.var_names]
    sc.tl.score_genes(mye, gene_list=bg, score_name="bam_score",
                      ctrl_size=50, n_bins=25)
    bam_score = mye.obs["bam_score"].to_numpy()
    is_bam, thr = gmm_high_component(bam_score)
    print(f"\nStage 1 BAM: {int(is_bam.sum()):,} / {mye.n_obs:,} "
          f"({100*is_bam.mean():.1f}%)  GMM threshold={thr:.3f}")

    # perivascular validation
    dist = cKDTree(vasc_xy).query(mye_xy, k=1)[0]
    p = mannwhitneyu(dist[is_bam], dist[~is_bam], alternative="less")[1]
    print(f"  median dist-to-vessel: BAM={np.median(dist[is_bam]):.1f}px  "
          f"other={np.median(dist[~is_bam]):.1f}px  (MWU p={p:.1e})")

    # ---- Stage 2: Microglia vs MDM on non-BAM ----
    nonbam = mye[~is_bam].copy()
    for label, genes in MARKERS.items():
        g = [x for x in genes if x in nonbam.var_names]
        sc.tl.score_genes(nonbam, gene_list=g, score_name="score_" + label,
                          ctrl_size=50, n_bins=25)
    S = nonbam.obs[["score_" + l for l in MARKERS]].copy()
    S.columns = list(MARKERS)
    thr2 = {l: mirrored_fdr_threshold(S[l].to_numpy(), fdr=FDR_CUTOFF)[0] for l in MARKERS}
    calls = scaled_margin_calls(S, thr2, ratio=MARGIN_RATIO)["calls"]
    print(f"\nStage 2 (non-BAM {nonbam.n_obs:,}): "
          + pd.Series(calls).value_counts().to_dict().__str__())

    # ---- combine subtype labels ----
    subtype = np.empty(mye.n_obs, dtype=object)
    subtype[is_bam] = "BAM"
    subtype[~is_bam] = np.asarray(calls)
    print("\n=== OvR Myeloid subtypes ===")
    print(pd.Series(subtype).value_counts().to_string())
    print(f"  (of {mye.n_obs:,}; "
          f"{100*np.mean(subtype!='unknown'):.0f}% subtyped)")

    # ---- marker sanity per subtype ----
    print("\nmean expression per subtype (BAM | Microglia | MDM | unknown):")
    em = {s: mean_expr(mye, subtype == s, AUDIT) for s in ["BAM", "Microglia", "MDM", "unknown"]}
    for g in [x for x in AUDIT if x in mye.var_names]:
        print(f"  {g:8s}: " + " | ".join(f"{em[s].get(g,0):5.2f}"
              for s in ["BAM", "Microglia", "MDM", "unknown"]))

    pd.DataFrame({"x": mye_xy[:, 0], "y": mye_xy[:, 1], "subtype": subtype,
                  "bam_score": bam_score.round(4), "is_bam": is_bam,
                  "dist_to_vessel": dist.round(1)}).to_csv(
        f"{OUT}/ovr_myeloid_subtypes.csv", index=False)

    # ================= figures =================
    fig, ax = plt.subplots(1, 2, figsize=(20, 9), dpi=160)
    # (a) spatial subtypes over all non-tumor
    ax[0].scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
    for s in ["unknown", "MDM", "Microglia", "BAM"]:
        m = subtype == s
        if m.any():
            ax[0].scatter(mye_xy[m, 0], mye_xy[m, 1], s=4, c=COL[s], linewidths=0,
                          rasterized=True, label=f"{s} ({int(m.sum()):,})")
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("OvR Myeloid subtypes (spatial)", fontweight="bold")
    ax[0].legend(loc="lower right", markerscale=4, fontsize=9)
    # (b) BAM score distribution + GMM threshold
    ax[1].hist(bam_score, bins=70, color="#999999")
    ax[1].axvline(thr, c="r", ls="--", lw=1.2, label=f"GMM threshold {thr:.2f}")
    ax[1].set_xlabel("BAM score (Mrc1/Cd163/Pf4)"); ax[1].set_ylabel("myeloid cells")
    ax[1].set_title(f"(b) BAM split — {int(is_bam.sum()):,} BAM", fontweight="bold")
    ax[1].legend(fontsize=9)
    fig.suptitle(f"OvR Myeloid subtyping (n={mye.n_obs:,})", fontsize=15, fontweight="bold")
    fig.savefig(f"{OUT}/ovr_myeloid_subtypes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved ovr_myeloid_subtypes.png, ovr_myeloid_subtypes.csv -> {OUT}")


if __name__ == "__main__":
    main()
