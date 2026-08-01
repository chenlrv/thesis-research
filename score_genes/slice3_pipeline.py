"""Full slice-3 annotation pipeline, mirroring the slice-1 method:

  1. score_genes on 5 broad classes (Myeloid, Vascular, Astrocytes, Ependymal,
     Neurons) + FDR 0.05 / scaled-margin 1.5 gate          -> celltype, top_scaled
  2. high-conf training set = per-group top 20% by top_scaled   (Method 2)
  3. OvR-abstain classifier: 5 binary 'type vs rest' LogReg (balanced), predict
     all non-tumor cells; T=0.5 with abstention -> backbone labels (+ unknown)
  4. Myeloid sub-annotation: BAM (Mrc1/Cd163/Pf4 score + GMM split) then
     Microglia/MDM (score_genes + FDR/margin) on the non-BAM remainder
  5. spatial plot of BAM coloured by Lyve1
  6. combined 'slice3 annotation' map (backbone + myeloid subtypes + tumor black)

Output -> score_genes_slice3_merged/classify/
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
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)
from run_score_genes_merged import MARKERS as BROAD  # noqa: E402  (5 broad classes)

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"
OUT = "D:/thesis-research/score_genes_slice3_merged/classify"
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
FDR_CUTOFF, MARGIN_RATIO, TOPFRAC, T = 0.05, 1.5, 0.20, 0.5
BAM_MARKERS = ["Mrc1", "Cd163", "Pf4"]
SUB_MARKERS = {"Microglia": ["TMEM119", "Selplg"], "MDM": ["Ccr2", "Plac8"]}
COL = {"unknown": "#dddddd", "MDM": "#00a087", "Microglia": "#17becf",
       "Vascular": "#2ca02c", "Astrocytes": "#1f77b4", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "BAM": "#d62728"}
SEED = 0
sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gmm_high_component(score):
    s = np.asarray(score, float).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(s)
    hi = int(np.argmax(gm.means_.ravel()))
    is_high = gm.predict(s) == hi
    return is_high, float(s[is_high].min())


def score_genes(adata, genes, name):
    g = [x for x in genes if x in adata.var_names]
    sc.tl.score_genes(adata, gene_list=g, score_name=name, ctrl_size=50, n_bins=25)


def main():
    os.makedirs(OUT, exist_ok=True)
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
    tx, ty = cx[tumor], cy[tumor]
    nt = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]
    print(f"slice 3: {adata.n_obs:,} total, {nt.n_obs:,} non-tumor, "
          f"{int(tumor.sum()):,} tumor")

    # ---- 1) broad score_genes + gate ----
    for lab in GROUPS:
        score_genes(nt, BROAD[lab], "score_" + lab)
    S = nt.obs[["score_" + l for l in GROUPS]].copy()
    S.columns = GROUPS
    thr = {l: mirrored_fdr_threshold(S[l].to_numpy(), fdr=FDR_CUTOFF)[0] for l in GROUPS}
    res = scaled_margin_calls(S, thr, ratio=MARGIN_RATIO)
    celltype = np.asarray(res["calls"])
    top_scaled = res["top_scaled"]
    print("\ngated calls:")
    print(pd.Series(celltype).value_counts().to_string())

    # ---- 2) Method-2 high-conf training set (per-group top 20%) ----
    train_mask = np.zeros(nt.n_obs, bool)
    for g in GROUPS:
        gm = celltype == g
        t = np.quantile(top_scaled[gm], 1 - TOPFRAC)
        train_mask |= gm & (top_scaled >= t)
    print(f"\nMethod-2 training cells: {int(train_mask.sum()):,}")
    print({g: int((train_mask & (celltype == g)).sum()) for g in GROUPS})

    # ---- 3) OvR-abstain over all non-tumor cells ----
    Xnt = nt.X
    Xnt = (Xnt.toarray() if sp.issparse(Xnt) else np.asarray(Xnt)).astype(np.float32)
    Xtr, ytr = Xnt[train_mask], celltype[train_mask]
    P = np.zeros((nt.n_obs, len(GROUPS)), np.float32)
    for j, g in enumerate(GROUPS):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced",
                                               C=1.0, random_state=SEED))
        clf.fit(Xtr, (ytr == g).astype(int))
        P[:, j] = clf.predict_proba(Xnt)[:, 1]
    passes = P >= T
    npass = passes.sum(1)
    final = np.full(nt.n_obs, "unknown", dtype=object)
    one = npass == 1
    final[one] = np.array(GROUPS)[P.argmax(1)[one]]
    final[train_mask] = celltype[train_mask]      # keep gold for training cells
    print("\nOvR-abstain backbone labels:")
    print(pd.Series(final).value_counts().to_string())
    pd.DataFrame({"x": cxnt, "y": cynt, "is_train": train_mask,
                  "final_label": final}).to_csv(
        f"{OUT}/ovr_nontumor_predictions.csv", index=False)

    # ---- 4) Myeloid sub-annotation ----
    mye_mask = final == "Myeloid"
    mye = nt[mye_mask].copy()
    mye_xy = np.c_[cxnt[mye_mask], cynt[mye_mask]]
    vasc_xy = np.c_[cxnt[final == "Vascular"], cynt[final == "Vascular"]]
    score_genes(mye, BAM_MARKERS, "bam_score")
    bam_score = mye.obs["bam_score"].to_numpy()
    is_bam, bthr = gmm_high_component(bam_score)
    dist_v = cKDTree(vasc_xy).query(mye_xy, k=1)[0]
    p = mannwhitneyu(dist_v[is_bam], dist_v[~is_bam], alternative="less")[1]
    print(f"\nBAM: {int(is_bam.sum()):,}/{mye.n_obs:,} (GMM thr={bthr:.3f}); "
          f"dist-to-vessel BAM={np.median(dist_v[is_bam]):.0f} vs "
          f"{np.median(dist_v[~is_bam]):.0f}px (p={p:.1e})")

    nonbam = mye[~is_bam].copy()
    for lab in SUB_MARKERS:
        score_genes(nonbam, SUB_MARKERS[lab], "score_" + lab)
    S2 = nonbam.obs[["score_" + l for l in SUB_MARKERS]].copy()
    S2.columns = list(SUB_MARKERS)
    thr2 = {l: mirrored_fdr_threshold(S2[l].to_numpy(), fdr=FDR_CUTOFF)[0] for l in SUB_MARKERS}
    calls2 = np.asarray(scaled_margin_calls(S2, thr2, ratio=MARGIN_RATIO)["calls"])
    subtype = np.empty(mye.n_obs, dtype=object)
    subtype[is_bam] = "BAM"
    subtype[~is_bam] = calls2
    print("\nMyeloid subtypes:")
    print(pd.Series(subtype).value_counts().to_string())

    lyve = mye[:, "Lyve1"].X if "Lyve1" in mye.var_names else None
    lyve = (lyve.toarray().ravel() if lyve is not None and sp.issparse(lyve)
            else (np.asarray(lyve).ravel() if lyve is not None else np.zeros(mye.n_obs)))
    pd.DataFrame({"x": mye_xy[:, 0], "y": mye_xy[:, 1], "subtype": subtype,
                  "bam_score": bam_score.round(4), "is_bam": is_bam,
                  "lyve1": lyve.round(4), "dist_to_vessel": dist_v.round(1)}).to_csv(
        f"{OUT}/ovr_myeloid_subtypes.csv", index=False)

    # ---- 5) BAM coloured by Lyve1 (spatial) ----
    bmask = subtype == "BAM"
    bx, by, bl = mye_xy[bmask, 0], mye_xy[bmask, 1], lyve[bmask]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    ax.scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
    ax.scatter(tx, ty, s=1.5, c="black", linewidths=0, rasterized=True,
               label=f"tumor ({len(tx):,})")
    o = np.argsort(bl)
    scat = ax.scatter(bx[o], by[o], s=8, c=bl[o], cmap="magma",
                      vmax=(np.quantile(bl, 0.99) if bl.size else 1) or 1,
                      linewidths=0, rasterized=True)
    fig.colorbar(scat, ax=ax, shrink=0.6, label="Lyve1 (log-norm)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Slice 3 — BAM coloured by Lyve1 (n={int(bmask.sum()):,})",
                 fontweight="bold")
    ax.legend(loc="lower right", markerscale=4, fontsize=9)
    fig.savefig(f"{OUT}/bam_lyve1_spatial.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 6) combined slice3 annotation map ----
    combined = np.array(final, dtype=object)        # backbone labels
    sub_idx = np.where(mye_mask)[0]
    for k, s in enumerate(subtype):
        combined[sub_idx[k]] = s if s != "unknown" else "unknown"
    counts = {g: int((combined == g).sum()) for g in COL}
    print("\nfinal annotation counts:", counts)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=180)
    m = combined == "unknown"
    ax.scatter(cxnt[m], cynt[m], s=1.2, c=COL["unknown"], linewidths=0,
               rasterized=True, label=f"unknown ({counts['unknown']:,})")
    for g in ["MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal", "Neurons"]:
        mm = combined == g
        ax.scatter(cxnt[mm], cynt[mm], s=1.6, c=COL[g], linewidths=0,
                   rasterized=True, label=f"{g} ({counts[g]:,})")
    ax.scatter(tx, ty, s=1.6, c="black", linewidths=0, rasterized=True,
               label=f"tumor ({len(tx):,})")
    mb = combined == "BAM"
    ax.scatter(cxnt[mb], cynt[mb], s=1.6, c=COL["BAM"], linewidths=0,
               rasterized=True, label=f"BAM ({counts['BAM']:,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("slice3 annotation", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", markerscale=5, fontsize=9, frameon=True)
    fig.savefig(f"{OUT}/slice3_annotation.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved ovr_nontumor_predictions.csv, ovr_myeloid_subtypes.csv, "
          f"bam_lyve1_spatial.png, slice3_annotation.png -> {OUT}")


if __name__ == "__main__":
    main()
