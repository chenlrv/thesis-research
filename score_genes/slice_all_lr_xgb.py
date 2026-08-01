"""Per-cell-type spatial comparison, LogReg vs XGBoost, for EVERY slice (1-6).

Self-contained per slice: score_genes broad 5 classes + FDR/margin gate ->
Method-2 top-20% training set -> OvR-abstain backbone (LogReg & XGBoost) ->
BAM/Microglia/MDM subtyping. One figure per slice (rows = cell types, cols =
LogReg | XGBoost, counts in titles) + a combined counts CSV.

Output -> score_genes_slice{N}_merged/classify/lr_vs_xgb_celltypes.png
          score_genes_slice_all/lr_vs_xgb_celltype_counts.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)
from run_score_genes_merged import MARKERS as BROAD  # noqa: E402

SLICES = [1, 4, 5, 6]
SLICE_TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ALLOUT = "D:/thesis-research/score_genes_slice_all"
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
BAM_MARKERS = ["Mrc1", "Cd163", "Pf4"]
SUB_MARKERS = {"Microglia": ["TMEM119", "Selplg"], "MDM": ["Ccr2", "Plac8"]}
TYPES = ["BAM", "MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal",
         "Neurons", "unknown"]
COL = {"unknown": "#999999", "MDM": "#00a087", "Microglia": "#17becf",
       "Vascular": "#2ca02c", "Astrocytes": "#1f77b4", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "BAM": "#d62728"}
FDR_CUTOFF, MARGIN_RATIO, TOPFRAC, T, SEED = 0.05, 1.5, 0.20, 0.5, 0
sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def score_genes(a, genes, name):
    g = [x for x in genes if x in a.var_names]
    if g:
        sc.tl.score_genes(a, gene_list=g, score_name=name, ctrl_size=50, n_bins=25)
    else:
        a.obs[name] = 0.0


def gmm_high(score):
    s = np.asarray(score, float).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(s)
    hi = int(np.argmax(gm.means_.ravel()))
    return gm.predict(s) == hi


def annotate(kind, Xnt, Xtr, ytr, is_train, gold, nt):
    P = np.zeros((Xnt.shape[0], len(GROUPS)), np.float32)
    for j, g in enumerate(GROUPS):
        yb = (ytr == g).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        if kind == "logreg":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, class_weight="balanced",
                                                   C=1.0, random_state=SEED))
        else:
            spw = float((yb == 0).sum()) / max((yb == 1).sum(), 1)
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                eval_metric="logloss", scale_pos_weight=spw, n_jobs=1,
                                random_state=SEED)
        clf.fit(Xtr, yb)
        P[:, j] = clf.predict_proba(Xnt)[:, 1]
    npass = (P >= T).sum(1)
    final = np.full(Xnt.shape[0], "unknown", dtype=object)
    one = npass == 1
    final[one] = np.array(GROUPS)[P.argmax(1)[one]]
    final[is_train] = gold[is_train]

    mye_mask = final == "Myeloid"
    combined = np.array(final, dtype=object)
    if mye_mask.sum() >= 10:
        mye = nt[mye_mask].copy()
        score_genes(mye, BAM_MARKERS, "bam_score")
        is_bam = gmm_high(mye.obs["bam_score"].to_numpy())
        subtype = np.empty(mye.n_obs, dtype=object)
        subtype[is_bam] = "BAM"
        nonbam = mye[~is_bam]
        if nonbam.n_obs > 0:
            nb = nonbam.copy()
            for lab in SUB_MARKERS:
                score_genes(nb, SUB_MARKERS[lab], "score_" + lab)
            S2 = nb.obs[["score_" + l for l in SUB_MARKERS]].copy()
            S2.columns = list(SUB_MARKERS)
            thr2 = {l: mirrored_fdr_threshold(S2[l].to_numpy(), fdr=FDR_CUTOFF)[0]
                    for l in SUB_MARKERS}
            subtype[~is_bam] = np.asarray(
                scaled_margin_calls(S2, thr2, ratio=MARGIN_RATIO)["calls"])
        combined[np.where(mye_mask)[0]] = subtype
    else:
        combined[mye_mask] = "unknown"
    return combined


def run_slice(n):
    path = SLICE_TMPL.format(n)
    out = f"D:/thesis-research/score_genes_slice{n}_merged/classify"
    os.makedirs(out, exist_ok=True)
    with h5py.File(path, "r") as h5:
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
    nt = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]

    # broad score_genes + gate
    for lab in GROUPS:
        score_genes(nt, BROAD[lab], "score_" + lab)
    S = nt.obs[["score_" + l for l in GROUPS]].copy()
    S.columns = GROUPS
    thr = {l: mirrored_fdr_threshold(S[l].to_numpy(), fdr=FDR_CUTOFF)[0] for l in GROUPS}
    res = scaled_margin_calls(S, thr, ratio=MARGIN_RATIO)
    celltype = np.asarray(res["calls"])
    top_scaled = res["top_scaled"]

    # Method-2 top 20% per group
    train_mask = np.zeros(nt.n_obs, bool)
    for g in GROUPS:
        gm = celltype == g
        if gm.sum() == 0:
            continue
        t = np.quantile(top_scaled[gm], 1 - TOPFRAC)
        train_mask |= gm & (top_scaled >= t)

    Xnt = nt.X
    Xnt = (Xnt.toarray() if sp.issparse(Xnt) else np.asarray(Xnt)).astype(np.float32)
    Xtr = Xnt[train_mask]
    gold = celltype.copy()

    lr = annotate("logreg", Xnt, Xtr, celltype[train_mask], train_mask, gold, nt)
    xgbc = annotate("xgb", Xnt, Xtr, celltype[train_mask], train_mask, gold, nt)

    cmp = pd.DataFrame({"type": TYPES,
                        "LogReg": [int((lr == t).sum()) for t in TYPES],
                        "XGBoost": [int((xgbc == t).sum()) for t in TYPES]})
    cmp.insert(0, "slice", n)
    cmp.to_csv(f"{out}/lr_vs_xgb_celltypes.csv", index=False)

    fig, axes = plt.subplots(len(TYPES), 2, figsize=(16, 3.0 * len(TYPES)), dpi=110)
    for r, ty in enumerate(TYPES):
        for c, (lab, name) in enumerate([(lr, "LogReg"), (xgbc, "XGBoost")]):
            ax = axes[r, c]
            m = lab == ty
            ax.scatter(cxnt, cynt, s=0.4, c="#eeeeee", linewidths=0, rasterized=True)
            ax.scatter(cxnt[m], cynt[m], s=2.5, c=COL[ty], linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{ty} — {name}  (n={int(m.sum()):,})", fontweight="bold",
                         fontsize=11)
    fig.suptitle(f"Slice {n} cell types — LogReg vs XGBoost (non-tumor "
                 f"{nt.n_obs:,})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{out}/lr_vs_xgb_celltypes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"slice {n}: non-tumor {nt.n_obs:,}, train {int(train_mask.sum()):,}, "
          f"tumor {int(tumor.sum()):,} -> {out}/lr_vs_xgb_celltypes.png")
    return cmp


def main():
    os.makedirs(ALLOUT, exist_ok=True)
    all_cmp = []
    for n in SLICES:
        try:
            all_cmp.append(run_slice(n))
        except Exception as e:
            print(f"slice {n} FAILED: {e}")
    if all_cmp:
        pd.concat(all_cmp).to_csv(f"{ALLOUT}/lr_vs_xgb_celltype_counts.csv", index=False)
        print(f"\nsaved combined counts -> {ALLOUT}/lr_vs_xgb_celltype_counts.csv")


if __name__ == "__main__":
    main()
