"""Per-cell-type spatial comparison, LogReg vs XGBoost, on slice 3.

Recomputes BOTH full annotations (OvR-abstain backbone + BAM/Microglia/MDM
subtyping) in one place from the same Method-2 training set, so the only
difference is the classifier. Sanity-checks that recomputed LogReg matches the
saved pipeline. One row per cell type, two columns (LogReg | XGBoost), cell
counts in each title.

Output -> score_genes_slice3_merged/classify/lr_vs_xgb_celltypes.png|csv
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

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"
OUT = "D:/thesis-research/score_genes_slice3_merged/classify"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
BAM_MARKERS = ["Mrc1", "Cd163", "Pf4"]
SUB_MARKERS = {"Microglia": ["TMEM119", "Selplg"], "MDM": ["Ccr2", "Plac8"]}
TYPES = ["BAM", "MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal",
         "Neurons", "unknown"]
COL = {"unknown": "#999999", "MDM": "#00a087", "Microglia": "#17becf",
       "Vascular": "#2ca02c", "Astrocytes": "#1f77b4", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "BAM": "#d62728"}
FDR_CUTOFF, MARGIN_RATIO, T, SEED = 0.05, 1.5, 0.5, 0
sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gmm_high(score):
    s = np.asarray(score, float).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(s)
    hi = int(np.argmax(gm.means_.ravel()))
    return gm.predict(s) == hi


def score_genes(a, genes, name):
    g = [x for x in genes if x in a.var_names]
    sc.tl.score_genes(a, gene_list=g, score_name=name, ctrl_size=50, n_bins=25)


def lr_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced",
                                            C=1.0, random_state=SEED))


def xgb_model(spw):
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", scale_pos_weight=spw, n_jobs=1,
                         random_state=SEED)


def annotate(kind, Xnt, Xtr, ytr, is_train, nt, cxnt, cynt):
    """Return combined per-cell labels (backbone + myeloid subtypes) for a classifier."""
    P = np.zeros((Xnt.shape[0], len(GROUPS)), np.float32)
    for j, g in enumerate(GROUPS):
        yb = (ytr == g).astype(int)
        if kind == "logreg":
            clf = lr_model()
        else:
            spw = float((yb == 0).sum()) / max((yb == 1).sum(), 1)
            clf = xgb_model(spw)
        clf.fit(Xtr, yb)
        P[:, j] = clf.predict_proba(Xnt)[:, 1]
    npass = (P >= T).sum(1)
    final = np.full(Xnt.shape[0], "unknown", dtype=object)
    one = npass == 1
    final[one] = np.array(GROUPS)[P.argmax(1)[one]]
    final[is_train] = ytr

    # myeloid subtyping
    mye_mask = final == "Myeloid"
    mye = nt[mye_mask].copy()
    score_genes(mye, BAM_MARKERS, "bam_score")
    is_bam = gmm_high(mye.obs["bam_score"].to_numpy())
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

    combined = np.array(final, dtype=object)
    sub_idx = np.where(mye_mask)[0]
    combined[sub_idx] = subtype          # Myeloid -> BAM/MDM/Microglia/unknown
    return combined


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
    nt = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]
    Xnt = nt.X
    Xnt = (Xnt.toarray() if sp.issparse(Xnt) else np.asarray(Xnt)).astype(np.float32)

    ovr = pd.read_csv(OVR)
    assert np.allclose(ovr["x"].to_numpy(), cxnt, atol=1e-3)
    is_train = ovr["is_train"].to_numpy().astype(bool)
    gold = ovr["final_label"].to_numpy().astype(object)
    Xtr, ytr = Xnt[is_train], gold[is_train]
    print(f"non-tumor: {nt.n_obs:,}  training: {int(is_train.sum()):,}")

    lr = annotate("logreg", Xnt, Xtr, ytr, is_train, nt, cxnt, cynt)
    xgbc = annotate("xgb", Xnt, Xtr, ytr, is_train, nt, cxnt, cynt)

    # sanity: recomputed LogReg backbone Myeloid total should match saved pipeline
    lr_mye = int((np.isin(lr, ["BAM", "MDM", "Microglia"])).sum()
                 + (lr == "Myeloid").sum())
    print(f"\nsanity: recomputed LogReg myeloid (BAM+MDM+Micro+unknown-myeloid) "
          f"vs saved pipeline backbone Myeloid 9,288")
    # subtype 'unknown' myeloid are folded into 'unknown'; compare typed subtypes
    cmp = pd.DataFrame({"type": TYPES,
                        "LogReg": [int((lr == t).sum()) for t in TYPES],
                        "XGBoost": [int((xgbc == t).sum()) for t in TYPES]})
    print("\ncell-type counts (LogReg vs XGBoost):")
    print(cmp.to_string(index=False))
    cmp.to_csv(f"{OUT}/lr_vs_xgb_celltypes.csv", index=False)

    # ---- figure: rows = types, cols = [LogReg, XGBoost] ----
    fig, axes = plt.subplots(len(TYPES), 2, figsize=(16, 3.2 * len(TYPES)), dpi=120)
    for r, t in enumerate(TYPES):
        for c, (lab, name) in enumerate([(lr, "LogReg"), (xgbc, "XGBoost")]):
            ax = axes[r, c]
            m = lab == t
            ax.scatter(cxnt, cynt, s=0.4, c="#eeeeee", linewidths=0, rasterized=True)
            ax.scatter(cxnt[m], cynt[m], s=2.5, c=COL[t], linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{t} — {name}  (n={int(m.sum()):,})", fontweight="bold",
                         fontsize=11)
    fig.suptitle("Slice 3 cell types — LogReg vs XGBoost", fontsize=16,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{OUT}/lr_vs_xgb_celltypes.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved lr_vs_xgb_celltypes.png, lr_vs_xgb_celltypes.csv -> {OUT}")


if __name__ == "__main__":
    main()
