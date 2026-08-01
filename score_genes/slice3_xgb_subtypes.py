"""Myeloid subtyping (BAM / Microglia / MDM) on the XGBoost OvR-abstain Myeloid
cells for slice 3, with per-subtype spatial maps and a count comparison vs the
LogReg subtyping (ovr_myeloid_subtypes.csv).

Output -> score_genes_slice3_merged/classify/xgb_myeloid_*.png|csv
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
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"
OUT = "D:/thesis-research/score_genes_slice3_merged/classify"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
LR_SUB = f"{OUT}/ovr_myeloid_subtypes.csv"
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
BAM_MARKERS = ["Mrc1", "Cd163", "Pf4"]
SUB_MARKERS = {"Microglia": ["TMEM119", "Selplg"], "MDM": ["Ccr2", "Plac8"]}
PANELS = ["BAM", "MDM", "Microglia", "unknown"]
COL = {"BAM": "#d62728", "MDM": "#00a087", "Microglia": "#17becf", "unknown": "#999999"}
FDR_CUTOFF, MARGIN_RATIO, T, SEED = 0.05, 1.5, 0.5, 0
sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gmm_high(score):
    s = np.asarray(score, float).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(s)
    hi = int(np.argmax(gm.means_.ravel()))
    m = gm.predict(s) == hi
    return m, float(s[m].min())


def score_genes(a, genes, name):
    g = [x for x in genes if x in a.var_names]
    sc.tl.score_genes(a, gene_list=g, score_name=name, ctrl_size=50, n_bins=25)


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

    # ---- XGBoost OvR-abstain backbone ----
    P = np.zeros((nt.n_obs, len(GROUPS)), np.float32)
    for j, g in enumerate(GROUPS):
        yb = (ytr == g).astype(int)
        spw = float((yb == 0).sum()) / max((yb == 1).sum(), 1)
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            eval_metric="logloss", scale_pos_weight=spw, n_jobs=1,
                            random_state=SEED).fit(Xtr, yb)
        P[:, j] = clf.predict_proba(Xnt)[:, 1]
    npass = (P >= T).sum(1)
    final = np.full(nt.n_obs, "unknown", dtype=object)
    one = npass == 1
    final[one] = np.array(GROUPS)[P.argmax(1)[one]]
    final[is_train] = ytr
    mye_mask = final == "Myeloid"
    print(f"XGBoost Myeloid: {int(mye_mask.sum()):,}")

    # ---- subtype: BAM + Microglia/MDM ----
    mye = nt[mye_mask].copy()
    mye_xy = np.c_[cxnt[mye_mask], cynt[mye_mask]]
    score_genes(mye, BAM_MARKERS, "bam_score")
    is_bam, bthr = gmm_high(mye.obs["bam_score"].to_numpy())
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

    print(f"\nXGBoost Myeloid subtypes (GMM thr={bthr:.3f}):")
    print(pd.Series(subtype).value_counts().to_string())

    # comparison vs LogReg subtyping
    lr = pd.read_csv(LR_SUB)
    cmp = pd.DataFrame({"subtype": PANELS,
        "LogReg": [int((lr["subtype"] == s).sum()) for s in PANELS],
        "XGBoost": [int((subtype == s).sum()) for s in PANELS]})
    print("\nsubtype counts: LogReg vs XGBoost")
    print(cmp.to_string(index=False))
    pd.DataFrame({"x": mye_xy[:, 0], "y": mye_xy[:, 1], "subtype": subtype}).to_csv(
        f"{OUT}/xgb_myeloid_subtypes.csv", index=False)

    # ---- per-subtype spatial ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=150)
    for ax, g in zip(axes.ravel(), PANELS):
        m = subtype == g
        ax.scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
        ax.scatter(mye_xy[m, 0], mye_xy[m, 1], s=4, c=COL[g], linewidths=0,
                   rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (n={int(m.sum()):,})", fontweight="bold")
    fig.suptitle("Slice 3 — XGBoost Myeloid subtypes (per subtype)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/xgb_myeloid_per_subtype.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved xgb_myeloid_per_subtype.png, xgb_myeloid_subtypes.csv -> {OUT}")


if __name__ == "__main__":
    main()
