"""Compare the OvR-abstain backbone annotation on slice 3 using XGBoost vs the
LogReg result already saved (ovr_nontumor_predictions.csv).

Reuses the Method-2 training cells (is_train) and their gold labels from the
LogReg run, trains 5 binary 'type vs rest' XGBoost classifiers (scale_pos_weight
balanced), applies the same T=0.5 abstention, and compares counts / agreement /
CV macro-F1 + a side-by-side spatial map.

Output -> score_genes_slice3_merged/classify/xgb_*.png|csv
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
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, TUMOR_COL)

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"
OUT = "D:/thesis-research/score_genes_slice3_merged/classify"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
PANELS = GROUPS + ["unknown"]
COL = {"unknown": "#dddddd", "Myeloid": "#00a087", "Vascular": "#2ca02c",
       "Astrocytes": "#1f77b4", "Ependymal": "#984ea3", "Neurons": "#e377c2"}
T, SEED = 0.5, 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def xgb(spw=1.0):
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", scale_pos_weight=spw, n_jobs=1,
                         random_state=SEED)


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
    lr_final = ovr["final_label"].to_numpy().astype(object)
    ytr = lr_final[is_train]               # gold labels for the training cells
    Xtr = Xnt[is_train]
    print(f"non-tumor: {nt.n_obs:,}  training: {int(is_train.sum()):,}")

    # ---- CV macro-F1 (multiclass) LogReg vs XGBoost on the training set ----
    le = LabelEncoder().fit(GROUPS)
    yenc = le.transform(ytr)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    lr_mc = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced",
                                             random_state=SEED))
    f1_lr = f1_score(yenc, cross_val_predict(lr_mc, Xtr, yenc, cv=skf), average="macro")
    f1_xgb = f1_score(yenc, cross_val_predict(xgb(), Xtr, yenc, cv=skf), average="macro")
    print(f"\nCV macro-F1:  LogReg={f1_lr:.4f}   XGBoost={f1_xgb:.4f}")

    # ---- OvR XGBoost + abstention ----
    P = np.zeros((nt.n_obs, len(GROUPS)), np.float32)
    for j, g in enumerate(GROUPS):
        yb = (ytr == g).astype(int)
        spw = float((yb == 0).sum()) / max((yb == 1).sum(), 1)
        clf = xgb(spw=spw).fit(Xtr, yb)
        P[:, j] = clf.predict_proba(Xnt)[:, 1]
    npass = (P >= T).sum(1)
    xgb_final = np.full(nt.n_obs, "unknown", dtype=object)
    one = npass == 1
    xgb_final[one] = np.array(GROUPS)[P.argmax(1)[one]]
    xgb_final[is_train] = ytr

    # ---- compare ----
    comp = pd.DataFrame({"type": PANELS,
                         "LogReg": [int((lr_final == g).sum()) for g in PANELS],
                         "XGBoost": [int((xgb_final == g).sum()) for g in PANELS]})
    print("\ncounts (backbone, OvR-abstain T=0.5):")
    print(comp.to_string(index=False))
    both = (lr_final != "unknown") & (xgb_final != "unknown")
    agree = float((lr_final[both] == xgb_final[both]).mean())
    lr_only = int(((lr_final != "unknown") & (xgb_final == "unknown")).sum())
    xgb_only = int(((xgb_final != "unknown") & (lr_final == "unknown")).sum())
    print(f"\namong cells typed by BOTH ({int(both.sum()):,}): {100*agree:.1f}% same")
    print(f"LogReg-typed but XGB unknown: {lr_only:,}   "
          f"XGB-typed but LogReg unknown: {xgb_only:,}")
    comp.to_csv(f"{OUT}/xgb_vs_logreg_counts.csv", index=False)

    # ---- side-by-side spatial ----
    fig, ax = plt.subplots(1, 2, figsize=(24, 9), dpi=150)
    for a, (lab, name) in zip(ax, [(lr_final, f"LogReg (F1={f1_lr:.3f})"),
                                   (xgb_final, f"XGBoost (F1={f1_xgb:.3f})")]):
        for g in ["unknown"] + GROUPS:
            m = lab == g
            a.scatter(cxnt[m], cynt[m], s=1.4, c=COL[g], linewidths=0,
                      rasterized=True, label=f"{g} ({int(m.sum()):,})")
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(name, fontweight="bold")
        a.legend(loc="lower right", markerscale=5, fontsize=8)
    fig.suptitle("Slice 3 backbone OvR-abstain — LogReg vs XGBoost",
                 fontsize=15, fontweight="bold")
    fig.savefig(f"{OUT}/xgb_vs_logreg_spatial.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved xgb_vs_logreg_counts.csv, xgb_vs_logreg_spatial.png -> {OUT}")


if __name__ == "__main__":
    main()
