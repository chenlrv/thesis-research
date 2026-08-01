"""Contrastive myeloid subtyping (Microglia/MDM/BAM), slices 1,3,5, both scoring
methods (A=positive+margin, B=POS-NEG+margin) and both subtype classifiers
(LogReg, XGBoost).

Per slice:
  broad backbone (gate -> Method-2 -> OvR-abstain LogReg) -> Myeloid cells.
  for METHOD in {A, B}:
    score_genes contrastive subtype scores -> FDR+margin gate -> subtype calls
    top-20% per subtype = anchors
    for CLF in {LogReg, XGBoost}:
      OvR-abstain subtype classifier on anchors -> propagate to all myeloid cells

Outputs: per-slice 2x2 spatial figure + master counts/CV/marker-purity CSV.
Disease-aware sets (review-finalised): BAM defined by Lyve1/Pf4/Maf/Cd5l (NOT
Mrc1/Cd163, shared with activated microglia); activation genes excluded.
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
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)
from run_score_genes_merged import MARKERS as BROAD  # noqa: E402

SLICES = [1, 3, 5]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ALLOUT = "D:/thesis-research/score_genes_slice_all"
BACKBONE = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
SUBS = ["Microglia", "MDM", "BAM"]
POS = {
    "Microglia": ["TMEM119", "Adgrg1", "Hpgds", "Gpr183", "Selplg"],
    "BAM":       ["Mrc1", "Cd163", "Pf4", "Maf", "Cd5l"],   # Mrc1/Cd163 added for sensitivity
    "MDM":       ["Ccr2", "Plac8", "S100a8", "S100a9", "Vcan", "Sell", "Fn1",
                  "Crip1", "Clec12a", "Fpr1", "Ccr1"],
}
# NEG uses disease-STABLE competitor markers only: Mrc1/Cd163 are excluded from
# microglia's penalty (activated microglia legitimately express them).
BAM_STABLE = ["Pf4", "Maf", "Cd5l"]
NEG = {
    "Microglia": BAM_STABLE + POS["MDM"],
    "BAM":       POS["Microglia"] + POS["MDM"],
    "MDM":       POS["Microglia"] + BAM_STABLE,
}
COL = {"Microglia": "#17becf", "MDM": "#00a087", "BAM": "#d62728", "unknown": "#cccccc"}
FDR, RATIO, TOPFRAC, T, SEED = 0.05, 1.5, 0.20, 0.5, 0
sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def sg(adata, genes, name):
    g = [x for x in genes if x in adata.var_names]
    sc.tl.score_genes(adata, gene_list=g, score_name=name, ctrl_size=50, n_bins=25)
    return adata.obs[name].to_numpy()


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", C=1.0, random_state=SEED))


def xgb(spw=1.0):
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", scale_pos_weight=spw, n_jobs=1,
                         random_state=SEED)


def proba1(clf, X, chunk=40000):
    out = np.empty(X.shape[0], np.float32)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = clf.predict_proba(X[i:i + chunk])[:, 1]
    return out


def ovr_abstain(Xall, Xtr, ytr, kind, classes):
    P = np.zeros((Xall.shape[0], len(classes)), np.float32)
    for j, g in enumerate(classes):
        yb = (ytr == g).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        clf = lr() if kind == "logreg" else xgb(float((yb == 0).sum()) / max(yb.sum(), 1))
        clf.fit(Xtr, yb)
        P[:, j] = proba1(clf, Xall)              # chunked -> avoids huge float64 copy
    npass = (P >= T).sum(1)
    out = np.full(Xall.shape[0], "unknown", dtype=object)
    one = npass == 1
    out[one] = np.array(classes)[P.argmax(1)[one]]
    return out


def broad_myeloid(nt, Xnt):
    """Broad gate -> Method-2 -> OvR-abstain LogReg -> Myeloid boolean mask."""
    for lab in BACKBONE:
        sg(nt, BROAD[lab], "bb_" + lab)
    S = nt.obs[["bb_" + l for l in BACKBONE]].copy(); S.columns = BACKBONE
    thr = {l: mirrored_fdr_threshold(S[l].to_numpy(), fdr=FDR)[0] for l in BACKBONE}
    res = scaled_margin_calls(S, thr, ratio=RATIO)
    ct, top = np.asarray(res["calls"]), res["top_scaled"]
    tr = np.zeros(nt.n_obs, bool)
    for g in BACKBONE:
        m = ct == g
        if m.sum():
            tr |= m & (top >= np.quantile(top[m], 1 - TOPFRAC))
    final = ovr_abstain(Xnt, Xnt[tr], ct[tr], "logreg", BACKBONE)
    final[tr] = ct[tr]
    return final == "Myeloid"


def subtype_scores(mye, method):
    cols = {}
    for s in SUBS:
        pos = sg(mye, POS[s], f"pos_{s}")
        cols[s] = pos if method == "A" else pos - sg(mye, NEG[s], f"neg_{s}")
    S = pd.DataFrame(cols)[SUBS]
    thr = {s: mirrored_fdr_threshold(S[s].to_numpy(), fdr=FDR)[0] for s in SUBS}
    res = scaled_margin_calls(S, thr, ratio=RATIO)
    return np.asarray(res["calls"]), res["top_scaled"]


def run_slice(n):
    out = f"D:/thesis-research/score_genes_slice{n}_merged/classify"
    os.makedirs(out, exist_ok=True)
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _read_X(h5); var = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X); adata.var_names = pd.Index(var); adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    nt = adata[~tumor].copy(); cxnt, cynt = cx[~tumor], cy[~tumor]

    mask = broad_myeloid(nt, _dense(nt.X))
    mye = nt[mask].copy()
    mxy = np.c_[cxnt[mask], cynt[mask]]
    Xmye = _dense(mye.X)
    print(f"slice {n}: non-tumor {nt.n_obs:,}, Myeloid {mye.n_obs:,}")

    le = LabelEncoder().fit(SUBS)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, panels = [], {}
    for method in ["A", "B"]:
        calls, top = subtype_scores(mye, method)
        anc = np.zeros(mye.n_obs, bool)
        for s in SUBS:
            m = calls == s
            if m.sum() >= 5:
                anc |= m & (top >= np.quantile(top[m], 1 - TOPFRAC))
        ytr = calls[anc]
        Xtr = Xmye[anc]
        for kind in ["logreg", "xgboost"]:
            final = ovr_abstain(Xmye, Xtr, ytr, kind, SUBS)
            final[anc] = ytr
            # CV macro-F1 on anchors (contiguous labels for present classes;
            # n_splits capped by smallest class so XGBoost never sees a gap)
            present = sorted(set(ytr))
            if len(present) >= 2:
                le2 = LabelEncoder().fit(ytr)
                ye = le2.transform(ytr)
                nsp = max(2, min(5, int(np.bincount(ye).min())))
                skf2 = StratifiedKFold(nsp, shuffle=True, random_state=SEED)
                cvm = lr() if kind == "logreg" else xgb()
                cv = f1_score(ye, cross_val_predict(cvm, Xtr, ye, cv=skf2),
                              average="macro")
            else:
                cv = np.nan
            cnt = {s: int((final == s).sum()) for s in SUBS + ["unknown"]}
            # marker purity of BAM (should be high Lyve1/Pf4, low TMEM119/Adgrg1)
            bm = final == "BAM"
            pur = {g: round(float(_dense(mye[:, g].X)[bm].mean()), 2) if g in mye.var_names and bm.sum() else np.nan
                   for g in ["Lyve1", "Pf4", "TMEM119", "Adgrg1"]}
            rows.append({"slice": n, "method": method, "clf": kind,
                         "anchors": int(anc.sum()), "cv_macroF1": round(cv, 3),
                         **cnt, **{f"BAM_{k}": v for k, v in pur.items()}})
            panels[(method, kind)] = final

    pd.DataFrame(rows).to_csv(f"{out}/myeloid_subtype_contrastive.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    # 2x2 spatial figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=140)
    for ax, (method, kind) in zip(axes.ravel(),
                                  [("A", "logreg"), ("A", "xgboost"),
                                   ("B", "logreg"), ("B", "xgboost")]):
        lab = panels[(method, kind)]
        ax.scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
        for s in ["unknown"] + SUBS:
            m = lab == s
            if m.any():
                ax.scatter(mxy[m, 0], mxy[m, 1], s=4, c=COL[s], linewidths=0,
                           rasterized=True, label=f"{s} ({int(m.sum())})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"method {method} — {kind}", fontweight="bold")
        ax.legend(loc="lower right", markerscale=3, fontsize=7)
    fig.suptitle(f"Slice {n} myeloid subtypes — A/B x LogReg/XGBoost",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out}/myeloid_subtype_contrastive.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def _dense(X):
    return (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)


def main():
    os.makedirs(ALLOUT, exist_ok=True)
    allr = []
    for n in SLICES:
        try:
            allr.append(run_slice(n))
        except Exception as e:
            print(f"slice {n} FAILED: {e}")
    if allr:
        pd.concat(allr).to_csv(f"{ALLOUT}/myeloid_subtype_contrastive_counts.csv", index=False)
        print(f"\nsaved combined -> {ALLOUT}/myeloid_subtype_contrastive_counts.csv")


if __name__ == "__main__":
    main()
