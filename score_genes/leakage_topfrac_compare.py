"""Does a more restricted training set reduce tumor leakage?

Train the backbone LogReg on the per-group top-q% of the gated cells for q in
{10%, 20%}, classify all non-tumor cells, and use the 4,248 tumor cells (never
seen) as a validation set for leakage. Reports, side by side: training size, CV
macro-F1, tumor leak rate vs threshold, and where tumor cells land.

Output -> score_genes_slice1_merged/classify/leakage_topfrac_compare.(png|csv)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
FRACS = [0.10, 0.20]
THRS = [0.5, 0.7, 0.9]
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2"}
SEED = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


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
    Xall = adata.X
    Xall = (Xall.toarray() if sp.issparse(Xall) else np.asarray(Xall)).astype(np.float32)
    Xnt, Xtu = Xall[~tumor], Xall[tumor]
    cxnt, cynt = cx[~tumor], cy[~tumor]
    xtu, ytu = cx[tumor], cy[tumor]

    df = pd.read_csv(SCORES_CSV)
    assert np.allclose(df["x"].to_numpy(), cxnt, atol=1e-3)
    lab_all = df["celltype"].to_numpy()
    score_all = df["top_scaled"].to_numpy()
    print(f"non-tumor: {Xnt.shape[0]:,}   tumor (validation): {Xtu.shape[0]:,}")

    le = LabelEncoder().fit(GROUPS)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, summaries, tu_probas = [], [], {}
    for q in FRACS:
        train_mask = np.zeros(Xnt.shape[0], bool)
        for g in GROUPS:
            gm = lab_all == g
            t = np.quantile(score_all[gm], 1 - q)
            train_mask |= gm & (score_all >= t)
        ytr = lab_all[train_mask]
        ytr_enc = le.transform(ytr)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               C=1.0, random_state=SEED))

        cvpred = cross_val_predict(clf, Xnt[train_mask], ytr_enc, cv=skf, n_jobs=1)
        cv_f1 = f1_score(ytr_enc, cvpred, average="macro")

        clf.fit(Xnt[train_mask], ytr_enc)
        nt_max = clf.predict_proba(Xnt[~train_mask]).max(1)
        ptu = clf.predict_proba(Xtu)
        tu_lab = le.inverse_transform(ptu.argmax(1))
        tu_max = ptu.max(1)
        tu_probas[q] = tu_max

        per_group = {g: int((ytr == g).sum()) for g in GROUPS}
        print(f"\n===== top {q:.0%}  (train n={int(train_mask.sum()):,}, "
              f"CV macro-F1={cv_f1:.4f}) =====")
        print("  train per group:", per_group)
        for thr in THRS:
            leak = float((tu_max >= thr).mean())
            ntc = float((nt_max >= thr).mean())
            print(f"  thr={thr}: tumor leaked={100*leak:5.1f}%   "
                  f"non-tumor confident={100*ntc:5.1f}%")
            rows.append({"top_frac": f"{q:.0%}", "threshold": thr,
                         "tumor_leaked_%": round(100 * leak, 1),
                         "nontumor_confident_%": round(100 * ntc, 1)})
        leak05 = tu_max >= 0.5
        bd = pd.Series(tu_lab[leak05]).value_counts().reindex(GROUPS).fillna(0).astype(int)
        print(f"  leaked@0.5 = {int(leak05.sum()):,}/{len(tu_max):,}; by type:",
              bd.to_dict())
        summaries.append({"top_frac": f"{q:.0%}", "train_n": int(train_mask.sum()),
                          "cv_macro_f1": round(cv_f1, 4),
                          "tumor_leaked@0.5": int(leak05.sum()),
                          **{f"leak_{g}": int(bd[g]) for g in GROUPS}})

    pd.DataFrame(rows).to_csv(f"{OUT}/leakage_topfrac_sweep.csv", index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(f"{OUT}/leakage_topfrac_summary.csv", index=False)
    print("\n=== summary ===")
    print(summ.to_string(index=False))

    # ---- figure: leak vs threshold (10% vs 20%) + tumor proba distributions ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    sweep = pd.DataFrame(rows)
    for q in FRACS:
        s = sweep[sweep["top_frac"] == f"{q:.0%}"]
        ax[0].plot(s["threshold"], s["tumor_leaked_%"], "-o", label=f"top {q:.0%}")
    ax[0].set_xlabel("confidence threshold"); ax[0].set_ylabel("tumor cells leaked (%)")
    ax[0].set_title("tumor leakage vs threshold", fontweight="bold")
    ax[0].set_ylim(0, 100); ax[0].legend(); ax[0].grid(alpha=0.3)
    for q in FRACS:
        ax[1].hist(tu_probas[q], bins=50, density=True, alpha=0.55,
                   label=f"top {q:.0%}")
    ax[1].axvline(0.5, c="k", ls="--", lw=1)
    ax[1].set_xlabel("max class probability (tumor cells)")
    ax[1].set_ylabel("density")
    ax[1].set_title("tumor-cell confidence distribution", fontweight="bold")
    ax[1].legend()
    fig.suptitle("Restricted (top 10%) vs (top 20%) training — tumor leakage",
                 fontweight="bold")
    fig.savefig(f"{OUT}/leakage_topfrac_compare.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved leakage_topfrac_compare.png, leakage_topfrac_sweep.csv, "
          f"leakage_topfrac_summary.csv -> {OUT}")


if __name__ == "__main__":
    main()
