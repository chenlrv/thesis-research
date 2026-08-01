"""Leakage stress-test: run the backbone LogReg (trained on the Method-2 non-tumor
high-conf set) on the 4,248 TUMOR-annotated cells it never saw, and measure how
many it confidently mislabels as a backbone type.

A 5-way softmax has no 'none' option, so it must assign every tumor cell to some
backbone class; the only guard is the confidence threshold. We report, at several
thresholds, what fraction of tumor cells stay 'uncertain' vs leak into a type,
and compare to the non-tumor uncertain rate.

Output -> score_genes_slice1_merged/classify/tumor_leakage.(png|csv)
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
TOPFRAC = 0.20
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
    sc.pp.normalize_total(adata, target_sum=1e4)   # per-cell; consistent across split
    sc.pp.log1p(adata)

    Xall = adata.X
    Xall = Xall.toarray() if sp.issparse(Xall) else np.asarray(Xall)
    Xall = Xall.astype(np.float32)

    Xnt, Xtu = Xall[~tumor], Xall[tumor]
    cxnt, cynt = cx[~tumor], cy[~tumor]
    xtu, ytu = cx[tumor], cy[tumor]
    print(f"non-tumor: {Xnt.shape[0]:,}   tumor: {Xtu.shape[0]:,}")

    # cell_scores.csv aligns to the non-tumor order
    df = pd.read_csv(SCORES_CSV)
    assert np.allclose(df["x"].to_numpy(), cxnt, atol=1e-3) \
        and np.allclose(df["y"].to_numpy(), cynt, atol=1e-3), "score rows misaligned"
    lab_all = df["celltype"].to_numpy()
    score_all = df["top_scaled"].to_numpy()

    # Method-2 training mask: per-group top 20% of the gated cells
    train_mask = np.zeros(Xnt.shape[0], bool)
    for g in GROUPS:
        gm = lab_all == g
        t = np.quantile(score_all[gm], 1 - TOPFRAC)
        train_mask |= gm & (score_all >= t)
    print(f"Method-2 training cells: {int(train_mask.sum()):,}")

    le = LabelEncoder().fit(GROUPS)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           C=1.0, random_state=SEED))
    clf.fit(Xnt[train_mask], le.transform(lab_all[train_mask]))

    # ---- predict the held-out non-tumor cells (reference) and the tumor cells ----
    def predict(Xm):
        p = clf.predict_proba(Xm)
        return le.inverse_transform(p.argmax(1)), p.max(1)

    nt_lab, nt_max = predict(Xnt[~train_mask])
    tu_lab, tu_max = predict(Xtu)

    # ---- threshold sweep: confident (leaked) vs uncertain ----
    print("\nfraction CONFIDENT (>= thr -> leaks into a backbone type):")
    rows = []
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tu_conf = float((tu_max >= thr).mean())
        nt_conf = float((nt_max >= thr).mean())
        rows.append({"threshold": thr,
                     "tumor_confident_%": round(100 * tu_conf, 1),
                     "tumor_uncertain_%": round(100 * (1 - tu_conf), 1),
                     "nontumor_confident_%": round(100 * nt_conf, 1)})
        print(f"  thr={thr}:  tumor leaked={100*tu_conf:5.1f}%   "
              f"(non-tumor confident={100*nt_conf:5.1f}%)")
    sweep = pd.DataFrame(rows)
    sweep.to_csv(f"{OUT}/tumor_leakage_sweep.csv", index=False)

    # breakdown of tumor leakage at 0.5 by predicted type
    leaked = tu_max >= 0.5
    print(f"\ntumor cells leaked at thr=0.5: {int(leaked.sum()):,} / {len(tu_max):,} "
          f"({100*leaked.mean():.1f}%)")
    print("predicted-type breakdown of leaked tumor cells:")
    print(pd.Series(tu_lab[leaked]).value_counts().to_string())

    pd.DataFrame({"x": xtu, "y": ytu, "pred_label": tu_lab,
                  "pred_proba": tu_max.round(4),
                  "leaks_at_0.5": leaked}).to_csv(
        f"{OUT}/tumor_predictions.csv", index=False)

    # ================= figures =================
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    # (a) proba distributions: tumor vs non-tumor
    ax[0].hist(nt_max, bins=50, density=True, alpha=0.6, color="#4c72b0",
               label="non-tumor (held-out)")
    ax[0].hist(tu_max, bins=50, density=True, alpha=0.6, color="#d62728",
               label="tumor")
    ax[0].axvline(0.5, c="k", ls="--", lw=1, label="thr 0.5")
    ax[0].set_xlabel("max class probability"); ax[0].set_ylabel("density")
    ax[0].set_title("confidence: tumor vs non-tumor", fontweight="bold")
    ax[0].legend(fontsize=8)
    # (b) leaked tumor by predicted type
    vc = pd.Series(tu_lab[leaked]).value_counts().reindex(GROUPS).fillna(0)
    ax[1].bar(vc.index, vc.values, color=[COL[g] for g in vc.index])
    ax[1].set_ylabel("tumor cells leaked (thr 0.5)")
    ax[1].set_title(f"where tumor cells leak ({int(leaked.sum())} of {len(tu_max)})",
                    fontweight="bold")
    ax[1].tick_params(axis="x", rotation=45)
    fig.suptitle("Backbone classifier — tumor leakage stress test", fontweight="bold")
    fig.savefig(f"{OUT}/tumor_leakage.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved tumor_leakage.png, tumor_leakage_sweep.csv, "
          f"tumor_predictions.csv -> {OUT}")


if __name__ == "__main__":
    main()
