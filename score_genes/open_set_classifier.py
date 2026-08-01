"""Open-set backbone classifier: a real 'none-of-the-above' option built only from
the 5 backbone classes (tumor stays held-out as validation).

Two-stage:
  1. 5-class LogReg gives the label (as before).
  2. Novelty guard: embed in scaled-PCA, fit a per-class Gaussian (Ledoit-Wolf)
     on the Method-2 training cells; a cell's novelty = Mahalanobis distance to
     the NEAREST class cloud. Cells beyond a calibrated cutoff -> 'none'.

The cutoff is calibrated on HELD-OUT backbone cells ('keep r% of real cells'),
then tumor rejection is measured at the same cutoff. We compare the distance
guard vs the softmax threshold by AUROC (tumor=positive) and a retention curve.

Output -> score_genes_slice1_merged/classify/openset_*.{png,csv}
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
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
TOPFRAC = 0.20
N_PCS = 50
KEEP = 0.95           # deployment cutoff: retain this fraction of real backbone cells
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2", "none": "#dddddd"}
SEED = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def fit_guard(E, y):
    return {g: LedoitWolf().fit(E[y == g]) for g in GROUPS}


def novelty(guard, E):
    """Min sqrt-Mahalanobis distance to any class cloud (lower = more in-distribution)."""
    D = np.column_stack([np.sqrt(np.clip(guard[g].mahalanobis(E), 0, None))
                         for g in GROUPS])
    return D.min(1)


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

    train_mask = np.zeros(Xnt.shape[0], bool)
    for g in GROUPS:
        gm = lab_all == g
        t = np.quantile(score_all[gm], 1 - TOPFRAC)
        train_mask |= gm & (score_all >= t)
    y_tr_all = lab_all[train_mask]
    print(f"non-tumor: {Xnt.shape[0]:,}  tumor: {Xtu.shape[0]:,}  "
          f"Method-2 train: {int(train_mask.sum()):,}")

    # ---- scaled-PCA embedding, fit on training cells ----
    scaler = StandardScaler().fit(Xnt[train_mask])
    pca = PCA(n_components=N_PCS, random_state=SEED).fit(scaler.transform(Xnt[train_mask]))
    emb = lambda M: pca.transform(scaler.transform(M))
    E_nt, E_tu = emb(Xnt), emb(Xtu)

    le = LabelEncoder().fit(GROUPS)

    # ---- honest calibration: split training into fit / held-out ----
    idx = np.where(train_mask)[0]
    fit_i, val_i = train_test_split(idx, test_size=0.3, stratify=lab_all[idx],
                                    random_state=SEED)
    guard_cal = fit_guard(E_nt[fit_i], lab_all[fit_i])
    nov_val = novelty(guard_cal, E_nt[val_i])     # held-out real backbone
    nov_tu = novelty(guard_cal, E_tu)             # tumor (never in fit)

    clf_cal = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
    clf_cal.fit(Xnt[fit_i], le.transform(lab_all[fit_i]))
    conf_val = clf_cal.predict_proba(Xnt[val_i]).max(1)
    conf_tu = clf_cal.predict_proba(Xtu).max(1)

    # ---- AUROC: tumor = positive (should be flagged OOD) ----
    yb = np.r_[np.zeros(len(nov_val)), np.ones(len(nov_tu))]
    auc_dist = roc_auc_score(yb, np.r_[nov_val, nov_tu])           # high novelty = OOD
    auc_soft = roc_auc_score(yb, np.r_[-conf_val, -conf_tu])       # low conf = OOD
    print(f"\nOOD detection AUROC (tumor vs held-out backbone):")
    print(f"  distance guard : {auc_dist:.3f}")
    print(f"  softmax conf   : {auc_soft:.3f}")

    # ---- retention / rejection trade-off ----
    rows = []
    print("\nat 'keep r% of real cells' -> tumor rejected:")
    for r in [0.90, 0.95, 0.99]:
        cut_d = np.quantile(nov_val, r)
        tr_d = float((nov_tu > cut_d).mean())
        cut_s = np.quantile(conf_val, 1 - r)
        tr_s = float((conf_tu < cut_s).mean())
        rows.append({"keep_real_%": int(r * 100),
                     "tumor_rejected_distance_%": round(100 * tr_d, 1),
                     "tumor_rejected_softmax_%": round(100 * tr_s, 1)})
        print(f"  keep {int(r*100)}%: distance rejects {100*tr_d:5.1f}%   "
              f"softmax rejects {100*tr_s:5.1f}%")
    pd.DataFrame(rows).to_csv(f"{OUT}/openset_tradeoff.csv", index=False)

    # ---- deploy: full guard + classifier, label all non-tumor with 'none' option ----
    guard = fit_guard(E_nt[train_mask], y_tr_all)
    cutoff = np.quantile(nov_val, KEEP)           # calibrated on held-out backbone
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
    clf.fit(Xnt[train_mask], le.transform(y_tr_all))

    nov_nt_all = novelty(guard, E_nt)
    lab_nt = le.inverse_transform(clf.predict_proba(Xnt).argmax(1))
    final = np.where(nov_nt_all > cutoff, "none", lab_nt).astype(object)
    final[train_mask] = y_tr_all
    nov_tu_all = novelty(guard, E_tu)
    tumor_rej = float((nov_tu_all > cutoff).mean())

    print(f"\n=== deployment (cutoff keeps {int(KEEP*100)}% of real backbone) ===")
    print(f"tumor cells flagged 'none': {100*tumor_rej:.1f}%  "
          f"({int((nov_tu_all>cutoff).sum()):,}/{len(nov_tu_all):,})")
    print("\nnon-tumor final labels:")
    print(pd.Series(final).value_counts().to_string())

    pd.DataFrame({"x": cxnt, "y": cynt, "is_train": train_mask,
                  "pred_label": lab_nt, "novelty": nov_nt_all.round(3),
                  "final_label": final}).to_csv(
        f"{OUT}/openset_nontumor_predictions.csv", index=False)

    # ================= figures =================
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), dpi=150)
    # (a) novelty distributions
    ax[0].hist(nov_val, bins=60, density=True, alpha=0.6, color="#4c72b0",
               label="held-out backbone")
    ax[0].hist(nov_tu, bins=60, density=True, alpha=0.6, color="#d62728", label="tumor")
    ax[0].axvline(np.quantile(nov_val, KEEP), c="k", ls="--", lw=1,
                  label=f"cutoff (keep {int(KEEP*100)}%)")
    ax[0].set_xlabel("novelty (min Mahalanobis dist)"); ax[0].set_ylabel("density")
    ax[0].set_title(f"distance guard (AUROC {auc_dist:.2f})", fontweight="bold")
    ax[0].legend(fontsize=8)
    # (b) retention/rejection curve (sweep)
    rr = np.linspace(0.5, 0.999, 60)
    td = [float((nov_tu > np.quantile(nov_val, r)).mean()) for r in rr]
    ts = [float((conf_tu < np.quantile(conf_val, 1 - r)).mean()) for r in rr]
    ax[1].plot(100 * rr, 100 * np.array(td), label=f"distance (AUROC {auc_dist:.2f})")
    ax[1].plot(100 * rr, 100 * np.array(ts), label=f"softmax (AUROC {auc_soft:.2f})")
    ax[1].set_xlabel("real backbone cells kept (%)")
    ax[1].set_ylabel("tumor cells rejected (%)")
    ax[1].set_title("tumor rejection vs real retention", fontweight="bold")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # (c) non-tumor final label counts
    vc = pd.Series(final).value_counts().reindex(GROUPS + ["none"]).fillna(0)
    ax[2].bar(vc.index, vc.values, color=[COL.get(g, "#dddddd") for g in vc.index])
    for i, v in enumerate(vc.values):
        ax[2].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    ax[2].set_title("non-tumor labels (with 'none')", fontweight="bold")
    ax[2].tick_params(axis="x", rotation=45)
    fig.suptitle("Open-set backbone classifier (Method-2 / 20% training)",
                 fontweight="bold")
    fig.savefig(f"{OUT}/openset_classifier.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved openset_classifier.png, openset_tradeoff.csv, "
          f"openset_nontumor_predictions.csv -> {OUT}")


if __name__ == "__main__":
    main()
