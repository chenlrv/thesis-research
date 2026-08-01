"""One-vs-rest backbone classifier with an abstention ('unknown') rule.

5 binary 'type vs rest' LogReg classifiers (class_weight balanced, so a fixed
threshold T means the same 'past the decision boundary' for every type). Scores
are independent (do NOT sum to 1), so a cell can be low on all -> unknown.

Abstention rule at threshold T:
  * 0 types with P>=T   -> unknown (no match)
  * exactly 1 type      -> that type
  * 2+ types with P>=T  -> unknown (ambiguous)

Trained on the Method-2/20% set; predicts all non-tumor cells. Compares to the
previous 5-way softmax (slice1_backbone_predictions.csv) and plots spatial maps.

Output -> score_genes_slice1_merged/classify/ovr_*.{png,csv}
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
PREV = f"{OUT}/slice1_backbone_predictions.csv"   # previous softmax result
TOPFRAC = 0.20
T = 0.5
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
PANELS = GROUPS + ["unknown"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2", "unknown": "#bbbbbb"}
SEED = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def spatial_plots(x, y, lab, tag, title):
    fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=160)
    for ax, g in zip(axes.ravel(), PANELS):
        m = lab == g
        ax.scatter(x, y, s=0.6, c="#eeeeee", linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=1.6, c=COL[g], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (n={int(m.sum()):,})", fontweight="bold")
    fig.suptitle(title + " — per type", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/{tag}_per_type.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 10), dpi=180)
    for g in ["unknown"] + GROUPS:
        m = lab == g
        if m.any():
            ax.scatter(x[m], y[m], s=1.2, c=COL[g], linewidths=0,
                       rasterized=True, label=f"{g} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title + " — all types", fontweight="bold")
    ax.legend(loc="lower right", markerscale=6, fontsize=9, frameon=True)
    fig.savefig(f"{OUT}/{tag}_all_types.png", bbox_inches="tight")
    plt.close(fig)


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
    Xall = adata.X
    Xall = (Xall.toarray() if sp.issparse(Xall) else np.asarray(Xall)).astype(np.float32)
    Xnt = Xall[~tumor]
    cxnt, cynt = cx[~tumor], cy[~tumor]

    df = pd.read_csv(SCORES_CSV)
    assert np.allclose(df["x"].to_numpy(), cxnt, atol=1e-3)
    lab_all = df["celltype"].to_numpy()
    score_all = df["top_scaled"].to_numpy()

    train_mask = np.zeros(Xnt.shape[0], bool)
    for g in GROUPS:
        gm = lab_all == g
        t = np.quantile(score_all[gm], 1 - TOPFRAC)
        train_mask |= gm & (score_all >= t)
    print(f"non-tumor: {Xnt.shape[0]:,}   Method-2 train: {int(train_mask.sum()):,}")

    Xtr = Xnt[train_mask]
    ytr = lab_all[train_mask]

    # ---- 5 binary 'type vs rest' classifiers (balanced) ----
    P = np.zeros((Xnt.shape[0], len(GROUPS)), dtype=np.float32)
    for j, g in enumerate(GROUPS):
        ybin = (ytr == g).astype(int)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               C=1.0, random_state=SEED))
        clf.fit(Xtr, ybin)
        P[:, j] = clf.predict_proba(Xnt)[:, 1]
        print(f"  {g:11s} trained (positives={int(ybin.sum())})")

    # ---- abstention rule ----
    passes = P >= T
    npass = passes.sum(1)
    argmax = P.argmax(1)
    final = np.full(Xnt.shape[0], "unknown", dtype=object)
    one = npass == 1
    final[one] = np.array(GROUPS)[argmax[one]]
    reason = np.where(npass == 0, "no_match",
                      np.where(npass >= 2, "ambiguous", "typed")).astype(object)
    final[train_mask] = ytr            # keep gold for training cells (as before)
    reason[train_mask] = "train"

    print(f"\n=== OvR abstention (T={T}) — non-tumor final labels ===")
    print(pd.Series(final).value_counts().to_string())
    print("\nunknown breakdown:")
    print(pd.Series(reason[final == "unknown"]).value_counts().to_string())

    out = pd.DataFrame({"x": cxnt, "y": cynt, "is_train": train_mask,
                        **{f"P_{g}": P[:, j].round(4) for j, g in enumerate(GROUPS)},
                        "n_pass": npass, "final_label": final, "reason": reason})
    out.to_csv(f"{OUT}/ovr_nontumor_predictions.csv", index=False)

    # ---- compare to previous softmax ----
    prev = pd.read_csv(PREV)
    assert np.allclose(prev["x"].to_numpy(), cxnt, atol=1e-3)
    prev_lab = prev["final_label"].to_numpy()
    comp = pd.DataFrame({"group": PANELS})
    comp["softmax"] = [int((prev_lab == g).sum()) if g != "unknown"
                       else int((prev_lab == "uncertain").sum()) for g in PANELS]
    comp["ovr_abstain"] = [int((final == g).sum()) for g in PANELS]
    print("\n=== counts: previous softmax vs OvR-abstain ===")
    print(comp.to_string(index=False))

    prev_norm = np.where(prev_lab == "uncertain", "unknown", prev_lab)
    both_typed = (prev_norm != "unknown") & (final != "unknown")
    agree = float((prev_norm[both_typed] == final[both_typed]).mean())
    now_unknown = int(((prev_norm != "unknown") & (final == "unknown")).sum())
    print(f"\namong cells typed by BOTH: {100*agree:.1f}% same label")
    print(f"cells softmax typed but OvR abstains: {now_unknown:,}")

    # ---- spatial plots ----
    spatial_plots(cxnt, cynt, final, "ovr_spatial",
                  f"Slice 1 OvR-abstain (T={T})")
    print(f"\nsaved ovr_nontumor_predictions.csv, ovr_spatial_per_type.png, "
          f"ovr_spatial_all_types.png -> {OUT}")


if __name__ == "__main__":
    main()
