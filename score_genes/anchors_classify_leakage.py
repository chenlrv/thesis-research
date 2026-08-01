"""Train the backbone classifier on the select_anchors high-conf set (the 3-cond
rule: marker gate + top-30% centroid gap + gap>0 + kNN purity >=70%) instead of
the Method-2 score-quantile set, then:
  * classify all non-tumor cells (propagation counts: Myeloid, Neurons, ...)
  * classify the 4,248 tumor cells as a leakage validation.

The anchors file is the OLD 7-type config; mapped to the 5-type merged backbone
(Microglia+Macrophage -> Myeloid, Endothelial+Pericytes -> Vascular) for
comparability with the Method-2 experiments.

Output -> score_genes_slice1_merged/classify/anchors_*.{png,csv}
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
ANCHORS = "D:/thesis-research/score_genes_slice1/anchors.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
CONF_THRESH = 0.5
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
MAP7TO5 = {"Microglia": "Myeloid", "Macrophage": "Myeloid",
           "Endothelial": "Vascular", "Pericytes": "Vascular",
           "Astrocytes": "Astrocytes", "Ependymal": "Ependymal",
           "Neurons": "Neurons"}
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2", "uncertain": "#dddddd"}
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
    xnt_arr, ynt_arr = df["x"].to_numpy(), df["y"].to_numpy()
    coord_to_idx = {(round(float(x), 2), round(float(y), 2)): i
                    for i, (x, y) in enumerate(zip(xnt_arr, ynt_arr))}

    # ---- load anchors, map 7->5, align to non-tumor rows by coordinate ----
    anc = pd.read_csv(ANCHORS)
    anc = anc[anc["is_anchor"].astype(str).isin(["True", "1", "TRUE", "true"])]
    train_mask = np.zeros(Xnt.shape[0], bool)
    train_lab = np.empty(Xnt.shape[0], dtype=object)
    matched = 0
    for x, y, lab in zip(anc["x"], anc["y"], anc["provisional_label"]):
        key = (round(float(x), 2), round(float(y), 2))
        i = coord_to_idx.get(key)
        if i is not None and lab in MAP7TO5:
            train_mask[i] = True
            train_lab[i] = MAP7TO5[lab]
            matched += 1
    print(f"non-tumor: {Xnt.shape[0]:,}   tumor: {Xtu.shape[0]:,}")
    print(f"anchors in file: {len(anc):,}   matched to non-tumor rows: {matched:,}")
    print(f"anchor training cells: {int(train_mask.sum()):,}")
    ytr = train_lab[train_mask]
    print("\nanchor training set per merged type:")
    print(pd.Series(ytr).value_counts().reindex(GROUPS).fillna(0).astype(int).to_string())
    print("\n(original 7-type anchor counts):")
    print(anc["provisional_label"].value_counts().to_string())

    le = LabelEncoder().fit(GROUPS)
    ytr_enc = le.transform(ytr)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced",
                           C=1.0, random_state=SEED))

    # CV on the anchor set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cvpred = cross_val_predict(clf, Xnt[train_mask], ytr_enc, cv=skf, n_jobs=1)
    cv_f1 = f1_score(ytr_enc, cvpred, average="macro")
    print(f"\nCV macro-F1 (anchor set) = {cv_f1:.4f}")

    clf.fit(Xnt[train_mask], ytr_enc)

    # ---- propagation on ALL non-tumor cells ----
    pnt = clf.predict_proba(Xnt)
    nt_lab = le.inverse_transform(pnt.argmax(1))
    nt_max = pnt.max(1)
    final = np.where(nt_max >= CONF_THRESH, nt_lab, "uncertain").astype(object)
    final[train_mask] = ytr  # keep gold for anchor cells
    print(f"\n=== NON-TUMOR propagation (thr {CONF_THRESH}) ===")
    print(pd.Series(final).value_counts().to_string())

    pd.DataFrame({"x": xnt_arr, "y": ynt_arr, "is_anchor": train_mask,
                  "pred_label": nt_lab, "pred_proba": nt_max.round(4),
                  "final_label": final}).to_csv(
        f"{OUT}/anchors_nontumor_predictions.csv", index=False)

    # ---- tumor leakage validation ----
    ptu = clf.predict_proba(Xtu)
    tu_lab = le.inverse_transform(ptu.argmax(1))
    tu_max = ptu.max(1)
    print("\n=== TUMOR leakage (validation) ===")
    for thr in [0.5, 0.7, 0.9]:
        print(f"  thr={thr}: tumor leaked={100*float((tu_max>=thr).mean()):5.1f}%")
    leak05 = tu_max >= 0.5
    print(f"\ntumor leaked@0.5 = {int(leak05.sum()):,}/{len(tu_max):,} "
          f"({100*leak05.mean():.1f}%); by type:")
    print(pd.Series(tu_lab[leak05]).value_counts().reindex(GROUPS).fillna(0)
          .astype(int).to_string())

    pd.DataFrame({"x": xtu, "y": ytu, "pred_label": tu_lab,
                  "pred_proba": tu_max.round(4),
                  "leaks_at_0.5": leak05}).to_csv(
        f"{OUT}/anchors_tumor_predictions.csv", index=False)

    # ================= figures =================
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    # (a) non-tumor propagation counts
    vc = pd.Series(final).value_counts().reindex(GROUPS + ["uncertain"]).fillna(0)
    ax[0].bar(vc.index, vc.values,
              color=[COL.get(g, "#dddddd") for g in vc.index])
    for i, v in enumerate(vc.values):
        ax[0].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    ax[0].set_title("non-tumor propagation (anchor-trained)", fontweight="bold")
    ax[0].set_ylabel("cells"); ax[0].tick_params(axis="x", rotation=45)
    # (b) tumor leakage by type
    bd = pd.Series(tu_lab[leak05]).value_counts().reindex(GROUPS).fillna(0)
    ax[1].bar(bd.index, bd.values, color=[COL[g] for g in bd.index])
    for i, v in enumerate(bd.values):
        ax[1].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    ax[1].set_title(f"tumor leakage @0.5 ({int(leak05.sum())}/{len(tu_max)})",
                    fontweight="bold")
    ax[1].set_ylabel("tumor cells"); ax[1].tick_params(axis="x", rotation=45)
    fig.suptitle(f"select_anchors high-conf training (n={int(train_mask.sum()):,}, "
                 f"CV macro-F1={cv_f1:.3f})", fontweight="bold")
    fig.savefig(f"{OUT}/anchors_classify_leakage.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved anchors_classify_leakage.png, anchors_nontumor_predictions.csv, "
          f"anchors_tumor_predictions.csv -> {OUT}")


if __name__ == "__main__":
    main()
