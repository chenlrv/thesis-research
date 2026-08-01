"""Train classifiers on the Method-2 high-confidence backbone set and propagate
labels to the rest of slice 1.

Method-2 training set = the GATED cells (score_genes -> FDR 0.05 + scaled-margin
>= 1.5) restricted to the top 20% by scaled top-score WITHIN each of the 5
backbone groups (Myeloid, Vascular, Astrocytes, Ependymal, Neurons).

Steps:
  1. Build the training set + features (full log-normalized panel expression).
  2. Compare LogisticRegression / RandomForest / XGBoost by 5-fold stratified CV
     (macro-F1, balanced accuracy, per-class F1). Pick the best.
  3. Honesty check: re-run the winner's CV with the marker genes removed (labels
     were marker-derived, so this shows how much is non-marker signal).
  4. Refit the winner on all training cells; predict every non-tumor cell with a
     class probability. Predictions below CONF_THRESH -> 'uncertain'.
  5. Save model comparison, predictions, and plots.

Output -> score_genes_slice1_merged/classify/
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402
from run_score_genes_merged import MARKERS as BACKBONE_MARKERS  # noqa: E402

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
TOPFRAC = 0.20
CONF_THRESH = 0.50
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2", "uncertain": "#dddddd"}
SEED = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def make_models():
    return {
        "LogReg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced",
                               C=1.0, n_jobs=1, random_state=SEED)),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=1, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            eval_metric="mlogloss", n_jobs=1, random_state=SEED),
    }


def cv_eval(model, X, y_enc, classes, skf):
    pred = cross_val_predict(model, X, y_enc, cv=skf, method="predict", n_jobs=1)
    macro_f1 = f1_score(y_enc, pred, average="macro")
    bal_acc = balanced_accuracy_score(y_enc, pred)
    per_class = f1_score(y_enc, pred, average=None,
                         labels=np.arange(len(classes)))
    return pred, macro_f1, bal_acc, per_class


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = SCORES_CSV
    adata = ch.load_adata_with_calls()
    df = pd.read_csv(SCORES_CSV)
    assert np.allclose(df["x"].to_numpy(), adata.obs["x"].to_numpy(), atol=1e-3)

    lab_all = df["celltype"].to_numpy()
    score_all = df["top_scaled"].to_numpy()
    x_all, y_all = df["x"].to_numpy(), df["y"].to_numpy()

    # ---- Method-2 training mask: per-group top 20% of the gated cells ----
    train_mask = np.zeros(adata.n_obs, bool)
    for g in GROUPS:
        gm = lab_all == g
        t = np.quantile(score_all[gm], 1 - TOPFRAC)
        train_mask |= gm & (score_all >= t)
    print(f"non-tumor cells: {adata.n_obs:,}   training (Method 2): "
          f"{int(train_mask.sum()):,}")

    # ---- features: full log-normalized expression ----
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)
    var_names = np.asarray(adata.var_names)

    Xtr = X[train_mask]
    ytr = lab_all[train_mask]
    le = LabelEncoder().fit(GROUPS)
    ytr_enc = le.transform(ytr)
    classes = list(le.classes_)
    print("class balance:", {c: int((ytr == c).sum()) for c in classes})

    # ---- 5-fold stratified CV: compare the three models ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, preds = [], {}
    for name, model in make_models().items():
        pred, mf1, bacc, pcf1 = cv_eval(model, Xtr, ytr_enc, classes, skf)
        preds[name] = pred
        rows.append({"model": name, "macro_f1": round(mf1, 4),
                     "balanced_acc": round(bacc, 4),
                     **{f"f1_{c}": round(pcf1[i], 3) for i, c in enumerate(classes)}})
        print(f"  {name:13s} macro-F1={mf1:.4f}  bal-acc={bacc:.4f}")
    cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    cmp.to_csv(f"{OUT}/model_comparison.csv", index=False)
    print("\nmodel comparison:\n" + cmp.to_string(index=False))

    best_name = cmp.iloc[0]["model"]
    print(f"\nbest model: {best_name}")

    # winner confusion matrix (from CV predictions)
    cmtx = confusion_matrix(ytr_enc, preds[best_name],
                            labels=np.arange(len(classes)))
    print("\nCV classification report (winner):")
    print(classification_report(ytr_enc, preds[best_name],
                                target_names=classes, digits=3))

    # ---- honesty check: winner CV with marker genes removed ----
    marker_set = set()
    for lst in BACKBONE_MARKERS.values():
        marker_set.update(lst)
    keep = ~np.isin(var_names, list(marker_set))
    n_markers = int((~keep).sum())
    _, mf1_nomark, bacc_nomark, _ = cv_eval(
        make_models()[best_name], Xtr[:, keep], ytr_enc, classes, skf)
    print(f"\nhonesty check ({n_markers} marker genes removed): "
          f"macro-F1={mf1_nomark:.4f}  bal-acc={bacc_nomark:.4f}  "
          f"(full={cmp.iloc[0]['macro_f1']:.4f})")

    # ---- refit winner on all training cells, predict every non-tumor cell ----
    winner = make_models()[best_name]
    winner.fit(Xtr, ytr_enc)
    proba = winner.predict_proba(X)
    pred_enc = proba.argmax(1)
    pred_lab = le.inverse_transform(pred_enc)
    pred_max = proba.max(1)

    final = np.where(pred_max >= CONF_THRESH, pred_lab, "uncertain").astype(object)
    final[train_mask] = lab_all[train_mask]  # keep gold labels for training cells
    print(f"\nconfidence threshold {CONF_THRESH}: "
          f"{int((pred_max < CONF_THRESH).sum()):,} cells -> uncertain "
          f"(of {adata.n_obs:,})")
    print("\nfinal label counts (slice 1, non-tumor):")
    print(pd.Series(final).value_counts().to_string())

    out = pd.DataFrame({
        "x": x_all, "y": y_all, "is_train": train_mask,
        "gated_label": lab_all, "pred_label": pred_lab,
        "pred_proba": pred_max.round(4), "final_label": final,
    })
    out.to_csv(f"{OUT}/slice1_backbone_predictions.csv", index=False)

    # ================= figures =================
    # (1) model comparison bar
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.bar(cmp["model"], cmp["macro_f1"], color="#4c72b0")
    for i, v in enumerate(cmp["macro_f1"]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(mf1_nomark, c="r", ls="--", lw=1,
               label=f"{best_name} no-markers ({mf1_nomark:.3f})")
    ax.set_ylabel("CV macro-F1"); ax.set_ylim(0, 1.05)
    ax.set_title("5-fold CV: model comparison", fontweight="bold")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # (2) winner confusion matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    cmn = cmtx / cmtx.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for a in range(len(classes)):
        for b in range(len(classes)):
            ax.text(b, a, f"{cmn[a, b]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if cmn[a, b] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true (Method-2 label)")
    ax.set_title(f"{best_name} CV confusion (row-normalized)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(f"{OUT}/winner_confusion.png", bbox_inches="tight")
    plt.close(fig)

    # (3) prediction probability histogram
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.hist(pred_max[~train_mask], bins=60, color="#999999")
    ax.axvline(CONF_THRESH, c="r", ls="--", lw=1, label=f"threshold {CONF_THRESH}")
    ax.set_xlabel("max class probability (non-training cells)")
    ax.set_ylabel("cells"); ax.set_title("prediction confidence", fontweight="bold")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT}/prediction_confidence.png", bbox_inches="tight")
    plt.close(fig)

    # (4) spatial map of final labels
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    order = ["uncertain"] + GROUPS
    for g in order:
        m = final == g
        if m.any():
            ax.scatter(x_all[m], y_all[m], s=2, c=COL[g], linewidths=0,
                       rasterized=True, label=f"{g} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Slice 1 backbone — {best_name} propagated labels", fontweight="bold")
    ax.legend(loc="lower right", markerscale=4, fontsize=8, frameon=True)
    fig.savefig(f"{OUT}/slice1_backbone_spatial.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved model_comparison.(csv|png), winner_confusion.png, "
          f"prediction_confidence.png, slice1_backbone_spatial.png, "
          f"slice1_backbone_predictions.csv -> {OUT}")


if __name__ == "__main__":
    main()
