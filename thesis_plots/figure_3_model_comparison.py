"""Thesis Figure 3 — Classifier comparison on the joint reference pool.

Two-panel figure:

  Left:  Bar chart of 5-fold OOF CV metrics (accuracy / precision / recall / F1 /
         ROC-AUC) at the default 0.5 decision threshold for five classifier
         configurations: LogReg, LogReg+PCA, LogReg+KNN (PCA), Random Forest,
         XGBoost. The marginal differences here justify the choice of XGBoost.

  Right: OOF precision-recall curves for all five classifiers overlaid, with
         average-precision (AP) values in the legend. The curves are nearly
         indistinguishable, showing that the choice of model has little effect
         on overall PR behaviour — the choice rests on the marginal bar-chart
         differences and on architectural reasoning (non-linearity, class
         imbalance handling).

The labelling convention follows the rest of the pipeline: positive class
(y = 1) means 'healthy / look-alike' — the class to be rejected from the
final tumor pool. score_healthy = P(y = 1).

Saves: thesis_plots/figure_3_model_comparison.png  (300 dpi)
"""
import pathlib

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    precision_recall_curve, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import (
    _get_healthy_ref_ids,
    _get_tumor_ref_ids,
)

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
OUT_DIR = BASE_DIR / "thesis_plots"

PROB_THRESH = 0.5    # P(healthy) operating threshold
N_PCS = 50
KNN_K = 15
RANDOM_STATE = 42
N_SPLITS = 5

MODEL_ORDER = ["LogReg", "LogReg + PCA", "LogReg + KNN (PCA)", "Random Forest", "XGBoost"]
MODEL_COLORS = {
    "LogReg":              "#4C72B0",
    "LogReg + PCA":        "#55A868",
    "LogReg + KNN (PCA)":  "#C44E52",
    "Random Forest":       "#DD8452",
    "XGBoost":             "#8172B2",
}
CHOSEN_MODEL = "XGBoost"


def _to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def build_reference_pool():
    """Load both slides, normalize per-sample, concatenate, return X_ref, y_ref.

    y_ref = 1  : healthy / look-alike (positive class in the binary frame)
    y_ref = 0  : tumor anchor
    """
    slide_ids = ["L321", "L34"]
    adatas = {}
    for sid in slide_ids:
        path = SLIDE_CACHE / f"sample_{sid}_adata.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
        adata = ad.read_h5ad(path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adatas[sid] = adata

    healthy_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ids = {sid: _get_tumor_ref_ids(sid) for sid in slide_ids}
    all_healthy = set().union(*healthy_ids.values())
    all_tumor = set().union(*tumor_ids.values())

    adata_joint = ad.concat(adatas, join="inner", label="slide_id")
    healthy_mask = adata_joint.obs_names.isin(all_healthy)
    tumor_mask = adata_joint.obs_names.isin(all_tumor)
    ref_mask = np.asarray(healthy_mask | tumor_mask)

    X_ref = _to_dense(adata_joint.X[ref_mask])
    y_ref = np.asarray(healthy_mask[ref_mask]).astype(int)

    print(f"reference pool: {len(y_ref):,} cells")
    print(f"  healthy (look-alike): {int((y_ref == 1).sum()):,}")
    print(f"  tumor anchor        : {int((y_ref == 0).sum()):,}")
    return X_ref, y_ref


def compute_oof_probabilities(X_ref, y_ref):
    """Returns dict of model_name -> OOF P(healthy) vector."""
    n_pos = int((y_ref == 1).sum())
    n_neg = int((y_ref == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Pre-compute PCA space for the KNN model (fit on the full reference pool)
    scaler_knn = StandardScaler()
    X_scaled = np.clip(scaler_knn.fit_transform(X_ref), -10, 10)
    n_comps_eff = min(N_PCS, X_ref.shape[0] - 1, X_ref.shape[1] - 1)
    pca_knn = PCA(n_components=n_comps_eff, random_state=RANDOM_STATE)
    X_pca = pca_knn.fit_transform(X_scaled)

    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    logreg_pca_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCS, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    logreg_knn_clf = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)

    print("\nComputing OOF probabilities (5-fold stratified CV)...")
    print("  LogReg...")
    oof_lr = cross_val_predict(logreg_pipe, X_ref, y_ref, cv=cv, method="predict_proba")[:, 1]
    print("  LogReg + PCA...")
    oof_lr_pca = cross_val_predict(logreg_pca_pipe, X_ref, y_ref, cv=cv, method="predict_proba")[:, 1]
    print("  XGBoost...")
    oof_xgb = cross_val_predict(xgb_model, X_ref, y_ref, cv=cv, method="predict_proba")[:, 1]
    print("  Random Forest...")
    oof_rf = cross_val_predict(rf_model, X_ref, y_ref, cv=cv, method="predict_proba")[:, 1]
    print("  LogReg + KNN (PCA)...")
    oof_knn_lr = cross_val_predict(logreg_knn_clf, X_pca, y_ref, cv=cv, method="predict_proba")[:, 1]
    knn_oof = np.zeros(len(X_pca))
    for train_idx, val_idx in cv.split(X_pca, y_ref):
        nn_fold = NearestNeighbors(n_neighbors=min(KNN_K, len(train_idx)))
        nn_fold.fit(X_pca[train_idx])
        knn_oof[val_idx] = y_ref[train_idx][
            nn_fold.kneighbors(X_pca[val_idx], return_distance=False)
        ].mean(axis=1)
    oof_knn_score = (oof_knn_lr + knn_oof) / 2

    return {
        "LogReg":              oof_lr,
        "LogReg + PCA":        oof_lr_pca,
        "LogReg + KNN (PCA)":  oof_knn_score,
        "Random Forest":       oof_rf,
        "XGBoost":             oof_xgb,
    }


def compute_metrics(y_true, oof_dict):
    rows = []
    for name in MODEL_ORDER:
        oof = oof_dict[name]
        pred = (oof >= PROB_THRESH).astype(int)
        rows.append({
            "model": name,
            "accuracy":  accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall":    recall_score(y_true, pred, zero_division=0),
            "f1":        f1_score(y_true, pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_true, oof),
            "ap":        average_precision_score(y_true, oof),
        })
    return pd.DataFrame(rows).set_index("model")


def make_figure(metrics_df, y_true, oof_dict, out_path):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        "Classifier comparison on the joint reference pool",
        fontsize=16, fontweight="bold", y=0.99,
    )

    # ── Left: CV metric bar chart ──────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metric_cols))
    width = 0.8 / len(MODEL_ORDER)

    for i, mname in enumerate(MODEL_ORDER):
        vals = metrics_df.loc[mname, metric_cols].values.astype(float)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax1.bar(x + offset, vals, width, label=mname,
                color=MODEL_COLORS[mname], edgecolor="white", linewidth=0.5)

    y_min = max(0.0, metrics_df[metric_cols].values.min() - 0.03)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)
    ax1.set_ylim(y_min, 1.005)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("5-fold out-of-fold cross-validation metrics", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9.5, frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    # ── Right: PR curves for all four classifiers overlaid ────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    for mname in MODEL_ORDER:
        oof = oof_dict[mname]
        precision, recall, _ = precision_recall_curve(y_true, oof)
        ap = average_precision_score(y_true, oof)
        ax2.plot(
            recall, precision,
            color=MODEL_COLORS[mname], linewidth=2.0, alpha=0.9,
            label=f"{mname}  (AP = {ap:.3f})",
        )

    ax2.set_xlabel("Recall  (positive = healthy / look-alike)", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("OOF precision-recall curves — all classifiers", fontsize=12)
    ax2.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(metrics_df["precision"].min() - 0.05, 1.005)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    X_ref, y_ref = build_reference_pool()
    oof_dict = compute_oof_probabilities(X_ref, y_ref)
    metrics_df = compute_metrics(y_ref, oof_dict)

    print("\n=== Model comparison (5-fold OOF CV, threshold = "
          f"{PROB_THRESH}) ===")
    print(metrics_df.round(4).to_string())

    csv_out = OUT_DIR / "figure_3_model_comparison.csv"
    metrics_df.to_csv(csv_out)
    print(f"\nmetrics table: {csv_out}")

    make_figure(
        metrics_df, y_ref, oof_dict,
        OUT_DIR / "figure_3_model_comparison.png",
    )
