"""Thesis Figure 3 — Classifier comparison on the joint reference pool.

Two-panel figure:

  Left:  Bar chart of 5-fold OOF CV metrics (accuracy / precision / recall / F1 /
         ROC-AUC) at the default 0.5 decision threshold for five classifier
         configurations: LogReg, LogReg+PCA, LogReg+KNN (PCA), Random Forest,
         XGBoost. The marginal differences here justify the choice of XGBoost.

  Right: OOF precision-recall curves for all five classifiers overlaid, with
         average-precision (AP) values in the legend. The curves are nearly
         indistinguishable, showing that the choice of model has little effect
         on overall PR behaviour — the model choice rests instead on behaviour
         on the candidate pool (see figure 4).

Tumor is the positive class (y = 1), so reported precision is the fraction of
tumor calls that are genuinely tumor — the purity of the refined tumor set —
and recall is the fraction of true tumor cells retained. The classifier
internally scores P(healthy); these are the same decision, stated from the
tumor side.

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

    y_ref = 1  : tumor anchor (positive class in the binary frame)
    y_ref = 0  : healthy / look-alike

    Tumor is the positive class so that reported precision is the fraction of
    tumor calls that are genuinely tumor -- the purity of the refined tumor set,
    which is the quantity of interest. Accuracy and ROC-AUC are unchanged by the
    choice; precision, recall, F1 and AP are reported for the tumor class.
    """
    slide_ids = ["L321", "L34"]
    healthy_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ids = {sid: _get_tumor_ref_ids(sid) for sid in slide_ids}
    all_healthy = set().union(*healthy_ids.values())
    all_tumor = set().union(*tumor_ids.values())
    all_ref = all_healthy | all_tumor

    # Subset each slide to its reference cells before normalizing and before
    # concatenating. Normalizing all 846k cells needs ~600 MB of temporaries per
    # slide and exhausts a 16 GB machine; normalize_total scales each cell by its
    # own total, so restricting to the reference cells first is numerically
    # identical.
    adatas = {}
    for sid in slide_ids:
        path = SLIDE_CACHE / f"sample_{sid}_adata.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
        adata = ad.read_h5ad(path)
        adata_ref = adata[np.asarray(adata.obs_names.isin(all_ref))].copy()
        del adata
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        adatas[sid] = adata_ref

    adata_joint = ad.concat(adatas, join="inner", label="slide_id")
    del adatas
    healthy_mask = adata_joint.obs_names.isin(all_healthy)
    tumor_mask = adata_joint.obs_names.isin(all_tumor)
    ref_mask = np.asarray(healthy_mask | tumor_mask)

    X_ref = _to_dense(adata_joint.X[ref_mask])
    y_ref = np.asarray(tumor_mask[ref_mask]).astype(int)

    print(f"reference pool: {len(y_ref):,} cells")
    print(f"  tumor anchor        : {int((y_ref == 1).sum()):,}")
    print(f"  healthy (look-alike): {int((y_ref == 0).sum()):,}")
    return X_ref, y_ref


def compute_oof_probabilities(X_ref, y_ref):
    """Returns (dict of model_name -> OOF P(tumor) vector, per-cell fold index).

    The fold index lets the same metrics be recomputed within each fold, which
    is what the error bars in the figure show: the spread across resampling
    folds, against which the between-model differences should be judged.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.zeros(len(y_ref), dtype=int)
    for k, (_, val_idx) in enumerate(cv.split(X_ref, y_ref)):
        fold_ids[val_idx] = k

    # Pre-compute PCA space for the KNN model (fit on the full reference pool)
    scaler_knn = StandardScaler()
    X_scaled = np.clip(scaler_knn.fit_transform(X_ref), -10, 10)
    n_comps_eff = min(N_PCS, X_ref.shape[0] - 1, X_ref.shape[1] - 1)
    pca_knn = PCA(n_components=n_comps_eff, random_state=RANDOM_STATE)
    X_pca = pca_knn.fit_transform(X_scaled)

    # Library defaults throughout, except max_iter (raised from 100 so the
    # solver converges in 958 dimensions), the 300-tree forest (variance
    # reduction; RF performance is monotone non-decreasing in tree count) and
    # the 50 components / 15 neighbours standard in single-cell analysis.
    # Class balancing was deliberately NOT applied: the reference pool is only
    # mildly imbalanced (863:630) and XGBoost is run without it, so weighting
    # the other models would make the comparison asymmetric.
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    logreg_pca_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCS, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    # xgboost 3.2.0 library defaults. A leave-one-out sensitivity analysis
    # (xgb_default_sensitivity.py) showed every hand-set value we previously
    # used was indistinguishable from its default on this reference pool, so
    # there is nothing to justify departing from.
    xgb_model = XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    rf_model = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
    )
    logreg_knn_clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)

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
    }, fold_ids


def compute_fold_spread(y_true, oof_dict, fold_ids):
    """Standard deviation of each metric across the CV folds.

    Reported metrics are computed once on the pooled out-of-fold predictions;
    this recomputes them within each fold to give the resampling spread the
    error bars display. It is a dispersion estimate, not a confidence interval.
    """
    rows = []
    for name in MODEL_ORDER:
        oof = oof_dict[name]
        per_fold = []
        for k in np.unique(fold_ids):
            m = fold_ids == k
            yt, p = y_true[m], oof[m]
            pred = (p >= PROB_THRESH).astype(int)
            per_fold.append({
                "accuracy":  accuracy_score(yt, pred),
                "precision": precision_score(yt, pred, zero_division=0),
                "recall":    recall_score(yt, pred, zero_division=0),
                "f1":        f1_score(yt, pred, zero_division=0),
                "roc_auc":   roc_auc_score(yt, p) if len(np.unique(yt)) == 2 else np.nan,
            })
        rows.append({"model": name, **pd.DataFrame(per_fold).std(ddof=1).to_dict()})
    return pd.DataFrame(rows).set_index("model")


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


def make_figure(metrics_df, y_true, oof_dict, out_path, spread_df=None):
    # No figure title: the caption carries it in the thesis.
    fig = plt.figure(figsize=(16, 7))

    # ── Left: CV metric bar chart ──────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(metric_cols))
    width = 0.8 / len(MODEL_ORDER)

    for i, mname in enumerate(MODEL_ORDER):
        vals = metrics_df.loc[mname, metric_cols].values.astype(float)
        errs = (spread_df.loc[mname, metric_cols].values.astype(float)
                if spread_df is not None else None)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax1.bar(x + offset, vals, width, label=mname, yerr=errs, capsize=2.5,
                error_kw={"elinewidth": 1.0, "ecolor": "#444444"},
                color=MODEL_COLORS[mname], edgecolor="white", linewidth=0.5)

    # Axis starts at zero: bar length is then proportional to the value, and the
    # between-model differences are not visually magnified.
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("5-fold out-of-fold cross-validation metrics\n"
                  "(error bars: standard deviation across folds)", fontsize=12)
    # Legend goes below the axes: the bars fill the panel, so an in-axes legend
    # sits on top of them.
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5,
               fontsize=9.5, frameon=False, handlelength=1.4, columnspacing=1.4)
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

    ax2.set_xlabel("Recall  (positive = tumor)", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("OOF precision-recall curves — all classifiers", fontsize=12)
    ax2.legend(loc="lower left", fontsize=9.5, frameon=True)
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(metrics_df["precision"].min() - 0.05, 1.005)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    X_ref, y_ref = build_reference_pool()
    oof_dict, fold_ids = compute_oof_probabilities(X_ref, y_ref)
    metrics_df = compute_metrics(y_ref, oof_dict)
    spread_df = compute_fold_spread(y_ref, oof_dict, fold_ids)

    print("\n=== Model comparison (5-fold OOF CV, threshold = "
          f"{PROB_THRESH}) ===")
    print(metrics_df.round(4).to_string())

    print("\n=== Fold-to-fold standard deviation (error bars) ===")
    print(spread_df.round(4).to_string())

    csv_out = OUT_DIR / "figure_3_model_comparison.csv"
    metrics_df.to_csv(csv_out)
    spread_df.to_csv(OUT_DIR / "figure_3_model_comparison_foldsd.csv")
    print(f"\nmetrics table: {csv_out}")

    make_figure(
        metrics_df, y_ref, oof_dict,
        OUT_DIR / "figure_3_model_comparison.png",
        spread_df=spread_df,
    )
