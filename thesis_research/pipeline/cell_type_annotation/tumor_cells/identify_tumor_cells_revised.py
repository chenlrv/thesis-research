from sklearn.base import clone
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import anndata as ad
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from thesis_research.utils.columns import CENTER_Y_GLOBAL_PX, CENTER_X_GLOBAL_PX

BASE_DIR = r"D:/thesis-research/"

def _to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def _get_classifier_models(random_state=42, scale_pos_weight=1.0):
    """
    Label convention:
        1 = tumor
        0 = healthy / non-tumor
    """
    return {
        "logistic_regression": LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }


def _get_scoring():
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "ap": "average_precision",
    }


def _fit_models(models, X_ref, y_ref, cv, scoring):
    metrics_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        cv_res = cross_validate(
            model,
            X_ref,
            y_ref,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )

        row = {"model": model_name}
        for metric_name in scoring:
            row[f"{metric_name}_mean"] = cv_res[f"test_{metric_name}"].mean()
            row[f"{metric_name}_std"] = cv_res[f"test_{metric_name}"].std()
        metrics_rows.append(row)

        fitted_model = clone(model)
        fitted_model.fit(X_ref, y_ref)
        fitted_models[model_name] = fitted_model

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .set_index("model")
        .sort_values(by="roc_auc_mean", ascending=False)
    )

    return fitted_models, metrics_df


def _add_scores_to_subset(
    fitted_models,
    X_all,
    adata_raw,
    selected_mask,
    healthy_ref_mask,
    tumor_ref_mask,
):
    """
    Adds per-cell tumor probabilities to a returned AnnData subset.
    """
    score_dict = {}
    for model_name, fitted_model in fitted_models.items():
        score_dict[model_name] = fitted_model.predict_proba(X_all)[:, 1]  # P(tumor)

    adata_sub = adata_raw[selected_mask].copy()

    group = np.full(adata_sub.n_obs, "Tumor Candidate", dtype=object)
    group[np.where(healthy_ref_mask[selected_mask])[0]] = "Healthy Ref"
    group[np.where(tumor_ref_mask[selected_mask])[0]] = "Tumor Ref"

    adata_sub.obs["classifier_group"] = group
    adata_sub.obs["is_reference"] = adata_sub.obs["classifier_group"].isin(
        ["Healthy Ref", "Tumor Ref"]
    )
    adata_sub.obs["reference_label"] = np.where(
        adata_sub.obs["classifier_group"] == "Tumor Ref",
        1,
        np.where(adata_sub.obs["classifier_group"] == "Healthy Ref", 0, np.nan),
    )

    adata_sub.obs["score_tumor_logreg"] = score_dict["logistic_regression"]
    adata_sub.obs["score_tumor_rf"] = score_dict["random_forest"]
    adata_sub.obs["score_tumor_xgb"] = score_dict["xgboost"]
    adata_sub.obs["score_tumor_mean"] = np.vstack(
        [
            score_dict["logistic_regression"],
            score_dict["random_forest"],
            score_dict["xgboost"],
        ]
    ).mean(axis=0)

    adata_sub.obs["pred_tumor_logreg"] = (adata_sub.obs["score_tumor_logreg"] >= 0.5).astype(int)
    adata_sub.obs["pred_tumor_rf"] = (adata_sub.obs["score_tumor_rf"] >= 0.5).astype(int)
    adata_sub.obs["pred_tumor_xgb"] = (adata_sub.obs["score_tumor_xgb"] >= 0.5).astype(int)
    adata_sub.obs["pred_tumor_mean"] = (adata_sub.obs["score_tumor_mean"] >= 0.5).astype(int)

    return adata_sub


def run_model2_classifiers_on_pca(
    adata,
    n_comps=50,
    n_pcs=30,
    n_splits=5,
    random_state=42,
):
    """
    Model 2:
    - use healthy refs + tumor refs as anchor/reference cells
    - fit scaler + PCA on reference cells only
    - train Logistic Regression / Random Forest / XGBoost on PCA space
    - run CV on reference cells
    - score reference + candidate cells

    Label convention:
        1 = tumor
        0 = healthy / non-tumor
    """
    adata_raw = adata.copy()
    adata_proc = adata.copy()

    healthy_ref_ids = _get_healthy_ref_ids()
    tumor_ref_ids = _get_tumor_ref_ids()
    tumor_candidate_ids = _get_tumor_candidates_ids()

    healthy_ref_mask = adata_proc.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata_proc.obs_names.isin(tumor_ref_ids)
    tumor_candidate_mask = adata_proc.obs_names.isin(tumor_candidate_ids)

    ref_mask = np.asarray(healthy_ref_mask | tumor_ref_mask)
    selected_mask = np.asarray(ref_mask | tumor_candidate_mask)

    if ref_mask.sum() == 0:
        raise ValueError("No reference cells found.")
    if healthy_ref_mask.sum() == 0:
        raise ValueError("No healthy reference cells found.")
    if tumor_ref_mask.sum() == 0:
        raise ValueError("No tumor reference cells found.")

    # ----------------------------
    # preprocess
    # ----------------------------
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)

    X_ref_raw = _to_dense(adata_proc.X[ref_mask])
    X_all_raw = _to_dense(adata_proc.X[selected_mask])

    # labels: 1=tumor, 0=healthy
    y_ref = np.asarray(tumor_ref_mask[ref_mask]).astype(int)

    # ----------------------------
    # fit scaling on refs only
    # ----------------------------
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(X_ref_raw)

    X_ref_scaled = np.clip(scaler.transform(X_ref_raw), -10, 10)
    X_all_scaled = np.clip(scaler.transform(X_all_raw), -10, 10)

    # ----------------------------
    # fit PCA on refs only
    # ----------------------------
    n_comps_eff = min(n_comps, X_ref_scaled.shape[0] - 1, X_ref_scaled.shape[1] - 1)
    pca_obj = PCA(n_components=n_comps_eff, random_state=random_state)
    pca_obj.fit(X_ref_scaled)

    X_ref_pca = pca_obj.transform(X_ref_scaled)
    X_all_pca = pca_obj.transform(X_all_scaled)

    n_pcs_eff = min(n_pcs, X_ref_pca.shape[1])
    X_ref_pca = X_ref_pca[:, :n_pcs_eff]
    X_all_pca = X_all_pca[:, :n_pcs_eff]

    # class imbalance handling for XGBoost
    n_tumor = int((y_ref == 1).sum())
    n_healthy = int((y_ref == 0).sum())
    scale_pos_weight = n_healthy / max(n_tumor, 1)

    models = _get_classifier_models(random_state=random_state, scale_pos_weight=scale_pos_weight)
    scoring = _get_scoring()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fitted_models, metrics_df = _fit_models(models, X_ref_pca, y_ref, cv, scoring)

    adata_sub = _add_scores_to_subset(
        fitted_models,
        X_all_pca,
        adata_raw,
        selected_mask,
        healthy_ref_mask,
        tumor_ref_mask,
    )

    print("\n=== Cross-validated performance on reference cells (PCA space) ===")
    print(metrics_df.round(4).to_string())
    _plot_results(adata_sub)
    return {
        "metrics_df": metrics_df,
        "adata_sub": adata_sub,
        "models": fitted_models,
        "scaler": scaler,
        "pca": pca_obj,
        "X_ref_pca": X_ref_pca,
        "X_all_pca": X_all_pca,
    }


def _get_healthy_ref_ids():
    CELL_COL = "cell_barcode"
    df_results1 = pd.read_csv(
        rf"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    df_results1 = df_results1[df_results1["predicted_cell_type"] != "Tumor"]
    score_cols = ["score_tumor", "score_brain_struct", "score_brain_immune"]

    df_results1["best_score"] = df_results1[score_cols].max(axis=1)
    df_results1["next_best_score"] = df_results1[score_cols].apply(
        lambda row: row.nlargest(2).iloc[-1], axis=1
    )
    df_results1["delta_score"] = abs(df_results1["best_score"] - df_results1["next_best_score"])

    df_results1 = df_results1[
        (df_results1["predicted_cell_type"] != "Tumor")
        & (df_results1["best_score"] > 0.5)
        & (df_results1["delta_score"] > 0.08)
    ]

    return set(df_results1[CELL_COL].astype(str))


def _get_tumor_ref_ids():
    CELL_COL = "cell_barcode"
    df_results1 = pd.read_csv(
        rf"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    df_results1 = df_results1[df_results1["predicted_cell_type"] == "Tumor"]
    df_results1["next_best_score"] = df_results1[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results1["delta_score"] = abs(df_results1["score_tumor"] - df_results1["next_best_score"])

    df_results1 = df_results1[
        (df_results1["predicted_cell_type"] == "Tumor")
        & (df_results1["score_tumor"] > 0.5)
        & (df_results1["delta_score"] > 0.08)
        & (df_results1["score_tumor"] > df_results1["next_best_score"])
    ]

    return set(df_results1[CELL_COL].astype(str))

def _get_tumor_candidates_ids():
    CELL_COL = "cell_barcode"
    df_results1 = pd.read_csv(
        rf"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    df_results1 = df_results1[df_results1["predicted_cell_type"] == "Tumor"]
    df_results1["next_best_score"] = df_results1[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results1["delta_score"] = abs(df_results1["score_tumor"] - df_results1["next_best_score"])

    df_results1 = df_results1[
        (df_results1["predicted_cell_type"] == "Tumor")
        & (df_results1["score_tumor"] > 0.2)
        & (df_results1["delta_score"] > 0.08)
        & (df_results1["score_tumor"] > df_results1["next_best_score"])
    ]

    return set(df_results1[CELL_COL].astype(str))




def _plot_tumor_cells(tumor_cells, classifier_name):
    adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\slice_1_adata.h5ad")

    adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]
    adata.obsm["spatial"] = np.stack(
        [adata.obs[CENTER_X_GLOBAL_PX].values, adata.obs[CENTER_Y_GLOBAL_PX].values], axis=1
    )
    coords = np.c_[
        adata.obs[CENTER_X_GLOBAL_PX].to_numpy(), adata.obs[CENTER_Y_GLOBAL_PX].to_numpy()
    ]

    cells_to_keep =  adata.obs_names.isin(tumor_cells)

    is_background = ~cells_to_keep
    plt.figure(figsize=(10, 10))
    plt.scatter(
        adata.obs.loc[is_background, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_background, CENTER_Y_GLOBAL_PX],
        c="#CFCFCF",
        s=0.7,  # Tiny dots
        alpha=1,  # Very faint
        edgecolors="none",
    )
    # LAYER 2: The "Tumor" (Foreground)
    # Highlighted in Red
    plt.scatter(
        adata.obs.loc[~is_background, CENTER_X_GLOBAL_PX],
        adata.obs.loc[~is_background, CENTER_Y_GLOBAL_PX],
        c="red",
        s=1.5,  # Larger dots to stand out
        alpha=0.8,  # Solid color
        edgecolors="none",
        label="D122 Tumor Cells",
    )
    plt.title(
        f"L321 slice 1 spatial Mapping tumor cells\n {classifier_name}\n{cells_to_keep.sum()} tumor cells",
        fontsize=15,
    )
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def _plot_results(adata_sub):
    map = {
        "Logistic Regression": "pred_tumor_logreg",
        "Random Forest": "pred_tumor_rf",
        "XGBoost": "pred_tumor_xgb",
    }

    for model_name, pred_col in map.items():
        healthy_cells = adata_sub.obs_names[adata_sub.obs[pred_col] == 1]
        _plot_tumor_cells(healthy_cells, model_name)





run_model2_classifiers_on_pca(
    adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/slice_1_adata.h5ad")
)