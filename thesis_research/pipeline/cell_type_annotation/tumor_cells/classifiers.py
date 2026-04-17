import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import anndata as ad


from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import (
    _get_healthy_ref_ids,
    _get_tumor_ref_ids,
    _get_tumor_candidates_ids,
    _plot_tumor_cells, _get_healthy_ref_ids_slice_1,
)
from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX, SAMPLE_ID, SLICE_ID

BASE_DIR = r"D:/thesis-research/"


def train_classifiers(adata, healthy_refs_same_slice, pca=False, n_splits=5, random_state=42):
    """
    Trains classifiers on a reference AnnData and returns the models.
    """
    adata_proc = adata.copy()

    # Identify reference cells using your existing helpers
    if healthy_refs_same_slice:
        healthy_ref_ids = _get_healthy_ref_ids_slice_1(adata_proc)
    else:
        healthy_ref_ids = _get_healthy_ref_ids(adata_proc)
    tumor_ref_ids = _get_tumor_ref_ids(adata_proc)

    healthy_ref_mask = adata_proc.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata_proc.obs_names.isin(tumor_ref_ids)
    ref_mask = np.asarray(healthy_ref_mask | tumor_ref_mask)

    # Standard Preprocessing
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)

    X_ref = _to_dense(adata_proc.X[ref_mask])
    y_ref = np.asarray(healthy_ref_mask[ref_mask]).astype(int)

    # Calculate weights and get model definitions
    n_positive = int((y_ref == 1).sum())
    n_negative = int((y_ref == 0).sum())
    scale_pos_weight = n_negative / max(n_positive, 1)

    if pca:
        models = _get_classifier_models_pca(random_state, scale_pos_weight)
    else:
        models = _get_classifier_models(random_state, scale_pos_weight)

    scoring = _get_scoring()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fitted_models, metrics_df, oof_probs = _fit_models(models, X_ref, y_ref, cv, scoring)

    print("\n=== Training Performance (Cross-Validated) ===")
    print(metrics_df.round(4).to_string())

    return fitted_models, metrics_df, oof_probs



def run_classifiers_compare_and_score(
    adata,
    healthy_refs_same_slice,
    pca = False,
    n_splits=5,
    random_state=42,
):
    """
    Run Logistic Regression, Random Forest, and XGBoost on reference cells,
    compare CV performance, plot logistic-regression PCA boundary, and return
    adata_sub with per-cell healthiness confidence scores.

    Assumes these helper functions already exist:
        _get_healthy_ref_ids(adata)
        _get_tumor_ref_ids(adata)
        _get_tumor_candidates_ids(adata)

    Labels:
        1 = healthy
        0 = tumor

    Returns
    -------
    results : dict with:
        - "metrics_df": CV summary table
        - "diff_vs_logreg_df": difference table vs logistic regression
        - "adata_sub": subset AnnData with score columns in .obs
        - "models": fitted models dict
        - "logreg_gene_coefficients": dataframe of logistic coefficients
    """
    fitted_models, metrics_df, oof_probs = train_classifiers(adata, healthy_refs_same_slice, pca, n_splits, random_state)
    adata_raw = adata.copy()
    adata_proc = adata.copy()

    if healthy_refs_same_slice:
        healthy_ref_ids = _get_healthy_ref_ids_slice_1(adata_proc)
    else:
        healthy_ref_ids = _get_healthy_ref_ids(adata_proc)
    tumor_ref_ids = _get_tumor_ref_ids(adata_proc)

    healthy_ref_mask = adata_proc.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata_proc.obs_names.isin(tumor_ref_ids)
    tumor_candidate_ids = _get_tumor_candidates_ids(adata_proc)
    tumor_candidate_mask = adata_proc.obs_names.isin(tumor_candidate_ids)

    ref_mask = np.asarray(healthy_ref_mask | tumor_ref_mask)
    selected_mask = np.asarray(ref_mask | tumor_candidate_mask)

    # For plotting only
    X_cand = _to_dense(adata_proc.X[tumor_candidate_mask])

    adata_sub = _get_fit_scores(
        fitted_models, oof_probs, adata_raw, selected_mask, healthy_ref_mask, tumor_ref_mask
    )

    X_ref = _to_dense(adata_proc.X[ref_mask])
    y_ref = np.asarray(healthy_ref_mask[ref_mask]).astype(int)

    _logistic_regression_plot_and_coeff(
        fitted_models, adata_proc, X_ref, y_ref, X_cand, random_state
    )
    # ------------------------------------------------------------------
    # Present tables
    # ------------------------------------------------------------------
    print("\n=== Cross-validated performance ===")
    print(metrics_df.round(4).to_string())

    _plot_results(adata_sub)

    slices_dir = pathlib.Path(rf"{BASE_DIR}/resources/cache/")
    slice_paths = sorted(slices_dir.glob("slice_*_adata.h5ad"))
    for slice_path in slice_paths:
        print(f"\n--- Applying models to new slice: {slice_path.name} ---")
        adata_new = ad.read_h5ad(slice_path)
        predict_tumor_on_new_slice(adata_new, fitted_models)


    return {
        "metrics_df": metrics_df,
        "adata_sub": adata_sub,
        "models": fitted_models,
    }


def _to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def _get_classifier_models(random_state, scale_pos_weight):
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
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

def _get_classifier_models_pca(random_state, scale_pos_weight, n_pcs=50):
    """
    Returns an ensemble of models, each wrapped in a pipeline that
    includes Scaling and PCA for noise reduction.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pcs, random_state=random_state)),
            ("clf", LogisticRegression(
                penalty="l2", C=1.0, class_weight="balanced",
                max_iter=5000, solver="liblinear", random_state=random_state
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pcs, random_state=random_state)),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=random_state, n_jobs=-1
            )),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pcs, random_state=random_state)),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                scale_pos_weight=scale_pos_weight, random_state=random_state,
                n_jobs=-1, verbosity=0
            )),
        ]),
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
    oof_probs = {}

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
        # 2. Get OOF Probabilities (Honest scores for reference cells)
        # method='predict_proba' returns [P(0), P(1)]
        probs = cross_val_predict(
            model, X_ref, y_ref, cv=cv, method='predict_proba', n_jobs=-1
        )
        oof_probs[model_name] = probs[:, 1]

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

    return fitted_models, metrics_df, oof_probs


def _get_fit_scores(
    fitted_models, oof_probs, adata_raw, selected_mask, healthy_ref_mask, tumor_ref_mask
):
    adata_sub = adata_raw[selected_mask].copy()

    ref_mask_global = healthy_ref_mask | tumor_ref_mask
    is_ref_in_sub = ref_mask_global[selected_mask]
    is_cand_in_sub = ~is_ref_in_sub

    X_cand_sub = _to_dense(adata_sub.X[is_cand_in_sub])

    group = np.full(adata_sub.n_obs, "Tumor Candidate", dtype=object)
    group[np.where(healthy_ref_mask[selected_mask])[0]] = "Healthy Ref"
    group[np.where(tumor_ref_mask[selected_mask])[0]] = "Tumor Ref"

    adata_sub.obs["classifier_group"] = group
    adata_sub.obs["is_reference"] = is_ref_in_sub
    adata_sub.obs["reference_label"] = np.where(
        adata_sub.obs["classifier_group"] == "Healthy Ref", 1,
        np.where(adata_sub.obs["classifier_group"] == "Tumor Ref", 0, np.nan)
    )

    for model_name, fitted_model in fitted_models.items():
        score_col = f"score_healthy_{model_name}"
        pred_col = f"pred_healthy_{model_name}"

        final_scores = np.zeros(adata_sub.n_obs)
        # OOF for Reference cells
        final_scores[is_ref_in_sub] = oof_probs[model_name]
        # Fresh prediction for Candidates
        final_scores[is_cand_in_sub] = fitted_model.predict_proba(X_cand_sub)[:, 1]

        adata_sub.obs[score_col] = final_scores
        adata_sub.obs[pred_col] = (final_scores >= 0.5).astype(int)

        # 4. Calculate Ensemble Mean (using the stitched scores)
    model_score_cols = [f"score_healthy_{m}" for m in fitted_models.keys()]
    adata_sub.obs["score_healthy_mean"] = adata_sub.obs[model_score_cols].mean(axis=1)
    adata_sub.obs["pred_healthy_mean"] = (adata_sub.obs["score_healthy_mean"] >= 0.5).astype(int)

    # 5. Identify Mismatches (The Audit)
    ref_obs = adata_sub.obs[adata_sub.obs["is_reference"]].copy()
    mismatches = ref_obs[ref_obs["reference_label"] != ref_obs["pred_healthy_mean"]].copy()

    if len(mismatches) > 0:
        mismatches["mismatch_type"] = np.where(
            mismatches["reference_label"] == 1,
            "False Tumor (Labeled Healthy, Predicted Tumor)",
            "False Healthy (Labeled Tumor, Predicted Healthy)"
        )
        print(f"\n--- OOF Audit: Found {len(mismatches)} mismatches in {len(ref_obs)} reference cells ---")
        print(mismatches[["classifier_group", "score_healthy_mean", "mismatch_type"]].to_string())
    else:
        print("\n--- OOF Audit: No mismatches found! Reference labels are consistent. ---")

    return adata_sub


def _logistic_regression_plot_and_coeff(
    fitted_models, adata_proc, X_ref, y_ref, X_cand, random_state
):
    model = fitted_models["logistic_regression"]

    # Check if the pipeline has a PCA step
    if "pca" in model.named_steps:
        pca_step = model.named_steps["pca"]
        logreg_clf = model.named_steps["clf"]
        # Project PCA coefficients back to gene space: (1, 50) @ (50, Genes)
        gene_weights = np.dot(logreg_clf.coef_, pca_step.components_)[0]
    else:
        # Standard pipeline (scaler + clf)
        logreg_clf = model.named_steps["clf"]
        gene_weights = logreg_clf.coef_[0]

    gene_df = pd.DataFrame({
        "gene": adata_proc.var_names,
        "coef": gene_weights
    }).sort_values("coef", ascending=False)

    top_healthy = gene_df.head(30)
    top_tumor = gene_df.sort_values("coef", ascending=True).head(30)

    print("\n=== Top genes associated with healthy (positive coefficients) ===")
    print(top_healthy.to_string(index=False))
    print("\n=== Top genes associated with tumor (negative coefficients) ===")
    print(top_tumor.to_string(index=False))

    # keep the explicit ordering: refs first, candidates after
    X_vis_all = np.vstack([X_ref, X_cand])

    scaler_vis = StandardScaler()
    X_vis_all_scaled = scaler_vis.fit_transform(X_vis_all)

    pca_vis = PCA(n_components=2, random_state=random_state)
    X_vis_all_2d = pca_vis.fit_transform(X_vis_all_scaled)

    n_ref = X_ref.shape[0]
    X_ref_2d = X_vis_all_2d[:n_ref]
    X_cand_2d = X_vis_all_2d[n_ref:]

    df_ref_plot = pd.DataFrame(X_ref_2d, columns=["PC1", "PC2"])
    df_ref_plot["Status"] = np.where(y_ref == 1, "Healthy Ref", "Tumor Ref")

    df_cand = pd.DataFrame(X_cand_2d, columns=["PC1", "PC2"])

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=df_cand,
        x="PC1",
        y="PC2",
        color="lightgray",
        alpha=0.35,
        s=12,
        label="Candidates",
        ax=ax,
    )

    sns.scatterplot(
        data=df_ref_plot,
        x="PC1",
        y="PC2",
        hue="Status",
        palette={"Healthy Ref": "dodgerblue", "Tumor Ref": "firebrick"},
        s=40,
        edgecolor="black",
        alpha=0.85,
        ax=ax,
    )

    clf_l2_plot = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=random_state,
    )
    clf_l2_plot.fit(X_ref_2d, y_ref)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf_l2_plot.predict_proba(grid)[:, 1].reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    ax.set_title("Logistic Regression Decision Boundary (PCA visualization)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def _plot_results(adata_sub):
    map = {
        "Logistic Regression": "pred_healthy_logistic_regression",
        "Random Forest": "pred_healthy_random_forest",
        "XGBoost": "pred_healthy_xgboost",
    }

    for model_name, pred_col in map.items():
        healthy_cells = adata_sub.obs_names[adata_sub.obs[pred_col] == 1]
        _plot_tumor_cells(healthy_cells, model_name)


def predict_tumor_on_new_slice(adata_new, fitted_models):
    """
    Applies pre-trained models to a new slice.
    """
    adata_target = adata_new.copy()

    # 2. Preprocess exactly like the training set
    sc.pp.normalize_total(adata_target, target_sum=1e4)
    sc.pp.log1p(adata_target)

    # Convert to dense matrix for sklearn/xgboost
    X_target = _to_dense(adata_target.X)

    # 3. Generate Predictions
    # We use the full models here (no OOF needed for new slices)
    results = {}
    for name, model in fitted_models.items():
        results[f"score_healthy_{name}"] = model.predict_proba(X_target)[:, 1]

    # 4. Add scores back to the ORIGINAL adata_new object
    for col, scores in results.items():
        adata_new.obs[col] = scores
        # Label as healthy (1) or tumor (0) based on 0.5 threshold
        adata_new.obs[col.replace("score", "pred")] = (scores >= 0.5).astype(int)

    # 5. Calculate Ensemble Mean
    score_cols = [f"score_healthy_{m}" for m in fitted_models.keys()]
    adata_new.obs["score_healthy_mean"] = adata_new.obs[score_cols].mean(axis=1)
    adata_new.obs["pred_healthy_mean"] = (adata_new.obs["score_healthy_mean"] >= 0.5).astype(int)

    for name, model in fitted_models.items():
        _plot_tumor_spatial_refined(adata_new, name, adata_new.uns[SAMPLE_ID], adata_new.uns[SLICE_ID])
    return adata_new


def _plot_tumor_spatial_refined(adata, classifier_name, sample_id, slice_id):
    """
    Plots tumor cells spatially across the entire slice based on ML predictions.
    """
    # 1. Prepare coordinates
    plot_obs = adata.obs.copy()

    # Apply coordinate transformations
    plot_obs[CENTER_Y_GLOBAL_PX] = -plot_obs[CENTER_Y_GLOBAL_PX]
    plot_obs[CENTER_X_GLOBAL_PX] = -plot_obs[CENTER_X_GLOBAL_PX]

    # 2. Define the Global Tumor logic
    # Assuming 1 = Healthy and 0 = Tumor (your current setup)
    # If a cell is NOT predicted healthy, it's a tumor candidate.
    pred_col = f"pred_healthy_{classifier_name}"
    if pred_col in plot_obs.columns:
        is_tumor_ml = plot_obs[pred_col] == 0
    else:
        is_tumor_ml = plot_obs["pred_healthy_mean"] == 0

    # Optional: If you want to see where SingleR and ML disagree:
    # is_tumor_base = plot_obs["predicted_cell_type"] == "Tumor"

    # ALL cells predicted as tumor by the ML model
    cells_to_keep = is_tumor_ml
    is_background = ~cells_to_keep

    # 3. Plotting
    plt.figure(figsize=(10, 10))

    # Layer 1: Background (Predicted Healthy)
    plt.scatter(
        plot_obs.loc[is_background, CENTER_X_GLOBAL_PX],
        plot_obs.loc[is_background, CENTER_Y_GLOBAL_PX],
        c="#CFCFCF",
        s=0.7,
        alpha=0.4,
        edgecolors="none",
    )

    # Layer 2: Predicted Tumor Cells (Everywhere in the slice)
    plt.scatter(
        plot_obs.loc[cells_to_keep, CENTER_X_GLOBAL_PX],
        plot_obs.loc[cells_to_keep, CENTER_Y_GLOBAL_PX],
        c="red",
        s=1.5,
        alpha=0.8,
        edgecolors="none",
        label=f"Sample {sample_id} slice {slice_id} ML Predicted Tumor"
    )

    # 4. Analytics for your Thesis
    # How many NEW tumor cells did we find that SingleR missed?
    n_total_ml = is_tumor_ml.sum()

    plt.title(
        f"Sample {sample_id} slice {slice_id} Global Spatial Mapping: {classifier_name}\n"
        f"Total ML Tumor: {n_total_ml}",
        fontsize=14
    )
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right", markerscale=5)
    plt.show()

if __name__ == "__main__":
    print("=== Running classifiers with healthy refs from the same slice with PCA ===")
    run_classifiers_compare_and_score(
        adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
        healthy_refs_same_slice=True,
        pca = True
    )
    print("\n\n=== Running classifiers with healthy refs from control slice with PCA ===")
    run_classifiers_compare_and_score(
        adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
        healthy_refs_same_slice=False,
        pca=True

    )
    print("=== Running classifiers with healthy refs from the same slice NO PCA ===")
    run_classifiers_compare_and_score(
        adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
        healthy_refs_same_slice=True,
    )
    print("\n\n=== Running classifiers with healthy refs from control slice NO PCA ===")
    run_classifiers_compare_and_score(
        adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
        healthy_refs_same_slice=False,
    )
