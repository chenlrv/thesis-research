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
    _plot_tumor_cells,
)
from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX, SAMPLE_ID, SLICE_ID

BASE_DIR = r"D:/thesis-research/"

PRED_HEALTHY_THRESHOLD = 0.75

def train_classifiers(adata, sample_id, pca=False, n_splits=5, random_state=42):
    """
    Trains classifiers on a reference AnnData and returns the models.
    """
    adata_proc = adata.copy()
    healthy_ref_ids = _get_healthy_ref_ids(sample_id)
    tumor_ref_ids = _get_tumor_ref_ids(sample_id)

    healthy_ref_mask = adata_proc.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata_proc.obs_names.isin(tumor_ref_ids)
    ref_mask = np.asarray(healthy_ref_mask | tumor_ref_mask)

    # Standard Preprocessing
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)

    X_ref = _to_dense(adata_proc.X[ref_mask])
    y_ref = np.asarray(healthy_ref_mask[ref_mask]).astype(int)

    # Calculate weights and get model definitions
    n_positive = int((y_ref == 1).sum()) # healthy
    n_negative = int((y_ref == 0).sum()) # tumor
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
    slide_id: str,
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
    adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_{slide_id}_adata.h5ad")
    fitted_models, metrics_df, oof_probs = train_classifiers(adata, slide_id, pca, n_splits, random_state)
    adata_raw = adata.copy()
    adata_proc = adata.copy()

    healthy_ref_ids = _get_healthy_ref_ids(slide_id)
    tumor_ref_ids = _get_tumor_ref_ids(slide_id)

    healthy_ref_mask = adata_proc.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata_proc.obs_names.isin(tumor_ref_ids)
    tumor_candidate_ids = _get_tumor_candidates_ids(slide_id)
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

    plot_slice_id = {"L321": 1, "L34": 5}
    _plot_results(slide_id, plot_slice_id[slide_id], adata_sub, pca)

    slices_dir = pathlib.Path(rf"{BASE_DIR}/resources/cache/")
    slide_map = {"L321": [2,3], "L34": [5,6]}
    for slide_id in slide_map:
        for slice_num in slide_map[slide_id]:
            slice_path = slices_dir / f"slice_{slice_num}_adata.h5ad"
            if slice_path.exists():
                print(f"\n--- Applying models to new slice: {slice_path.name} ---")
                slice_adata = ad.read_h5ad(slice_path)
                slice_id = slice_adata.uns[SLICE_ID]
                annotation_path =   rf"{BASE_DIR}/outputs/cell_annotation/{slide_id}/05/{slice_num}/slice_{slice_num}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
                if pathlib.Path(annotation_path).exists():
                    annotation_df = pd.read_csv(annotation_path)
                    tumor_cells = _get_tumor_candidates_ids(slide_id, annotation_df)
                    predict_tumor_on_new_slice(slice_adata, tumor_cells, fitted_models)
                else:
                    continue
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

        preds = (probs[:, 1] >= PRED_HEALTHY_THRESHOLD).astype(int)

        audit_df = pd.DataFrame({
            "true_label": y_ref,  # 1=healthy, 0=tumor
            "oof_prob_healthy": probs[:, 1],
            "oof_pred_label": preds,
        })
        audit_df["true_label_name"] = audit_df["true_label"].map({1: "Healthy Ref", 0: "Tumor Ref"})
        audit_df["oof_pred_name"] = audit_df["oof_pred_label"].map({1: "Healthy", 0: "Tumor"})
        audit_df["correct"] = audit_df["true_label"] == audit_df["oof_pred_label"]

        n_correct_healthy = ((audit_df["true_label"] == 1) & (audit_df["oof_pred_label"] == 1)).sum()
        n_correct_tumor = ((audit_df["true_label"] == 0) & (audit_df["oof_pred_label"] == 0)).sum()

        print(f"Model {model_name} correct healthy labeling: {n_correct_healthy} out of {(y_ref == 1).sum()} healthy reference cells")
        print(f"Model {model_name} correct tumor labeling: {n_correct_tumor} out of {(y_ref == 0).sum()} tumor reference cells")

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

        plot_spatial_continuous_scores(
            adata_sub,
            model_names=(model_name,),
            x_col="CenterX_global_px",
            y_col="CenterY_global_px",
            tumor_score=True,  # plots 1 - score_healthy
            only_candidates=False,  # or True if you want only candidate cells colored
            suptitle="Continuous tumor score across models"
        )

        adata_sub.obs[pred_col] = (final_scores >= PRED_HEALTHY_THRESHOLD).astype(int)

        ref_obs = adata_sub.obs[adata_sub.obs["is_reference"]].copy()

        n_healthy_as_tumor = (
                (ref_obs["reference_label"] == 1) &
                (ref_obs[f"pred_healthy_{model_name}"] == 0)
        ).sum()

        n_tumor_as_healthy = (
                (ref_obs["reference_label"] == 0) &
                (ref_obs[f"pred_healthy_{model_name}"] == 1)
        ).sum()

        print(f"Model {model_name}: Healthy refs predicted as tumor: {n_healthy_as_tumor} out of {(healthy_ref_mask.sum())} healthy reference cells")
        print(f"Model {model_name}: Tumor refs predicted as healthy: {n_tumor_as_healthy} out of {(tumor_ref_mask.sum())} tumor reference cells")

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
            "Healthy ref predicted as Tumor",
            "Tumor ref predicted as Healthy"
        )
        print(f"\n--- OOF Audit for pred_healthy_mean: Found {len(mismatches)} mismatches in {len(ref_obs)} reference cells ---")
        n_healthy_as_tumor = (
                (ref_obs["reference_label"] == 1) &
                (ref_obs["pred_healthy_mean"] == 0)
        ).sum()

        n_tumor_as_healthy = (
                (ref_obs["reference_label"] == 0) &
                (ref_obs["pred_healthy_mean"] == 1)
        ).sum()

        print(f"Healthy refs predicted as tumor: {n_healthy_as_tumor}")
        print(f"Tumor refs predicted as healthy: {n_tumor_as_healthy}")
    else:
        print("\n--- OOF Audit: No mismatches found! Reference labels are consistent. ---")

    return adata_sub


def _logistic_regression_plot_and_coeff(
    fitted_models, adata_proc, X_ref, y_ref, X_cand, random_state, model_label=""
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

    prefix = f"[{model_label}] " if model_label else ""
    print(f"\n=== {prefix}Top genes associated with healthy (positive coefficients) ===")
    print(top_healthy.to_string(index=False))
    print(f"\n=== {prefix}Top genes associated with tumor (negative coefficients) ===")
    print(top_tumor.to_string(index=False))

    # Bar plot of top gene coefficients
    fig_genes, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(14, 8))
    fig_genes.suptitle(model_label if model_label else "Logistic Regression", fontsize=13, fontweight="bold")

    ax_h.barh(top_healthy["gene"][::-1], top_healthy["coef"][::-1], color="dodgerblue")
    ax_h.set_xlabel("Coefficient")
    ax_h.set_title("Top 30 Healthy-associated genes")
    ax_h.axvline(0, color="black", linewidth=0.8)

    ax_t.barh(top_tumor["gene"][::-1], top_tumor["coef"][::-1], color="firebrick")
    ax_t.set_xlabel("Coefficient")
    ax_t.set_title("Top 30 Tumor-associated genes")
    ax_t.axvline(0, color="black", linewidth=0.8)

    fig_genes.tight_layout()
    plt.show()

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

    ax.set_title(f"{prefix}Decision Boundary (PCA visualization)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


def _plot_results(slide_id, slice_id, adata_sub, pca):
    map = {
        "Logistic Regression": "pred_healthy_logistic_regression",
        "Random Forest": "pred_healthy_random_forest",
        "XGBoost": "pred_healthy_xgboost",
    }

    for model_name, pred_col in map.items():
        healthy_cells = adata_sub.obs_names[adata_sub.obs[pred_col] == 1]
        _plot_tumor_cells(slide_id, slice_id, healthy_cells, model_name, pca)


def predict_tumor_on_new_slice(slice_adata, tumor_cells, fitted_models):
    """
    Applies pre-trained models to a new slice.
    """
    is_tumor_singler = slice_adata.obs_names.isin(tumor_cells)
    adata_candidates = slice_adata[is_tumor_singler].copy()

    # 2. Preprocess exactly like the training set
    sc.pp.normalize_total(adata_candidates, target_sum=1e4)
    sc.pp.log1p(adata_candidates)

    # Convert to dense matrix for sklearn/xgboost
    X_target = _to_dense(adata_candidates.X)

    # 3. Generate Predictions
    # We use the full models here (no OOF needed for new slices)
    results = {}
    for classifier_name, model in fitted_models.items():
        scores = model.predict_proba(X_target)[:, 1]

        # Initialize the full adata column to 'Healthy' (1.0)
        score_col = f"score_healthy_{classifier_name}"
        slice_adata.obs[score_col] = 1.0

        slice_adata.obs.loc[is_tumor_singler, score_col] = scores

        # Labeling (1=Healthy, 0=Tumor)
        pred_col = f"pred_healthy_{classifier_name}"
        slice_adata.obs[pred_col] = (slice_adata.obs[score_col] >= PRED_HEALTHY_THRESHOLD).astype(int)


    # 5. Calculate Ensemble Mean
    score_cols = [f"score_healthy_{m}" for m in fitted_models.keys()]
    slice_adata.obs["score_healthy_mean"] = slice_adata.obs[score_cols].mean(axis=1)
    slice_adata.obs["pred_healthy_mean"] = (slice_adata.obs["score_healthy_mean"] >= 0.5).astype(int)

    for classifier_name, model in fitted_models.items():
        _plot_tumor_spatial_refined(slice_adata, tumor_cells, classifier_name, slice_adata.uns[SAMPLE_ID], slice_adata.uns[SLICE_ID])
    return slice_adata


def _plot_tumor_spatial_refined(adata, tumor_cells, classifier_name, sample_id, slice_id):
    """
    Plots tumor cells spatially across the entire slice based on ML predictions.
    """
    # 1. Prepare coordinates
    plot_obs = adata.obs.copy()

    # Apply coordinate transformations
    plot_obs[CENTER_Y_GLOBAL_PX] = plot_obs[CENTER_Y_GLOBAL_PX]
    plot_obs[CENTER_X_GLOBAL_PX] = plot_obs[CENTER_X_GLOBAL_PX]

    # 2. Define the Global Tumor logic
    # Assuming 1 = Healthy and 0 = Tumor (your current setup)
    # If a cell is NOT predicted healthy, it's a tumor candidate.
    is_tumor_singler = plot_obs.index.isin(tumor_cells)
    pred_col = f"pred_healthy_{classifier_name}"
    if pred_col in plot_obs.columns:
        is_tumor_ml = plot_obs[pred_col] == 0
    else:
        is_tumor_ml = plot_obs["pred_healthy_mean"] == 0

    cells_rejected = is_tumor_singler & (~is_tumor_ml)

    # Optional: If you want to see where SingleR and ML disagree:
    # is_tumor_base = plot_obs["predicted_cell_type"] == "Tumor"

    # ALL cells predicted as tumor by the ML model
    cells_to_keep = is_tumor_singler & is_tumor_ml
    is_background = ~(cells_to_keep | cells_rejected)

    # 3. Plotting
    plt.figure(figsize=(10, 10))

    # Layer 1: Background
    plt.scatter(
        plot_obs.loc[is_background, CENTER_X_GLOBAL_PX],
        plot_obs.loc[is_background, CENTER_Y_GLOBAL_PX],
        c="#E0E0E0", s=0.7, alpha=0.3, edgecolors="none"
    )

    # Layer 2: Mislabeled/Rejected (Gray-Blue) - "The Noise"
    plt.scatter(
        plot_obs.loc[cells_rejected, CENTER_X_GLOBAL_PX],
        plot_obs.loc[cells_rejected, CENTER_Y_GLOBAL_PX],
        c="#607D8B", s=1.2, alpha=0.5, label="ML Rejected (Mislabeled)"
    )

    # Layer 3: Refined Tumor (Red) - "The Signal"
    plt.scatter(
        plot_obs.loc[cells_to_keep, CENTER_X_GLOBAL_PX],
        plot_obs.loc[cells_to_keep, CENTER_Y_GLOBAL_PX],
        c="red", s=2.0, alpha=0.9, label="Refined Tumor"
    )

    n_tumor_candidates = is_tumor_singler.sum()
    n_kept = cells_to_keep.sum()
    n_rejected = cells_rejected.sum()

    plt.title(
        f"Sample {sample_id} Slice {slice_id} | Refinement: {classifier_name}\n"
        f"Refined: {n_tumor_candidates} | Rejected Mislabeled: {n_rejected}| Tumor Cells Left: {n_kept}",
        fontsize=14
    )
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right", markerscale=4)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def plot_spatial_continuous_scores(
    adata,
    model_names=("logistic_regression", "random_forest", "xgboost"),
    x_col="CenterX_global_px",
    y_col="CenterY_global_px",
    point_size_bg=1,
    point_size_score=6,
    alpha_bg=0.15,
    alpha_score=0.9,
    cmap="Reds",
    flip_y=True,
    figsize_per_panel=(6, 6),
    tumor_score=True,
    vmin=0.0,
    vmax=1.0,
    only_candidates=False,
    candidate_col="classifier_group",
    candidate_value="Tumor Candidate",
    suptitle=None,
):
    """
    Plot continuous spatial scores for each model.

    Parameters
    ----------
    adata : AnnData
    model_names : tuple/list
        Model names matching obs columns score_healthy_<model>
    x_col, y_col : str
        Spatial coordinate columns in adata.obs
    tumor_score : bool
        If True, plot 1 - score_healthy_<model>
        If False, plot score_healthy_<model>
    only_candidates : bool
        If True, plot scores only for candidate cells
    """

    x = adata.obs[x_col].values
    y = adata.obs[y_col].values

    if flip_y:
        y_plot = -y
    else:
        y_plot = y

    if only_candidates:
        mask = adata.obs[candidate_col] == candidate_value
    else:
        mask = np.ones(adata.n_obs, dtype=bool)

    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(figsize_per_panel[0] * n_models, figsize_per_panel[1]))

    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        score_col = f"score_healthy_{model_name}"
        if score_col not in adata.obs.columns:
            raise ValueError(f"{score_col} not found in adata.obs")

        healthy_score = adata.obs[score_col].values
        score = 1 - healthy_score if tumor_score else healthy_score

        # background: all cells
        ax.scatter(
            x, y_plot,
            s=point_size_bg,
            c="lightgrey",
            alpha=alpha_bg,
            linewidths=0,
            rasterized=True
        )

        # overlay: selected cells with continuous score
        sc = ax.scatter(
            x[mask], y_plot[mask],
            s=point_size_score,
            c=score[mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha_score,
            linewidths=0,
            rasterized=True
        )

        label = "Tumor score" if tumor_score else "Healthy score"
        ax.set_title(f"{model_name}\n{label}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    plt.show()

def run_classifiers_joint(pca=False, n_splits=5, random_state=42):
    """
    Trains classifiers jointly on reference cells from both L321 and L34,
    then scores tumor candidates per slide using the jointly trained models.

    Healthy refs come from control slices (3=L321, 4=L34).
    Tumor refs come from high-confidence tumor slices (1=L321, 5=L34).
    Normalization is done per-sample before concatenation to reduce library-size batch effects.
    OOF probabilities are mapped back per-slide so reference-cell scores remain honest.
    """
    slide_ids = ["L321", "L34"]
    plot_slice_ids = {"L321": 1, "L34": 5}

    # 1. Load and normalize per-sample before concat
    adatas_raw = {}
    adatas_proc = {}
    for slide_id in slide_ids:
        adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_{slide_id}_adata.h5ad")
        adatas_raw[slide_id] = adata
        proc = adata.copy()
        sc.pp.normalize_total(proc, target_sum=1e4)
        sc.pp.log1p(proc)
        adatas_proc[slide_id] = proc

    # 2. Collect reference IDs from both slides
    healthy_ref_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ref_ids = {sid: _get_tumor_ref_ids(sid) for sid in slide_ids}
    all_healthy = set().union(*healthy_ref_ids.values())
    all_tumor = set().union(*tumor_ref_ids.values())

    print("\n=== Joint Reference Pool ===")
    for sid in slide_ids:
        print(f"  {sid}: {len(healthy_ref_ids[sid])} healthy refs, {len(tumor_ref_ids[sid])} tumor refs")
    print(f"  Total: {len(all_healthy)} healthy, {len(all_tumor)} tumor")

    # 3. Concatenate normalized adatas (inner join = common genes only)
    adata_joint = ad.concat(adatas_proc, join="inner", label="slide_id")
    joint_var_names = adata_joint.var_names.tolist()
    print(f"\nJoint adata: {adata_joint.n_obs} cells x {adata_joint.n_vars} genes")

    # 4. Build joint ref masks and extract training data
    healthy_ref_mask_joint = adata_joint.obs_names.isin(all_healthy)
    tumor_ref_mask_joint = adata_joint.obs_names.isin(all_tumor)
    ref_mask_joint = np.asarray(healthy_ref_mask_joint | tumor_ref_mask_joint)

    X_ref = _to_dense(adata_joint.X[ref_mask_joint])
    y_ref = np.asarray(healthy_ref_mask_joint[ref_mask_joint]).astype(int)

    n_pos = int((y_ref == 1).sum())
    n_neg = int((y_ref == 0).sum())
    print(f"\nTraining on {n_pos} healthy + {n_neg} tumor ref cells across both slides")
    scale_pos_weight = n_neg / max(n_pos, 1)

    # 5. Train jointly
    models = _get_classifier_models_pca(random_state, scale_pos_weight) if pca else _get_classifier_models(random_state, scale_pos_weight)
    scoring = _get_scoring()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fitted_models, metrics_df, oof_probs = _fit_models(models, X_ref, y_ref, cv, scoring)

    print("\n=== Joint Training Performance (Cross-Validated) ===")
    print(metrics_df.round(4).to_string())

    # 6. Map OOF probs to obs names so we can extract per-slide subsets in the correct order
    ref_obs_names = adata_joint.obs_names[ref_mask_joint]
    oof_by_cell = {
        model_name: dict(zip(ref_obs_names, probs))
        for model_name, probs in oof_probs.items()
    }

    # 7. Score candidates per slide
    results = {}
    for slide_id in slide_ids:
        print(f"\n--- Scoring candidates for {slide_id} ---")
        adata_proc = adatas_proc[slide_id]

        h_mask = adata_proc.obs_names.isin(healthy_ref_ids[slide_id])
        t_mask = adata_proc.obs_names.isin(tumor_ref_ids[slide_id])
        cand_ids = _get_tumor_candidates_ids(slide_id)
        cand_mask = adata_proc.obs_names.isin(cand_ids)
        ref_mask = np.asarray(h_mask | t_mask)
        selected_mask = np.asarray(ref_mask | cand_mask)

        # OOF probs for this slide's ref cells, in adata_proc order (preserves alignment)
        ref_obs_slide = adata_proc.obs_names[ref_mask]
        oof_probs_slide = {
            model_name: np.array([prob_map[name] for name in ref_obs_slide])
            for model_name, prob_map in oof_by_cell.items()
        }

        # Pass adata_proc (normalized) so candidate predictions use the same scale as training
        adata_sub = _get_fit_scores(
            fitted_models, oof_probs_slide, adata_proc, selected_mask, h_mask, t_mask
        )

        _plot_results(slide_id, plot_slice_ids[slide_id], adata_sub, pca)
        results[slide_id] = adata_sub

    # 8. Logistic regression PCA visualization + gene coefficients for L321 slice 1
    X_ref_viz = _to_dense(adata_joint.X[ref_mask_joint])
    cand_mask_L321 = adatas_proc["L321"].obs_names.isin(_get_tumor_candidates_ids("L321"))
    X_cand_viz = _to_dense(adatas_proc["L321"].X[np.asarray(cand_mask_L321)])
    y_ref_viz = np.asarray(healthy_ref_mask_joint[ref_mask_joint]).astype(int)
    _logistic_regression_plot_and_coeff(
        fitted_models, adatas_proc["L321"], X_ref_viz, y_ref_viz, X_cand_viz, random_state
    )

    # 9. Apply jointly trained models to all remaining slices (L321→2,3; L34→4,6)
    remaining_slide_slices = {"L321": [2, 3], "L34": [4, 6]}  # slices 1 and 5 already scored above
    slices_dir = pathlib.Path(rf"{BASE_DIR}/resources/cache/")

    for slide_id, slice_nums in remaining_slide_slices.items():
        for slice_num in slice_nums:
            slice_path = slices_dir / f"slice_{slice_num}_adata.h5ad"
            annotation_path = pathlib.Path(
                rf"{BASE_DIR}/outputs/cell_annotation/{slide_id}/05/{slice_num}"
                rf"/slice_{slice_num}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
            )
            if not slice_path.exists() or not annotation_path.exists():
                print(f"Skipping {slide_id} slice {slice_num}: missing data")
                continue
            print(f"\n--- Applying joint models to {slide_id} slice {slice_num} ---")
            slice_adata = ad.read_h5ad(slice_path)
            # Align genes to joint training gene set
            slice_adata = slice_adata[:, joint_var_names].copy()
            annotation_df = pd.read_csv(annotation_path)
            tumor_cells = _get_tumor_candidates_ids(slide_id, annotation_df)
            print(f"  {len(tumor_cells)} tumor candidates")
            predict_tumor_on_new_slice(slice_adata, tumor_cells, fitted_models)

    return {
        "metrics_df": metrics_df,
        "models": fitted_models,
        "results": results,
    }


def _plot_threshold_calibration(y_true, oof_prob, current_thresh=0.7):
    """
    Two-panel threshold calibration plot for a binary OOF probability vector.

    Left panel:  Precision-Recall curve with the operating point marked.
    Right panel: Precision, Recall, and F1 as a function of threshold,
                 with the current threshold and the max-F1 threshold marked.

    y_true : 1=healthy, 0=tumor  (as used throughout this file)
    oof_prob : P(healthy) from OOF cross-validation
    """
    from sklearn.metrics import precision_recall_curve

    # sklearn's precision_recall_curve treats the *higher* score as the positive class.
    # Here 1=healthy is "positive", so oof_prob (P(healthy)) is already correct.
    precision, recall, thresholds = precision_recall_curve(y_true, oof_prob)
    # thresholds has length n-1; precision/recall have length n (last element is trivial)
    pr_thresh = thresholds          # shape (n-1,)
    pr_prec   = precision[:-1]
    pr_rec    = recall[:-1]
    f1_vals   = np.where(
        (pr_prec + pr_rec) > 0,
        2 * pr_prec * pr_rec / (pr_prec + pr_rec),
        0.0,
    )

    best_idx   = int(np.argmax(f1_vals))
    best_thresh = float(pr_thresh[best_idx])
    best_f1     = float(f1_vals[best_idx])

    # Index closest to current operating threshold
    op_idx = int(np.argmin(np.abs(pr_thresh - current_thresh)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LogReg + PCA — Threshold Calibration (OOF)", fontsize=13, fontweight="bold")

    # ── Left: Precision-Recall curve ────────────────────────────────────────
    ax1.plot(pr_rec, pr_prec, color="steelblue", linewidth=2, label="PR curve")
    ax1.scatter(
        pr_rec[op_idx], pr_prec[op_idx],
        c="red", s=120, zorder=5, label=f"threshold={current_thresh} (operating)"
    )
    ax1.scatter(
        pr_rec[best_idx], pr_prec[best_idx],
        c="green", s=120, marker="*", zorder=5,
        label=f"best F1 threshold={best_thresh:.2f} (F1={best_f1:.3f})"
    )
    ax1.set_xlabel("Recall (healthy)")
    ax1.set_ylabel("Precision (healthy)")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Right: Precision / Recall / F1 vs threshold ──────────────────────────
    ax2.plot(pr_thresh, pr_prec, label="Precision", color="dodgerblue")
    ax2.plot(pr_thresh, pr_rec,  label="Recall",    color="tomato")
    ax2.plot(pr_thresh, f1_vals, label="F1",        color="seagreen", linewidth=2)
    ax2.axvline(current_thresh, color="red",   linestyle="--", linewidth=1.5,
                label=f"operating threshold={current_thresh}")
    ax2.axvline(best_thresh,    color="green", linestyle="--", linewidth=1.5,
                label=f"best F1 threshold={best_thresh:.2f}")
    ax2.set_xlabel("Threshold (P(healthy))")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision / Recall / F1 vs Threshold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    print(f"\n=== LogReg+PCA Threshold Calibration ===")
    print(f"  Operating threshold : {current_thresh:.2f}  ->  "
          f"precision={pr_prec[op_idx]:.4f}  recall={pr_rec[op_idx]:.4f}  F1={f1_vals[op_idx]:.4f}")
    print(f"  Best-F1 threshold   : {best_thresh:.2f}  ->  "
          f"precision={pr_prec[best_idx]:.4f}  recall={pr_rec[best_idx]:.4f}  F1={best_f1:.4f}")

    plt.tight_layout()
    out_path = pathlib.Path(BASE_DIR) / "resources" / "cache" / "threshold_calibration.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Calibration plot saved to: {out_path}")
    plt.show()


def run_model_comparison(n_splits=5, random_state=42, k=15, prob_thresh=0.75, n_pcs=50):
    """
    Compares 4 tumor classifiers on the joint L321+L34 reference pool:
      1. LogReg             - StandardScaler + LogisticRegression (full gene space)
      2. LogReg + PCA       - StandardScaler + PCA(50) + LogisticRegression
      3. LogReg + KNN (PCA) - PCA space, is_healthy = (logreg >= thresh) | (knn_frac >= thresh)
      4. XGBoost            - XGBClassifier (full gene space)

    Outputs:
      - OOF metrics table and bar chart at operating threshold
      - Top-gene bar charts and PCA boundary plots for both logistic models
      - Tumor candidate counts per slice
      - Spatial comparison: 4-panel plot per slice
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        precision_recall_curve, roc_curve,
    )

    slide_ids = ["L321", "L34"]
    primary_slices = {"L321": 1, "L34": 5}
    remaining_slices_map = {"L321": [2, 3], "L34": [4, 6]}

    # ── 1. Load + normalize per-sample ──────────────────────────────────────
    adatas_proc = {}
    for sid in slide_ids:
        adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_{sid}_adata.h5ad")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adatas_proc[sid] = adata

    healthy_ref_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ref_ids   = {sid: _get_tumor_ref_ids(sid)   for sid in slide_ids}
    all_healthy = set().union(*healthy_ref_ids.values())
    all_tumor   = set().union(*tumor_ref_ids.values())

    print("\n=== Joint Reference Pool ===")
    for sid in slide_ids:
        print(f"  {sid}: {len(healthy_ref_ids[sid])} healthy, {len(tumor_ref_ids[sid])} tumor")
    print(f"  Total: {len(all_healthy)} healthy + {len(all_tumor)} tumor")

    adata_joint = ad.concat(adatas_proc, join="inner", label="slide_id")
    joint_var_names = adata_joint.var_names.tolist()

    healthy_mask_j = adata_joint.obs_names.isin(all_healthy)
    tumor_mask_j   = adata_joint.obs_names.isin(all_tumor)
    ref_mask_j     = np.asarray(healthy_mask_j | tumor_mask_j)

    X_ref_raw = _to_dense(adata_joint.X[ref_mask_j])
    y_ref = np.asarray(healthy_mask_j[ref_mask_j]).astype(int)

    n_pos = int((y_ref == 1).sum())
    n_neg = int((y_ref == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # ── 2. PCA projection for KNN model (fit on ref cells only) ─────────────
    scaler_knn = StandardScaler()
    X_ref_scaled = np.clip(scaler_knn.fit_transform(X_ref_raw), -10, 10)
    n_comps_eff = min(n_pcs, X_ref_raw.shape[0] - 1, X_ref_raw.shape[1] - 1)
    pca_knn = PCA(n_components=n_comps_eff, random_state=random_state)
    X_ref_pca = pca_knn.fit_transform(X_ref_scaled)

    # ── 3. Model definitions ─────────────────────────────────────────────────
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=5000, solver="liblinear", random_state=random_state)),
    ])
    logreg_pca_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pcs, random_state=random_state)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=5000, solver="liblinear", random_state=random_state)),
    ])
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=random_state,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    logreg_knn_clf = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=random_state
    )

    # ── 4. OOF probabilities ─────────────────────────────────────────────────
    print("\nOOF: LogReg...")
    oof_logreg = cross_val_predict(logreg_pipe, X_ref_raw, y_ref, cv=cv, method="predict_proba")[:, 1]

    print("OOF: LogReg + PCA...")
    oof_logreg_pca = cross_val_predict(logreg_pca_pipe, X_ref_raw, y_ref, cv=cv, method="predict_proba")[:, 1]

    print("OOF: XGBoost...")
    oof_xgb = cross_val_predict(xgb_model, X_ref_raw, y_ref, cv=cv, method="predict_proba")[:, 1]

    print("OOF: LogReg + KNN (PCA)...")
    oof_lr_pca = cross_val_predict(logreg_knn_clf, X_ref_pca, y_ref, cv=cv, method="predict_proba")[:, 1]
    knn_oof = np.zeros(len(X_ref_pca))
    for train_idx, val_idx in cv.split(X_ref_pca, y_ref):
        nn_fold = NearestNeighbors(n_neighbors=min(k, len(train_idx)))
        nn_fold.fit(X_ref_pca[train_idx])
        knn_oof[val_idx] = y_ref[train_idx][
            nn_fold.kneighbors(X_ref_pca[val_idx], return_distance=False)
        ].mean(axis=1)
    oof_knn_score = (oof_lr_pca + knn_oof) / 2
    oof_knn_pred  = ((oof_lr_pca >= prob_thresh) | (knn_oof >= prob_thresh)).astype(int)

    # ── 5. Fit final models ──────────────────────────────────────────────────
    logreg_pipe.fit(X_ref_raw, y_ref)
    logreg_pca_pipe.fit(X_ref_raw, y_ref)
    xgb_model.fit(X_ref_raw, y_ref)
    logreg_knn_clf.fit(X_ref_pca, y_ref)
    nn_full = NearestNeighbors(n_neighbors=min(k, X_ref_pca.shape[0]))
    nn_full.fit(X_ref_pca)

    # ── 6. Top genes + PCA boundary plots ───────────────────────────────────
    cand_mask_L321 = adatas_proc["L321"].obs_names.isin(_get_tumor_candidates_ids("L321"))
    X_cand_viz = _to_dense(adatas_proc["L321"].X[np.asarray(cand_mask_L321)])

    _logistic_regression_plot_and_coeff(
        {"logistic_regression": logreg_pipe},
        adatas_proc["L321"], X_ref_raw, y_ref, X_cand_viz, random_state,
        model_label="LogReg (no PCA)",
    )

    _logistic_regression_plot_and_coeff(
        {"logistic_regression": logreg_pca_pipe},
        adatas_proc["L321"], X_ref_raw, y_ref, X_cand_viz, random_state,
        model_label="LogReg + PCA",
    )

    # ── 7. Metrics table ─────────────────────────────────────────────────────
    def _row(name, y_true, oof_prob, pred=None):
        if pred is None:
            pred = (oof_prob >= prob_thresh).astype(int)
        return {
            "model":            name,
            "accuracy":         accuracy_score(y_true, pred),
            "precision":        precision_score(y_true, pred, zero_division=0),
            "recall":           recall_score(y_true, pred, zero_division=0),
            "f1":               f1_score(y_true, pred, zero_division=0),
            "roc_auc":          roc_auc_score(y_true, oof_prob),
            "ap":               average_precision_score(y_true, oof_prob),
            "healthy_as_tumor":  int(((y_true == 1) & (pred == 0)).sum()),
            "tumor_as_healthy":  int(((y_true == 0) & (pred == 1)).sum()),
        }

    rows = [
        _row("LogReg",             y_ref, oof_logreg),
        _row("LogReg + PCA",       y_ref, oof_logreg_pca),
        _row("LogReg + KNN (PCA)", y_ref, oof_knn_score, pred=oof_knn_pred),
        _row("XGBoost",            y_ref, oof_xgb),
    ]
    comp_df = pd.DataFrame(rows).set_index("model")

    print(f"\n=== Model Comparison (OOF at threshold={prob_thresh}) ===")
    print(comp_df.round(4).to_string())

    # ── 8. Metrics bar chart ─────────────────────────────────────────────────
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = comp_df.index.tolist()
    x = np.arange(len(metric_cols))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mname in enumerate(model_names):
        vals = comp_df.loc[mname, metric_cols].values.astype(float)
        offset = (i - (len(model_names) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=mname)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — OOF at threshold={prob_thresh}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ── 8b. Threshold calibration plot (LogReg + PCA, OOF) ──────────────────
    _plot_threshold_calibration(y_ref, oof_logreg_pca, prob_thresh)

    # ── 9. Score candidates + spatial comparison ─────────────────────────────
    model_labels = ["LogReg", "LogReg+PCA", "LogReg+KNN (PCA)", "XGBoost"]

    def _score_candidates(X_cand):
        """Returns dict of boolean is_tumor arrays for each model."""
        prob_lr     = logreg_pipe.predict_proba(X_cand)[:, 1]
        prob_lr_pca = logreg_pca_pipe.predict_proba(X_cand)[:, 1]
        prob_xgb    = xgb_model.predict_proba(X_cand)[:, 1]

        X_cand_pca  = pca_knn.transform(np.clip(scaler_knn.transform(X_cand), -10, 10))
        prob_lr_knn = logreg_knn_clf.predict_proba(X_cand_pca)[:, 1]
        knn_frac    = y_ref[nn_full.kneighbors(X_cand_pca, return_distance=False)].mean(axis=1)

        return {
            "LogReg":           prob_lr     < prob_thresh,
            "LogReg+PCA":       prob_lr_pca < prob_thresh,
            "LogReg+KNN (PCA)": ~((prob_lr_knn >= prob_thresh) | (knn_frac >= prob_thresh)),
            "XGBoost":          prob_xgb    < prob_thresh,
        }

    def _spatial_comparison(slice_adata, tumor_cells, sid, snum):
        """4-panel spatial plot, one panel per model."""
        is_cand = np.asarray(slice_adata.obs_names.isin(tumor_cells))
        if is_cand.sum() == 0:
            return

        X_cand = _to_dense(slice_adata.X[is_cand])
        is_tumor_by_model = _score_candidates(X_cand)

        obs = slice_adata.obs
        x_all = obs[CENTER_X_GLOBAL_PX].values
        y_all = obs[CENTER_Y_GLOBAL_PX].values

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        fig.suptitle(f"{sid} slice {snum} — Spatial comparison (threshold={prob_thresh})", fontsize=14)

        for ax, mname in zip(axes, model_labels):
            is_tumor_m = np.zeros(slice_adata.n_obs, dtype=bool)
            is_tumor_m[is_cand] = is_tumor_by_model[mname]
            is_rejected = is_cand & ~is_tumor_m
            is_bg = ~is_cand

            ax.scatter(x_all[is_bg],       y_all[is_bg],       c="#E0E0E0", s=0.4, alpha=0.2, linewidths=0, rasterized=True)
            ax.scatter(x_all[is_rejected], y_all[is_rejected], c="black",   s=0.8, alpha=0.4, linewidths=0, rasterized=True, label="Rejected")
            ax.scatter(x_all[is_tumor_m],  y_all[is_tumor_m],  c="red",     s=1.2, alpha=0.9, linewidths=0, rasterized=True, label="Tumor")

            n_tumor = int(is_tumor_m.sum())
            n_rej   = int(is_rejected.sum())
            ax.set_title(f"{mname}\n{n_tumor} tumor | {n_rej} rejected", fontsize=11)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.legend(loc="upper right", markerscale=4, fontsize=8)

        plt.tight_layout()
        plt.show()

    count_rows = []
    slices_dir = pathlib.Path(rf"{BASE_DIR}/resources/cache/")

    for sid in slide_ids:
        snum = primary_slices[sid]
        slice_path = slices_dir / f"slice_{snum}_adata.h5ad"
        if not slice_path.exists():
            continue
        slice_adata = ad.read_h5ad(slice_path)
        sc.pp.normalize_total(slice_adata, target_sum=1e4)
        sc.pp.log1p(slice_adata)
        slice_adata = slice_adata[:, joint_var_names].copy()
        tumor_cells = _get_tumor_candidates_ids(sid)

        is_cand = np.asarray(slice_adata.obs_names.isin(tumor_cells))
        if is_cand.sum() > 0:
            X_cand = _to_dense(slice_adata.X[is_cand])
            scores = _score_candidates(X_cand)
            row = {"slice": f"{sid} slice {snum}", "n_candidates": int(is_cand.sum())}
            for m in model_labels:
                row[m] = int(scores[m].sum())
            count_rows.append(row)

        _spatial_comparison(slice_adata, tumor_cells, sid, snum)

    for sid, nums in remaining_slices_map.items():
        for snum in nums:
            slice_path = slices_dir / f"slice_{snum}_adata.h5ad"
            ann_path = pathlib.Path(
                rf"{BASE_DIR}/outputs/cell_annotation/{sid}/05/{snum}"
                rf"/slice_{snum}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
            )
            if not slice_path.exists() or not ann_path.exists():
                continue
            slice_adata = ad.read_h5ad(slice_path)
            sc.pp.normalize_total(slice_adata, target_sum=1e4)
            sc.pp.log1p(slice_adata)
            slice_adata = slice_adata[:, joint_var_names].copy()
            annotation_df = pd.read_csv(ann_path)
            tumor_cells = _get_tumor_candidates_ids(sid, annotation_df)

            is_cand = np.asarray(slice_adata.obs_names.isin(tumor_cells))
            if is_cand.sum() > 0:
                X_cand = _to_dense(slice_adata.X[is_cand])
                scores = _score_candidates(X_cand)
                row = {"slice": f"{sid} slice {snum}", "n_candidates": int(is_cand.sum())}
                for m in model_labels:
                    row[m] = int(scores[m].sum())
                count_rows.append(row)

            _spatial_comparison(slice_adata, tumor_cells, sid, snum)

    counts_df = pd.DataFrame(count_rows).set_index("slice")
    print("\n=== Tumor Cell Counts per Slice ===")
    print(counts_df.to_string())

    counts_df[model_labels].plot(kind="bar", figsize=(12, 6))
    plt.title("Tumor Cells per Slice by Model")
    plt.ylabel("Tumor cell count")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"comp_df": comp_df, "counts_df": counts_df}


def _get_nuclear_cell_ratio():
    slide_ids = ["L321", "L34"]

    for slide_id in slide_ids:
        healthy_ids = _get_healthy_ref_ids(slide_id)
        tumor_ids = _get_tumor_ref_ids(slide_id)
        df = pd.read_csv(rf"D:/thesis-research/resources/cosmx/{slide_id}/{slide_id}_metadata_file.csv")
        df["global_cell_id"] = df["cell"].apply(lambda x: f"{x}_{slide_id}")

        healthy_df = df[df["global_cell_id"].isin(healthy_ids)].copy()
        tumor_df = df[df["global_cell_id"].isin(tumor_ids)].copy()

        healthy_df['NC_ratio'] = healthy_df['NucArea'] / (healthy_df['Area'] - healthy_df['NucArea'])
        tumor_df['NC_ratio'] = tumor_df['NucArea'] / (tumor_df['Area'] - tumor_df['NucArea'])

        # Add labels and combine
        healthy_df['condition'] = 'Healthy'
        tumor_df['condition'] = 'Tumor'
        combined = pd.concat([healthy_df, tumor_df], ignore_index=True)

        # --- Optional: filter outliers (bad segmentation) ---
        combined = combined[combined['NC_ratio'] < combined['NC_ratio'].quantile(0.99)]
        combined = combined[combined['NC_ratio'] > 0]

        fig, ax = plt.subplots(figsize=(8, 5))
        for cond, color in zip(['Healthy', 'Tumor'], ['steelblue', 'tomato']):
            subset = combined[combined['condition'] == cond]
            subset['NC_ratio'].plot.kde(ax=ax, label=f"{cond} (n={len(subset):,})", color=color)

        ax.set_title(f'Nucleus:Cell Area Ratio Distribution — Slide {slide_id}')
        ax.set_xlim(left=0)
        ax.set_xlabel('Nucleus:Cell Area Ratio')
        ax.set_ylabel('Density')
        ax.legend()
        plt.tight_layout()
        # plt.savefig(rf"D:/thesis-research/resources/cosmx/{slide_id}/{slide_id}_NC_ratio_kde.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    _get_nuclear_cell_ratio()
    run_model_comparison()
