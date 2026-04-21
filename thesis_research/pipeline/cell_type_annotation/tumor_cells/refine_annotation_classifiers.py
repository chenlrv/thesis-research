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


def train_classifiers(adata, pca=False, n_splits=5, random_state=42):
    """
    Trains classifiers on a reference AnnData and returns the models.
    """
    adata_proc = adata.copy()
    healthy_ref_ids = _get_healthy_ref_ids()
    tumor_ref_ids = _get_tumor_ref_ids()

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
    adata,
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
    fitted_models, metrics_df, oof_probs = train_classifiers(adata, pca, n_splits, random_state)
    adata_raw = adata.copy()
    adata_proc = adata.copy()

    healthy_ref_ids = _get_healthy_ref_ids()
    tumor_ref_ids = _get_tumor_ref_ids()

    healthy_ref_mask = adata_proc.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata_proc.obs_names.isin(tumor_ref_ids)
    tumor_candidate_ids = _get_tumor_candidates_ids()
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

    _plot_results(adata_sub, pca)

    slices_dir = pathlib.Path(rf"{BASE_DIR}/resources/cache/")
    slide_map = {"L321": [2,3], "L34": [4,5,6]}
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
                    tumor_cells = _get_tumor_candidates_ids(annotation_df)
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

        preds = (probs[:, 1] >= 0.6).astype(int)

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

        adata_sub.obs[pred_col] = (final_scores >= 0.6).astype(int)

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


def _plot_results(adata_sub, pca):
    map = {
        "Logistic Regression": "pred_healthy_logistic_regression",
        "Random Forest": "pred_healthy_random_forest",
        "XGBoost": "pred_healthy_xgboost",
    }

    for model_name, pred_col in map.items():
        healthy_cells = adata_sub.obs_names[adata_sub.obs[pred_col] == 1]
        _plot_tumor_cells(healthy_cells, model_name, pca)


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
        slice_adata.obs[pred_col] = (slice_adata.obs[score_col] >= 0.6).astype(int)


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

if __name__ == "__main__":
    # print("=== Running classifiers with healthy refs from the same slice with PCA ===")
    # run_classifiers_compare_and_score(
    #     adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
    #     healthy_refs_same_slice=True,
    #     pca = True
    # )
    # print("\n\n=== Running classifiers with healthy refs from control slice with PCA ===")
    # run_classifiers_compare_and_score(
    #     adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
    #     healthy_refs_same_slice=False,
    #     pca=True
    #
    # )
    # print("=== Running classifiers with healthy refs from the same slice NO PCA ===")
    # run_classifiers_compare_and_score(
    #     adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
    #     healthy_refs_same_slice=True,
    # )
    print("\n\n=== Running classifiers with healthy refs from control slice NO PCA ===")
    run_classifiers_compare_and_score(
        adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad"),
    )
