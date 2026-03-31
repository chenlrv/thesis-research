import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import anndata as ad


from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import _get_healthy_ref_ids, \
    _get_tumor_ref_ids, _get_tumor_candidates_ids
from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX

BASE_DIR = r"D:/thesis-research/"


def run_classifiers_compare_and_score(
    adata,
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
    # Keep original adata intact for the returned subset
    adata_raw = adata.copy()
    adata_proc = adata.copy()

    healthy_ref_ids = _get_healthy_ref_ids(adata_proc)
    tumor_ref_ids = _get_tumor_ref_ids(adata_proc)
    tumor_candidate_ids = _get_tumor_candidates_ids(adata_proc)

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

    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)

    X_ref = _to_dense(adata_proc.X[ref_mask])
    X_all = _to_dense(adata_proc.X[selected_mask])

    y_ref = np.asarray(healthy_ref_mask[ref_mask]).astype(int)

    # For plotting only
    X_cand = _to_dense(adata_proc.X[tumor_candidate_mask])

    n_positive = int((y_ref == 1).sum())  # healthy
    n_negative = int((y_ref == 0).sum())  # tumor
    scale_pos_weight = n_negative / max(n_positive, 1)

    models = _get_classifier_models(random_state, scale_pos_weight)
    scoring = _get_scoring()

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fitted_models, metrics_df = _fit_models(models, X_ref, y_ref, cv, scoring)
    #
    # # Difference table vs logistic regression
    # mean_cols = [c for c in metrics_df.columns if c.endswith("_mean")]
    # diff_vs_logreg_df = metrics_df[mean_cols].subtract(
    #     metrics_df.loc["logistic_regression", mean_cols], axis=1
    # )
    # diff_vs_logreg_df = diff_vs_logreg_df.rename_axis("model")
    # diff_vs_logreg_df.columns = [c.replace("_mean", "_delta_vs_logreg") for c in diff_vs_logreg_df.columns]


    adata_sub = _get_fit_scores(fitted_models, X_all, adata_raw, selected_mask, healthy_ref_mask, tumor_ref_mask)

    _logistic_regression_plot_and_coeff(fitted_models, adata_proc, X_ref, y_ref, X_cand, random_state)
    # ------------------------------------------------------------------
    # Present tables
    # ------------------------------------------------------------------
    print("\n=== Cross-validated performance ===")
    print(metrics_df.round(4).to_string())

    _plot_results(adata_sub)
    # print("\n=== Difference vs Logistic Regression ===")
    # print(diff_vs_logreg_df.round(4).to_string())

    return {
        "metrics_df": metrics_df,
        # "diff_vs_logreg_df": diff_vs_logreg_df,
        "adata_sub": adata_sub,
        "models": fitted_models,
        # "logreg_gene_coefficients": logreg_gene_coefficients,
    }

def _to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)

def _get_classifier_models(random_state, scale_pos_weight):
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                solver="liblinear",
                random_state=random_state,
            ))
        ]),
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

    metrics_df = pd.DataFrame(metrics_rows).set_index("model").sort_values(
        by="roc_auc_mean", ascending=False
    )

    return fitted_models, metrics_df


def _get_fit_scores(fitted_models, X_all, adata_raw, selected_mask, healthy_ref_mask, tumor_ref_mask):
    score_dict = {}
    for model_name, fitted_model in fitted_models.items():
        score_dict[model_name] = fitted_model.predict_proba(X_all)[:, 1]  # P(healthy)

    # Create returned subset from original adata (not normalized/log-transformed)
    adata_sub = adata_raw[selected_mask].copy()

    group = np.full(adata_sub.n_obs, "Tumor Candidate", dtype=object)
    group[np.where(healthy_ref_mask[selected_mask])[0]] = "Healthy Ref"
    group[np.where(tumor_ref_mask[selected_mask])[0]] = "Tumor Ref"

    adata_sub.obs["classifier_group"] = group
    adata_sub.obs["is_reference"] = adata_sub.obs["classifier_group"].isin(["Healthy Ref", "Tumor Ref"])
    adata_sub.obs["reference_label"] = np.where(
        adata_sub.obs["classifier_group"] == "Healthy Ref", 1,
        np.where(adata_sub.obs["classifier_group"] == "Tumor Ref", 0, np.nan)
    )

    adata_sub.obs["score_healthy_logreg"] = score_dict["logistic_regression"]
    adata_sub.obs["score_healthy_rf"] = score_dict["random_forest"]
    adata_sub.obs["score_healthy_xgb"] = score_dict["xgboost"]
    adata_sub.obs["score_healthy_mean"] = np.vstack([
        score_dict["logistic_regression"],
        score_dict["random_forest"],
        score_dict["xgboost"],
    ]).mean(axis=0)

    adata_sub.obs["pred_healthy_logreg"] = (adata_sub.obs["score_healthy_logreg"] >= 0.5).astype(int)
    adata_sub.obs["pred_healthy_rf"] = (adata_sub.obs["score_healthy_rf"] >= 0.5).astype(int)
    adata_sub.obs["pred_healthy_xgb"] = (adata_sub.obs["score_healthy_xgb"] >= 0.5).astype(int)
    adata_sub.obs["pred_healthy_mean"] = (adata_sub.obs["score_healthy_mean"] >= 0.5).astype(int)

    return adata_sub

def _logistic_regression_plot_and_coeff(
    fitted_models, adata_proc, X_ref, y_ref, X_cand, random_state
):
    logreg_clf = fitted_models["logistic_regression"].named_steps["clf"]

    gene_df = pd.DataFrame({
        "gene": adata_proc.var_names,
        "coef": logreg_clf.coef_[0]
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
        x="PC1", y="PC2",
        color="lightgray",
        alpha=0.35,
        s=12,
        label="Candidates",
        ax=ax,
    )

    sns.scatterplot(
        data=df_ref_plot,
        x="PC1", y="PC2",
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
    map = {"Logistic Regression": "pred_healthy_logreg",
           "Random Forest": "pred_healthy_rf",
           "XGBoost": "pred_healthy_xgb"}

    for model_name, pred_col in map.items():
        healthy_cells = adata_sub.obs_names[adata_sub.obs[pred_col] == 1]
        _plot_tumor_cells(healthy_cells, model_name)



def _plot_tumor_cells(healthy_cells, classifier_name):
    adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\slice_1_adata.h5ad")
    df_results = pd.read_csv(
        r"D:\thesis-research\outputs\cell_annotation\L321\05\slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results = df_results[df_results['predicted_cell_type'] == 'Tumor']
    df_results['next_best_score'] = df_results[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results['delta_score'] = abs(df_results['score_tumor'] - df_results['next_best_score'])
    abs_threshold = 0.2
    delta_threshold = 0.08
    df_results = df_results[
        (df_results['predicted_cell_type'] == 'Tumor') &
        (df_results['score_tumor'] > abs_threshold) &  # New: Absolute match filter
        (df_results['delta_score'] > delta_threshold) &
        (df_results['score_tumor'] > df_results['next_best_score'])
        ]
    adata.obs = adata.obs.merge(df_results[['cell_barcode', 'predicted_cell_type']],
                                left_index=True, right_on='cell_barcode', how='left').set_index('cell_barcode')
    adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]
    adata.obsm['spatial'] = np.stack([
        adata.obs[CENTER_X_GLOBAL_PX].values,
        adata.obs[CENTER_Y_GLOBAL_PX].values
    ], axis=1)
    adata.obs["predicted_cell_type"] = adata.obs["predicted_cell_type"].astype("category")
    is_tumor = adata.obs["predicted_cell_type"] == "Tumor"
    coords = np.c_[
        adata.obs[CENTER_X_GLOBAL_PX].to_numpy(),
        adata.obs[CENTER_Y_GLOBAL_PX].to_numpy()
    ]

    cells_to_keep = adata.obs["predicted_cell_type"] == "Tumor"

    cells_to_keep = (
            (adata.obs["predicted_cell_type"] == "Tumor") &
            (~adata.obs_names.isin(healthy_cells))
    )
    is_background = ~cells_to_keep
    plt.figure(figsize=(10, 10))
    plt.scatter(
        adata.obs.loc[is_background, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_background, CENTER_Y_GLOBAL_PX],
        c='#CFCFCF',
        s=0.7,  # Tiny dots
        alpha=1,  # Very faint
        edgecolors='none'
    )
    # LAYER 2: The "Tumor" (Foreground)
    # Highlighted in Red
    plt.scatter(
        adata.obs.loc[~is_background, CENTER_X_GLOBAL_PX],
        adata.obs.loc[~is_background, CENTER_Y_GLOBAL_PX],
        c="red",
        s=1.5,  # Larger dots to stand out
        alpha=0.8,  # Solid color
        edgecolors='none',
        label='D122 Tumor Cells'
    )
    plt.title(f"L321 slice 1 spatial Mapping tumor cells\nclassifier {classifier_name} - {len(cells_to_keep)} tumor cells out of {len(is_tumor)}, filtered {len(healthy_cells)}", fontsize=15)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


    run_classifiers_compare_and_score(
        adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad")
    )