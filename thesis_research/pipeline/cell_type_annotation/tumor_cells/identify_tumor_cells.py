import os

import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import seaborn as sns
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

from thesis_research.utils.columns import CENTER_Y_GLOBAL_PX, CENTER_X_GLOBAL_PX
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

BASE_DIR = r"D:/thesis-research/"

def identify_tumor_cells(
    adata,
    n_comps=50,
    n_pcs=30,
    k=15,
    neighbor_thresh=0.7,
    prob_thresh=0.7,
):
    adata = adata.copy()

    healthy_ref_ids = _get_healthy_ref_ids(adata) #cells marked as tumor but they are 100% healthy
    tumor_ref_ids = _get_tumor_ref_ids(adata) #cells marked as tumor with very high confidence
    tumor_candidate_ids = _get_tumor_candidates_ids(adata) #cells marked as tumor but potentially mislabeled (lower confidence)

    healthy_ref_mask = adata.obs_names.isin(healthy_ref_ids)
    tumor_ref_mask = adata.obs_names.isin(tumor_ref_ids)
    tumor_candidate_mask = adata.obs_names.isin(tumor_candidate_ids)

    subset_mask = healthy_ref_mask | tumor_ref_mask | tumor_candidate_mask
    adata_sub = adata[subset_mask].copy()

    # Correctly aligned masks for the subset
    h_mask_sub = np.asarray(healthy_ref_mask[subset_mask])
    t_mask_sub = np.asarray(tumor_ref_mask[subset_mask])
    cand_mask_sub = np.asarray(tumor_candidate_mask[subset_mask])
    ref_mask_sub = h_mask_sub | t_mask_sub

    # Standard preprocessing for PCA space
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    # extract matrix
    data_mat = adata_sub.X.toarray() if issparse(adata_sub.X) else np.asarray(adata_sub.X)

    # fit scaling on reference cells only
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(data_mat[ref_mask_sub])

    # transform all cells using reference-defined scaling
    data_scaled = np.clip(scaler.transform(data_mat), -10, 10)
    adata_sub.X = data_scaled

    # fit PCA on reference cells only
    n_comps_eff = min(n_comps, ref_mask_sub.sum() - 1, data_scaled.shape[1] - 1)
    pca_obj = PCA(n_components=n_comps_eff, random_state=42)
    pca_obj.fit(data_scaled[ref_mask_sub])

    # project all cells into the reference-defined PCA space
    X = pca_obj.transform(data_scaled)

    n_pcs_eff = min(n_pcs, X.shape[1])
    X = X[:, :n_pcs_eff]

    ####### Cross Validation
    X_ref = X[ref_mask_sub]
    y_ref = h_mask_sub[ref_mask_sub].astype(int)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_res = cross_validate(
        clf, X_ref, y_ref, cv=cv, scoring=scoring_metrics, return_train_score=False
    )
    metrics_row = {"model": "logistic_regression_pca"}
    for metric in scoring_metrics:
        metrics_row[f"{metric}_mean"] = cv_res[f"test_{metric}"].mean()
        metrics_row[f"{metric}_std"] = cv_res[f"test_{metric}"].std()

    metrics_df = pd.DataFrame([metrics_row]).set_index("model")

    print("\n=== Cross-validated performance on Reference Cells ===")
    print(metrics_df.round(4).to_string())


    # ---------- score 1: fraction of healthy nearest neighbors in pca space -------
    nn = NearestNeighbors(n_neighbors=min(k, X_ref.shape[0]))
    nn.fit(X_ref)
    nn_idx = nn.kneighbors(X, return_distance=False)

    healthy_neighbor_frac = y_ref[nn_idx].mean(axis=1)

    # ---------- score 2: classifier probability of being healthy ----------
    clf.fit(X_ref, y_ref)
    healthy_prob = clf.predict_proba(X)[:, 1]

    ### pca plot
    df_plot = pd.DataFrame(
        X_ref[:, :2],
        columns=['PC1', 'PC2'],
        index=adata_sub.obs_names[ref_mask_sub]
    )

    # Add the Ground Truth labels (1=Healthy, 0=Tumor)
    df_plot['Status'] = np.where(y_ref == 1, 'Healthy Ref', 'Tumor Ref')

    # Add the Candidate cells as a background (optional but helpful)
    X_cand = X[cand_mask_sub]
    df_cand = pd.DataFrame(
        X_cand[:, :2],
        columns=['PC1', 'PC2']
    )
    df_cand['Status'] = 'Candidates'

    plt.figure(figsize=(14, 7))

    # Plot the candidates in light gray to see the "context"
    sns.scatterplot(
        data=df_cand, x='PC1', y='PC2',
        color='lightgray', alpha=0.3, label='Candidates', s=10
    )

    # Plot the Reference groups with distinct colors
    sns.scatterplot(
        data=df_plot, x='PC1', y='PC2',
        hue='Status', palette={'Healthy Ref': 'dodgerblue', 'Tumor Ref': 'firebrick'},
        s=40, edgecolor='black', alpha=0.8
    )

    plt.title("Reference Projection: Is a Linear Boundary Appropriate?")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Fit the classifier
    clf = LogisticRegression(class_weight='balanced').fit(X_ref[:, :2], y_ref)

    # Create a mesh grid to plot the decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Draw the boundary line (where probability is 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

    # ------------------------------
    seed_neighbor_thresh = 0.7
    seed_prob_thresh = 0.7

    healthy_like_seed_sub = (
            cand_mask_sub &
            (healthy_neighbor_frac >= seed_neighbor_thresh) &
            (healthy_prob >= seed_prob_thresh)
    )

    adata_sub.obs["healthy_neighbor_frac"] = healthy_neighbor_frac
    adata_sub.obs["healthy_prob"] = healthy_prob

    # # ---------- score 3: fraction of healthy nearest neighbors in spatial space ----------
    spatial_cols = ["CenterX_global_px", "CenterY_global_px"]
    adata1 = ad.read_h5ad(fr"{BASE_DIR}/resources/cache/slice_1_adata.h5ad")
    X_spatial = adata1.obs[spatial_cols].to_numpy()

    tumor_ref_mask_spatial = adata1.obs_names.isin(tumor_ref_ids)
    tumor_candidate_mask_spatial = adata1.obs_names.isin(tumor_candidate_ids)

    # background cells = not trusted tumor refs and not tumor candidates
    background_healthy_mask_spatial = (
            (~tumor_ref_mask_spatial) &
            (~tumor_candidate_mask_spatial)
    )

    adata1.obs["healthy_like_seed"] = adata1.obs_names.isin(
       adata_sub.obs_names[healthy_like_seed_sub]
    )
    healthy_support_mask_spatial = (
           background_healthy_mask_spatial |
           adata1.obs["healthy_like_seed"].to_numpy()
    )
#
#
#
    nn_spatial = NearestNeighbors(n_neighbors=k+1)
    nn_spatial.fit(X_spatial)
    nn_spatial_idx = nn_spatial.kneighbors(X_spatial, return_distance=False)
    neighbor_indices = nn_spatial_idx[:, 1:]
#
    healthy_support_spatial_frac = healthy_support_mask_spatial[neighbor_indices].mean(axis=1)
    adata1.obs["healthy_support_spatial_frac"] = healthy_support_spatial_frac
#
    healthy_support_spatial_frac_sub = (
        adata1.obs["healthy_support_spatial_frac"]
        .reindex(adata_sub.obs_names)
        .values
    )

    # ---------- combine scores ----------
    spatial_seed_thresh = 0.5
    likely_mislabeled_healthy = ( #mislabeled as tumor but likely healthy
            cand_mask_sub &
            ((healthy_neighbor_frac >= neighbor_thresh) &
            (healthy_prob >= prob_thresh))
            # (healthy_support_spatial_frac_sub >= spatial_seed_thresh)
    )
    # Save results back to original adata
    adata_sub.obsm["X_pca_filtering"] = X
    adata_sub.obs["healthy_neighbor_frac"] = healthy_neighbor_frac
    adata_sub.obs["healthy_prob"] = healthy_prob
    adata_sub.obs["healthy_like_seed"] = healthy_like_seed_sub
    # adata_sub.obs["healthy_support_spatial_frac_sub"] = healthy_support_spatial_frac_sub
    adata_sub.obs["likely_mislabeled_healthy"] = likely_mislabeled_healthy

    # is_real_tumor = (
    #         adata_sub.obs["is_tumor_ref"] &
    #         (~adata_sub.obs["likely_mislabeled_healthy"])
    # )
    #
    # tumor_ids = adata_sub.obs_names[is_real_tumor]
    # adata_t = adata[adata.obs_names.isin(tumor_ids)].copy()
    #
    # sc.pp.normalize_total(adata_t, target_sum=1e4)
    # sc.pp.log1p(adata_t)
    #
    # # optional but often helpful
    # # sc.pp.highly_variable_genes(adata_t, n_top_genes=1000, flavor="seurat")
    # # adata_t = adata_t[:, adata_t.var["highly_variable"]].copy()
    #
    # sc.pp.scale(adata_t, max_value=10)
    # sc.tl.pca(adata_t, n_comps=min(n_comps, adata_t.n_obs - 1, adata_t.n_vars - 1))
    #
    # n_pcs_eff = min(n_pcs, adata_t.obsm["X_pca"].shape[1])
    # sc.pp.neighbors(adata_t, n_neighbors=k, n_pcs=n_pcs_eff)
    # sc.tl.leiden(adata_t, resolution=0.3, key_added="tumor_subcluster")
    # sc.tl.umap(adata_t)
    #
    # sc.pl.pca(adata_t, color="tumor_subcluster", show=True)

    healthy_cells = adata_sub.obs_names[adata_sub.obs["likely_mislabeled_healthy"] == 1]
    _plot_tumor_cells(healthy_cells, "Logistic Regression and KNN on PCA space")


    return adata_sub

def plot_spatial_score_threshold(adata_sub, threshold=0.7):
    adata = ad.read_h5ad(r"/resources/cache/slice_1_adata.h5ad")

    # copy the score from adata_sub into the full slice object by cell id
    adata.obs["healthy_spatial_prob"] = (
        adata_sub.obs["healthy_spatial_prob"]
        .reindex(adata.obs_names)
        .values
    )

    # same coordinate handling you already use
    adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]

    # masks
    has_score = adata.obs["healthy_spatial_prob"].notna()
    is_pass = adata.obs["healthy_spatial_prob"] >= threshold
    is_fail = has_score & (~is_pass)
    is_background = ~has_score

    plt.figure(figsize=(10, 10))

    # background: cells not in adata_sub / no score
    plt.scatter(
        adata.obs.loc[is_background, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_background, CENTER_Y_GLOBAL_PX],
        c="#D9D9D9",
        s=0.7,
        alpha=1,
        edgecolors="none",
        label="No spatial score"
    )

    # fail threshold
    plt.scatter(
        adata.obs.loc[is_fail, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_fail, CENTER_Y_GLOBAL_PX],
        c="dodgerblue",
        s=1.5,
        alpha=0.8,
        edgecolors="none",
        label=f"healthy_spatial_prob < {threshold}"
    )

    # pass threshold
    plt.scatter(
        adata.obs.loc[is_pass, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_pass, CENTER_Y_GLOBAL_PX],
        c="red",
        s=1.8,
        alpha=0.9,
        edgecolors="none",
        label=f"healthy_spatial_prob >= {threshold}"
    )

    plt.title(f"Slice 1 spatial map: healthy_spatial_prob threshold = {threshold}", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(markerscale=4, frameon=False)
    plt.show()


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
    plt.title(f"L321 slice 1 spatial Mapping tumor cells\n {classifier_name}\n{cells_to_keep.sum()} tumor cells out of {is_tumor.sum()}, filtered {len(healthy_cells)}", fontsize=15)
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def _get_healthy_ref_ids(adata: AnnData):
    CELL_COL = 'cell_barcode'
    df_results3 = pd.read_csv(
        fr"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_3_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results3 = df_results3[df_results3['predicted_cell_type'] == 'Tumor']
    df_results3['next_best_score'] = df_results3[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results3['delta_score'] = abs(df_results3['score_tumor'] - df_results3['next_best_score'])

    df_results3 = df_results3[
        (df_results3['predicted_cell_type'] == 'Tumor') &
        (df_results3['score_tumor'] > 0.2) &  # New: Absolute match filter
        (df_results3['delta_score'] > 0.08) &
        (df_results3['score_tumor'] > df_results3['next_best_score'])
        ]

    return set(df_results3[CELL_COL].astype(str))

def _get_tumor_ref_ids(adata: AnnData):
    CELL_COL = 'cell_barcode'
    df_results1 = pd.read_csv(
        fr"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results1 = df_results1[df_results1['predicted_cell_type'] == 'Tumor']
    df_results1['next_best_score'] = df_results1[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results1['delta_score'] = abs(df_results1['score_tumor'] - df_results1['next_best_score'])

    df_results1 = df_results1[
        (df_results1['predicted_cell_type'] == 'Tumor') &
        (df_results1['score_tumor'] > 0.5) &
        (df_results1['delta_score'] > 0.08) &
        (df_results1['score_tumor'] > df_results1['next_best_score'])
        ]

    return set(df_results1[CELL_COL].astype(str))

def _get_tumor_candidates_ids(adata: AnnData):
    CELL_COL = 'cell_barcode'
    df_results1 = pd.read_csv(
        fr"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results1 = df_results1[df_results1['predicted_cell_type'] == 'Tumor']
    df_results1['next_best_score'] = df_results1[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results1['delta_score'] = abs(df_results1['score_tumor'] - df_results1['next_best_score'])

    df_results1 = df_results1[
        (df_results1['predicted_cell_type'] == 'Tumor') &
        (df_results1['score_tumor'] > 0.2) &
        (df_results1['delta_score'] > 0.08) &
        (df_results1['score_tumor'] > df_results1['next_best_score'])
        ]

    return set(df_results1[CELL_COL].astype(str))


adata_new = identify_tumor_cells(
    adata=ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_L321_adata.h5ad")
)
