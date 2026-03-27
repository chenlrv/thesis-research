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


def identify_tumor_cells(
    adata,
    n_comps=50,
    n_pcs=30,
    k=15,
    neighbor_thresh=0.7,
    prob_thresh=0.7,
):
    adata = adata.copy()

    healthy_ref_mask = _get_healthy_ref_mask(adata) #cells marked as tumor but they are 100% healthy
    tumor_ref_mask = _get_tumor_ref_mask(adata) #cells marked as tumor with very high confidence
    tumor_candidate_mask = _get_tumor_candidates_mask(adata) #cells marked as tumor but potentially mislabeled (lower confidence)
    subset_mask = healthy_ref_mask | tumor_ref_mask | tumor_candidate_mask
    adata_sub = adata[subset_mask].copy()

    healthy_ref_mask_sub = healthy_ref_mask[subset_mask]
    tumor_ref_mask_sub = tumor_ref_mask[subset_mask]
    tumor_candidate_mask_sub = tumor_candidate_mask[subset_mask]

    # Standard preprocessing for PCA space
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    # reference cells inside the subset
    ref_mask = np.asarray(healthy_ref_mask_sub | tumor_ref_mask_sub)

    # extract matrix
    data_mat = adata_sub.X.toarray() if issparse(adata_sub.X) else np.asarray(adata_sub.X)

    # fit scaling on reference cells only
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(data_mat[ref_mask])

    # transform all cells using reference-defined scaling
    data_scaled = scaler.transform(data_mat)
    data_scaled = np.clip(data_scaled, -10, 10)
    adata_sub.X = data_scaled

    # fit PCA on reference cells only
    n_comps_eff = min(n_comps, ref_mask.sum() - 1, data_scaled.shape[1] - 1)
    pca_obj = PCA(n_components=n_comps_eff, random_state=42)
    pca_obj.fit(data_scaled[ref_mask])

    # project all cells into the reference-defined PCA space
    X = pca_obj.transform(data_scaled)

    n_pcs_eff = min(n_pcs, X.shape[1])
    X = X[:, :n_pcs_eff]


    #####
    healthy_ref_mask = np.asarray(healthy_ref_mask_sub)
    tumor_ref_mask = np.asarray(tumor_ref_mask_sub)
    tumor_candidate_mask = np.asarray(tumor_candidate_mask_sub)

    # ---------- score 1: fraction of healthy nearest neighbors in pca space ----------
    ref_mask = healthy_ref_mask | tumor_ref_mask
    X_ref = X[ref_mask]
    y_ref = healthy_ref_mask[ref_mask].astype(int)   # 1 = healthy, 0 = tumor

    nn = NearestNeighbors(n_neighbors=min(k, X_ref.shape[0]))
    nn.fit(X_ref)
    nn_idx = nn.kneighbors(X, return_distance=False)

    healthy_neighbor_frac = y_ref[nn_idx].mean(axis=1)

    # ---------- score 2: classifier probability of being healthy ----------
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X_ref, y_ref)
    healthy_prob = clf.predict_proba(X)[:, 1]

    ### pca plot
    df_plot = pd.DataFrame(
        X_ref[:, :2],
        columns=['PC1', 'PC2'],
        index=adata_sub.obs_names[ref_mask]
    )

    # Add the Ground Truth labels (1=Healthy, 0=Tumor)
    df_plot['Status'] = np.where(y_ref == 1, 'Healthy Ref', 'Tumor Ref')

    # Add the Candidate cells as a background (optional but helpful)
    X_cand = X[tumor_candidate_mask]
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
    plt.show()

    # ------------------------------
    seed_neighbor_thresh = 0.7
    seed_prob_thresh = 0.7

    healthy_like_seed_sub = (
            tumor_candidate_mask &
            (healthy_neighbor_frac >= seed_neighbor_thresh) &
            (healthy_prob >= seed_prob_thresh)
    )

    # ---------- score 3: fraction of healthy nearest neighbors in spatial space ----------
    spatial_cols = ["CenterX_global_px", "CenterY_global_px"]
    adata1 = ad.read_h5ad("D:\\thesis-research\\resources\\cache\\slice_1_adata.h5ad")
    X_spatial = adata1.obs[spatial_cols].to_numpy()

    tumor_ref_mask_spatial = np.asarray(_get_tumor_ref_mask(adata1))
    tumor_candidate_mask_spatial = np.asarray(_get_tumor_candidates_mask(adata1))

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



    nn_spatial = NearestNeighbors(n_neighbors=min(k + 1, X_spatial.shape[0]))
    nn_spatial.fit(X_spatial)
    nn_spatial_idx = nn_spatial.kneighbors(X_spatial, return_distance=False)
    neighbor_indices = nn_spatial_idx[:, 1:]

    healthy_support_spatial_frac = healthy_support_mask_spatial[neighbor_indices].mean(axis=1)
    adata1.obs["healthy_support_spatial_frac"] = healthy_support_spatial_frac

    healthy_support_spatial_frac_sub = (
        adata1.obs["healthy_support_spatial_frac"]
        .reindex(adata_sub.obs_names)
        .values
    )

    # ---------- combine scores ----------
    spatial_seed_thresh = 0.7
    likely_mislabeled_healthy = ( #mislabeled as tumor but likely healthy
            tumor_candidate_mask &
            (healthy_neighbor_frac >= neighbor_thresh) &
            (healthy_prob >= prob_thresh) &
            (healthy_support_spatial_frac_sub >= spatial_seed_thresh)
    )
    # Save results back to original adata
    adata_sub.obsm["X_pca_filtering"] = X
    adata_sub.obs["healthy_neighbor_frac"] = healthy_neighbor_frac
    adata_sub.obs["healthy_prob"] = healthy_prob
    adata_sub.obs["healthy_like_seed"] = healthy_like_seed_sub
    adata_sub.obs["healthy_support_spatial_frac_sub"] = healthy_support_spatial_frac_sub
    adata_sub.obs["likely_mislabeled_healthy"] = likely_mislabeled_healthy
    return adata_sub


def plot_spatial_score_threshold(adata_sub, threshold=0.7):
    adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\slice_1_adata.h5ad")

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

def _get_healthy_ref_mask(adata: AnnData):
    CELL_COL = 'cell_barcode'
    df_results3 = pd.read_csv(
        fr"D:\thesis-research\outputs\cell_annotation\L321\05\slice_3_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results3 = df_results3[df_results3['predicted_cell_type'] == 'Tumor']
    df_results3['next_best_score'] = df_results3[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results3['delta_score'] = abs(df_results3['score_tumor'] - df_results3['next_best_score'])

    df_results3 = df_results3[
        (df_results3['predicted_cell_type'] == 'Tumor') &
        (df_results3['score_tumor'] > 0.2) &  # New: Absolute match filter
        (df_results3['delta_score'] > 0.08) &
        (df_results3['score_tumor'] > df_results3['next_best_score'])
        ]

    df_results3[CELL_COL] = df_results3['cell_barcode']
    return adata.obs_names.isin(df_results3[CELL_COL])

def _get_tumor_ref_mask(adata: AnnData):
    CELL_COL = 'cell_barcode'
    df_results1 = pd.read_csv(
        fr"D:\thesis-research\outputs\cell_annotation\L321\05\slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results1 = df_results1[df_results1['predicted_cell_type'] == 'Tumor']
    df_results1['next_best_score'] = df_results1[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results1['delta_score'] = abs(df_results1['score_tumor'] - df_results1['next_best_score'])

    df_results1 = df_results1[
        (df_results1['predicted_cell_type'] == 'Tumor') &
        (df_results1['score_tumor'] > 0.5) &
        (df_results1['delta_score'] > 0.08) &
        (df_results1['score_tumor'] > df_results1['next_best_score'])
        ]

    return adata.obs_names.isin(df_results1[CELL_COL])

def _get_tumor_candidates_mask(adata: AnnData):
    CELL_COL = 'cell_barcode'
    df_results1 = pd.read_csv(
        fr"D:\thesis-research\outputs\cell_annotation\L321\05\slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results1 = df_results1[df_results1['predicted_cell_type'] == 'Tumor']
    df_results1['next_best_score'] = df_results1[['score_brain_struct', 'score_brain_immune']].max(axis=1)
    df_results1['delta_score'] = abs(df_results1['score_tumor'] - df_results1['next_best_score'])

    df_results1 = df_results1[
        (df_results1['predicted_cell_type'] == 'Tumor') &
        (df_results1['score_tumor'] > 0.2) &
        (df_results1['delta_score'] > 0.08) &
        (df_results1['score_tumor'] > df_results1['next_best_score'])
        ]

    return adata.obs_names.isin(df_results1[CELL_COL])


adata_new = identify_tumor_cells(
    adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\sample_L321_adata.h5ad")
)
