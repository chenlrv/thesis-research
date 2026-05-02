import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import seaborn as sns
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from scipy.sparse import issparse
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from thesis_research.utils.columns import CENTER_Y_GLOBAL_PX, CENTER_X_GLOBAL_PX

BASE_DIR = r"D:/thesis-research/"


def identify_tumor_cells(
    adata,
    slide_id,
    n_comps=50,
    n_pcs=30,
    k=15,
    neighbor_thresh=0.7,
    prob_thresh=0.7,
):
    adata = adata.copy()

    healthy_ref_ids = _get_healthy_ref_ids(slide_id)
    tumor_ref_ids = _get_tumor_ref_ids(slide_id)
    tumor_candidate_ids = _get_tumor_candidates_ids(slide_id)

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

    # ---------- OOF scores for ref cells, fresh predictions for candidates ----------

    # LogReg OOF: ref cells get held-out probabilities
    logreg_oof = cross_val_predict(clf, X_ref, y_ref, cv=cv, method="predict_proba")[:, 1]

    # KNN OOF: each ref cell queries neighbors from the other folds only
    knn_oof = np.zeros(len(X_ref))
    for train_idx, val_idx in cv.split(X_ref, y_ref):
        nn_fold = NearestNeighbors(n_neighbors=min(k, len(train_idx)))
        nn_fold.fit(X_ref[train_idx])
        nn_idx_fold = nn_fold.kneighbors(X_ref[val_idx], return_distance=False)
        knn_oof[val_idx] = y_ref[train_idx][nn_idx_fold].mean(axis=1)

    # Full models for candidate predictions
    clf.fit(X_ref, y_ref)
    nn_full = NearestNeighbors(n_neighbors=min(k, X_ref.shape[0]))
    nn_full.fit(X_ref)

    X_cand_pca = X[cand_mask_sub]
    nn_idx_cand = nn_full.kneighbors(X_cand_pca, return_distance=False)

    # ---------- score 1: fraction of healthy nearest neighbors in pca space -------
    healthy_neighbor_frac = np.zeros(len(X))
    healthy_neighbor_frac[ref_mask_sub] = knn_oof
    healthy_neighbor_frac[cand_mask_sub] = y_ref[nn_idx_cand].mean(axis=1)

    # ---------- score 2: classifier probability of being healthy ----------
    healthy_prob = np.zeros(len(X))
    healthy_prob[ref_mask_sub] = logreg_oof
    healthy_prob[cand_mask_sub] = clf.predict_proba(X_cand_pca)[:, 1]

    # OOF audit on reference cells
    ref_pred = ((logreg_oof >= 0.7) & (knn_oof >= 0.7)).astype(int)
    n_healthy_as_tumor = ((y_ref == 1) & (ref_pred == 0)).sum()
    n_tumor_as_healthy = ((y_ref == 0) & (ref_pred == 1)).sum()
    print(f"\n=== OOF Audit (threshold=0.7 on both scores) ===")
    print(f"  Healthy refs predicted as tumor: {n_healthy_as_tumor} / {(y_ref==1).sum()}")
    print(f"  Tumor refs predicted as healthy: {n_tumor_as_healthy} / {(y_ref==0).sum()}")

    ### pca plot
    df_plot = pd.DataFrame(
        X_ref[:, :2], columns=["PC1", "PC2"], index=adata_sub.obs_names[ref_mask_sub]
    )

    # Add the Ground Truth labels (1=Healthy, 0=Tumor)
    df_plot["Status"] = np.where(y_ref == 1, "Healthy Ref", "Tumor Ref")

    # Add the Candidate cells as a background (optional but helpful)
    X_cand = X[cand_mask_sub]
    df_cand = pd.DataFrame(X_cand[:, :2], columns=["PC1", "PC2"])
    df_cand["Status"] = "Candidates"

    plt.figure(figsize=(14, 7))

    # Plot the candidates in light gray to see the "context"
    sns.scatterplot(
        data=df_cand, x="PC1", y="PC2", color="lightgray", alpha=0.3, label="Candidates", s=10
    )

    # Plot the Reference groups with distinct colors
    sns.scatterplot(
        data=df_plot,
        x="PC1",
        y="PC2",
        hue="Status",
        palette={"Healthy Ref": "dodgerblue", "Tumor Ref": "firebrick"},
        s=40,
        edgecolor="black",
        alpha=0.8,
    )

    plt.title("Reference Projection: Is a Linear Boundary Appropriate?")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Fit the classifier
    clf = LogisticRegression(class_weight="balanced").fit(X_ref[:, :2], y_ref)

    # Create a mesh grid to plot the decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Draw the boundary line (where probability is 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

    # ------------------------------
    seed_neighbor_thresh = 0.7
    seed_prob_thresh = 0.7

    healthy_like_seed_sub = (
        cand_mask_sub
        & (healthy_neighbor_frac >= seed_neighbor_thresh)
        & (healthy_prob >= seed_prob_thresh)
    )

    adata_sub.obs["healthy_neighbor_frac"] = healthy_neighbor_frac
    adata_sub.obs["healthy_prob"] = healthy_prob

    # ---------- combine scores ----------
    likely_mislabeled_healthy = (  # mislabeled as tumor but likely healthy
        cand_mask_sub
        & ((healthy_neighbor_frac >= neighbor_thresh) & (healthy_prob >= prob_thresh))
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

    plot_slice_ids = {"L321": 1, "L34": 5}
    healthy_cells = adata_sub.obs_names[adata_sub.obs["likely_mislabeled_healthy"] == 1]
    _plot_tumor_cells(slide_id, plot_slice_ids[slide_id], healthy_cells, "LogReg+KNN (PCA)", pca=False)

    return adata_sub



def plot_spatial_score_threshold(adata_sub, threshold=0.7):
    adata = ad.read_h5ad(r"/resources/cache/slice_1_adata.h5ad")

    # copy the score from adata_sub into the full slice object by cell id
    adata.obs["healthy_spatial_prob"] = (
        adata_sub.obs["healthy_spatial_prob"].reindex(adata.obs_names).values
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
        label="No spatial score",
    )

    # fail threshold
    plt.scatter(
        adata.obs.loc[is_fail, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_fail, CENTER_Y_GLOBAL_PX],
        c="dodgerblue",
        s=1.5,
        alpha=0.8,
        edgecolors="none",
        label=f"healthy_spatial_prob < {threshold}",
    )

    # pass threshold
    plt.scatter(
        adata.obs.loc[is_pass, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_pass, CENTER_Y_GLOBAL_PX],
        c="red",
        s=1.8,
        alpha=0.9,
        edgecolors="none",
        label=f"healthy_spatial_prob >= {threshold}",
    )

    plt.title(f"Slice 1 spatial map: healthy_spatial_prob threshold = {threshold}", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(markerscale=4, frameon=False)
    plt.show()


def _plot_tumor_cells(slide_id, slice_id, healthy_cells, classifier_name, pca):
    adata = ad.read_h5ad(fr"D:\thesis-research\resources\cache\slice_{slice_id}_adata.h5ad")
    df_results = pd.read_csv(
        fr"D:/thesis-research/outputs/cell_annotation/{slide_id}/05/{slice_id}/slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    df_results = df_results[df_results["predicted_cell_type"] == "Tumor"]
    df_results["next_best_score"] = df_results[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results["delta_score"] = abs(df_results["score_tumor"] - df_results["next_best_score"])
    abs_threshold = 0.2
    delta_threshold = 0.08
    df_results = df_results[
        (df_results["predicted_cell_type"] == "Tumor")
        & (df_results["score_tumor"] > abs_threshold)  # New: Absolute match filter
        & (df_results["delta_score"] > delta_threshold)
        & (df_results["score_tumor"] > df_results["next_best_score"])
    ]
    adata.obs = adata.obs.merge(
        df_results[["cell_barcode", "predicted_cell_type"]],
        left_index=True,
        right_on="cell_barcode",
        how="left",
    ).set_index("cell_barcode")
    # adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    # adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]
    adata.obsm["spatial"] = np.stack(
        [adata.obs[CENTER_X_GLOBAL_PX].values, adata.obs[CENTER_Y_GLOBAL_PX].values], axis=1
    )
    adata.obs["predicted_cell_type"] = adata.obs["predicted_cell_type"].astype("category")
    is_tumor = adata.obs["predicted_cell_type"] == "Tumor"
    coords = np.c_[
        adata.obs[CENTER_X_GLOBAL_PX].to_numpy(), adata.obs[CENTER_Y_GLOBAL_PX].to_numpy()
    ]

    cells_to_keep = adata.obs["predicted_cell_type"] == "Tumor"

    cells_to_keep = (adata.obs["predicted_cell_type"] == "Tumor") & (
        ~adata.obs_names.isin(healthy_cells)
    )
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

    n_initial = is_tumor.sum()
    n_kept = cells_to_keep.sum()
    n_filtered = n_initial - n_kept

    plt.title(
        f"{slide_id} slice {slice_id} spatial Mapping tumor cells\n"
        f"PCA={pca}\n"
        f"{classifier_name}\n"
        f"{n_kept} tumor cells out of {n_initial}, filtered {n_filtered}",
        fontsize=15,
    )
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def _get_healthy_ref_ids(slide_id: str):
    CELL_COL = "cell_barcode"
    if slide_id == "L321":
        df_results = pd.read_csv(
        rf"{BASE_DIR}/outputs/cell_annotation/L321/05/3/slice_3_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    else:
        df_results = pd.read_csv(
            rf"{BASE_DIR}/outputs/cell_annotation/L34/05/4/slice_4_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")

    df_results = df_results[df_results["predicted_cell_type"] == "Tumor"]
    df_results["next_best_score"] = df_results[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results["delta_score"] = abs(df_results["score_tumor"] - df_results["next_best_score"])

    df_results = df_results[
        (df_results["predicted_cell_type"] == "Tumor")
        & (df_results["score_tumor"] > 0.2)  # New: Absolute match filter
        & (df_results["delta_score"] > 0.08)
        & (df_results["score_tumor"] > df_results["next_best_score"])
    ]

    return set(df_results[CELL_COL].astype(str))


def _get_healthy_ref_ids_slice_1():
    CELL_COL = "cell_barcode"
    df_results1 = pd.read_csv(
        rf"{BASE_DIR}/outputs/cell_annotation/L321/05/1/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
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



def _get_tumor_ref_ids(slide_id):
    CELL_COL = "cell_barcode"
    if slide_id == "L321":
        df_results = pd.read_csv(
            rf"{BASE_DIR}/outputs/cell_annotation/L321/05/1/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    else:
        df_results = pd.read_csv(
            rf"{BASE_DIR}/outputs/cell_annotation/L34/05/5/slice_5_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df_results = df_results[df_results["predicted_cell_type"] == "Tumor"]
    df_results["next_best_score"] = df_results[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results["delta_score"] = abs(df_results["score_tumor"] - df_results["next_best_score"])

    if slide_id == "L321":
        SCORE_TUMOR = 0.45
    else:
        SCORE_TUMOR = 0.4
    df_results = df_results[
        (df_results["predicted_cell_type"] == "Tumor")
        & (df_results["score_tumor"] >= SCORE_TUMOR)
        & (df_results["delta_score"] > 0.08)
        & (df_results["score_tumor"] > df_results["next_best_score"])
    ]

    return set(df_results[CELL_COL].astype(str))


def _get_tumor_candidates_ids(slide_id: str, annotation_df: pd.DataFrame = None):
    CELL_COL = "cell_barcode"
    if annotation_df is not None:
        df_results = annotation_df
    else:
        if slide_id == "L321":
            df_results = pd.read_csv(
                rf"{BASE_DIR}/outputs/cell_annotation/L321/05/1/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
        else:
            df_results = pd.read_csv(
                rf"{BASE_DIR}/outputs/cell_annotation/L34/05/5/slice_5_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")

    df_results = df_results[df_results["predicted_cell_type"] == "Tumor"].copy()
    df_results["next_best_score"] = df_results[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results["delta_score"] = abs(df_results["score_tumor"] - df_results["next_best_score"])

    df_results = df_results[
        (df_results["predicted_cell_type"] == "Tumor")
        & (df_results["score_tumor"] > 0.2)
        & (df_results["delta_score"] > 0.08)
        & (df_results["score_tumor"] > df_results["next_best_score"])
    ]

    return set(df_results[CELL_COL].astype(str))


def run_identify_tumor_cells_joint(
    n_comps=50, n_pcs=30, k=15,
    neighbor_thresh=0.7, prob_thresh=0.7,
    n_splits=5, random_state=42,
):
    """
    Joint LogReg+KNN (PCA) classifier across both mice.
    - Scaler and PCA fit on ref cells from both slides (batch-effect aware)
    - OOF probabilities for ref cells via k-fold CV
    - Applied to all 6 slices
    """
    slide_ids = ["L321", "L34"]
    primary_slices = {"L321": 1, "L34": 5}
    remaining_slices_map = {"L321": [2, 3], "L34": [4, 6]}

    # 1. Load and normalize per-sample
    adatas_proc = {}
    for sid in slide_ids:
        adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_{sid}_adata.h5ad")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adatas_proc[sid] = adata

    # 2. Reference IDs from both slides
    healthy_ref_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ref_ids   = {sid: _get_tumor_ref_ids(sid)   for sid in slide_ids}
    all_healthy = set().union(*healthy_ref_ids.values())
    all_tumor   = set().union(*tumor_ref_ids.values())

    print("\n=== Joint Reference Pool ===")
    for sid in slide_ids:
        print(f"  {sid}: {len(healthy_ref_ids[sid])} healthy, {len(tumor_ref_ids[sid])} tumor")
    print(f"  Total: {len(all_healthy)} healthy + {len(all_tumor)} tumor")

    # 3. Concatenate normalized adatas
    adata_joint = ad.concat(adatas_proc, join="inner", label="slide_id")
    joint_var_names = adata_joint.var_names.tolist()
    print(f"  Joint adata: {adata_joint.n_obs} cells x {adata_joint.n_vars} genes")

    healthy_mask_j = adata_joint.obs_names.isin(all_healthy)
    tumor_mask_j   = adata_joint.obs_names.isin(all_tumor)
    ref_mask_j     = np.asarray(healthy_mask_j | tumor_mask_j)

    # 4. Scaler and PCA fit on ref cells only — extract only 255 rows, not all 846K
    data_ref = adata_joint.X[ref_mask_j]
    data_ref = data_ref.toarray() if issparse(data_ref) else np.asarray(data_ref)

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(data_ref)
    data_ref_scaled = np.clip(scaler.transform(data_ref), -10, 10)

    n_comps_eff = min(n_comps, ref_mask_j.sum() - 1, data_ref.shape[1] - 1)
    pca_obj = PCA(n_components=n_comps_eff, random_state=random_state)
    pca_obj.fit(data_ref_scaled)

    X_ref = pca_obj.transform(data_ref_scaled)
    n_pcs_eff = min(n_pcs, X_ref.shape[1])
    X_ref = X_ref[:, :n_pcs_eff]
    y_ref = np.asarray(healthy_mask_j[ref_mask_j]).astype(int)

    # 5. CV metrics + OOF
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_res     = cross_validate(clf, X_ref, y_ref, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    logreg_oof = cross_val_predict(clf, X_ref, y_ref, cv=cv, method="predict_proba")[:, 1]

    knn_oof = np.zeros(len(X_ref))
    for train_idx, val_idx in cv.split(X_ref, y_ref):
        nn_fold = NearestNeighbors(n_neighbors=min(k, len(train_idx)))
        nn_fold.fit(X_ref[train_idx])
        knn_oof[val_idx] = y_ref[train_idx][
            nn_fold.kneighbors(X_ref[val_idx], return_distance=False)
        ].mean(axis=1)

    print("\n=== Joint LogReg+KNN (PCA) — CV Performance ===")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {m}: {cv_res[f'test_{m}'].mean():.4f} ± {cv_res[f'test_{m}'].std():.4f}")

    ref_pred_oof = ((logreg_oof >= prob_thresh) | (knn_oof >= neighbor_thresh)).astype(int)
    print(f"\n=== OOF Audit (joint, threshold={prob_thresh}) ===")
    print(f"  Healthy as tumor: {((y_ref==1)&(ref_pred_oof==0)).sum()} / {(y_ref==1).sum()}")
    print(f"  Tumor as healthy: {((y_ref==0)&(ref_pred_oof==1)).sum()} / {(y_ref==0).sum()}")

    # 6. Fit full models on all refs for candidate prediction
    clf.fit(X_ref, y_ref)
    nn_full = NearestNeighbors(n_neighbors=min(k, X_ref.shape[0]))
    nn_full.fit(X_ref)

    ref_obs_j    = adata_joint.obs_names[ref_mask_j]
    logreg_oof_map = dict(zip(ref_obs_j, logreg_oof))
    knn_oof_map    = dict(zip(ref_obs_j, knn_oof))

    # 7. Helper: project slice → score → plot
    def _score_and_plot(slice_adata, tumor_cells, slide_id, slice_num, use_oof=False):
        slice_aligned = slice_adata[:, joint_var_names].copy()
        data_s = slice_aligned.X.toarray() if issparse(slice_aligned.X) else np.asarray(slice_aligned.X)
        X_proj = pca_obj.transform(np.clip(scaler.transform(data_s), -10, 10))[:, :n_pcs_eff]

        is_cand = np.asarray(slice_adata.obs_names.isin(tumor_cells))
        if is_cand.sum() == 0:
            print(f"  No candidates in slice {slice_num}, skipping")
            return

        logreg_sc = np.ones(slice_adata.n_obs)
        knn_sc    = np.ones(slice_adata.n_obs)

        # OOF for known ref cells
        if use_oof:
            for i, name in enumerate(slice_adata.obs_names):
                if name in logreg_oof_map:
                    logreg_sc[i] = logreg_oof_map[name]
                    knn_sc[i]    = knn_oof_map[name]

        # Fresh predictions for candidates
        X_cand = X_proj[is_cand]
        nn_idx = nn_full.kneighbors(X_cand, return_distance=False)
        logreg_sc[is_cand] = clf.predict_proba(X_cand)[:, 1]
        knn_sc[is_cand]    = y_ref[nn_idx].mean(axis=1)

        is_healthy_pred = is_cand & ((logreg_sc >= prob_thresh) | (knn_sc >= neighbor_thresh))
        is_tumor_kept   = is_cand & ~is_healthy_pred
        is_background   = ~is_cand

        n_cand = is_cand.sum()
        n_kept = is_tumor_kept.sum()
        n_rej  = is_healthy_pred.sum()
        print(f"  {slide_id} slice {slice_num}: {n_cand} candidates -> {n_kept} tumor, {n_rej} rejected")

        obs = slice_adata.obs
        plt.figure(figsize=(10, 10))
        plt.scatter(obs.loc[is_background,   CENTER_X_GLOBAL_PX], obs.loc[is_background,   CENTER_Y_GLOBAL_PX], c="#E0E0E0", s=0.7, alpha=0.3, edgecolors="none")
        plt.scatter(obs.loc[is_healthy_pred, CENTER_X_GLOBAL_PX], obs.loc[is_healthy_pred, CENTER_Y_GLOBAL_PX], c="#607D8B", s=1.2, alpha=0.5, label="Rejected (Healthy)")
        plt.scatter(obs.loc[is_tumor_kept,   CENTER_X_GLOBAL_PX], obs.loc[is_tumor_kept,   CENTER_Y_GLOBAL_PX], c="red",     s=2.0, alpha=0.9, label="Refined Tumor")
        plt.title(f"{slide_id} slice {slice_num} | LogReg+KNN (PCA, Joint)\n{n_cand} candidates → {n_kept} tumor, {n_rej} rejected", fontsize=14)
        plt.axis("equal")
        plt.axis("off")
        plt.legend(loc="upper right", markerscale=4)
        plt.show()

    # 8. Primary tumor slices
    for sid in slide_ids:
        snum = primary_slices[sid]
        print(f"\n--- {sid} slice {snum} (primary) ---")
        slice_adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/slice_{snum}_adata.h5ad")
        sc.pp.normalize_total(slice_adata, target_sum=1e4)
        sc.pp.log1p(slice_adata)
        _score_and_plot(slice_adata, _get_tumor_candidates_ids(sid), sid, snum, use_oof=True)

    # 9. Remaining slices
    slices_dir = pathlib.Path(rf"{BASE_DIR}/resources/cache/")
    for sid, nums in remaining_slices_map.items():
        for snum in nums:
            slice_path = slices_dir / f"slice_{snum}_adata.h5ad"
            ann_path   = pathlib.Path(rf"{BASE_DIR}/outputs/cell_annotation/{sid}/05/{snum}/slice_{snum}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
            if not slice_path.exists() or not ann_path.exists():
                print(f"Skipping {sid} slice {snum}: missing data")
                continue
            print(f"\n--- {sid} slice {snum} ---")
            slice_adata = ad.read_h5ad(slice_path)
            sc.pp.normalize_total(slice_adata, target_sum=1e4)
            sc.pp.log1p(slice_adata)
            annotation_df = pd.read_csv(ann_path)
            tumor_cells = _get_tumor_candidates_ids(sid, annotation_df)
            print(f"  {len(tumor_cells)} candidates")
            _score_and_plot(slice_adata, tumor_cells, sid, snum)

    return {"scaler": scaler, "pca": pca_obj, "clf": clf, "nn": nn_full}


def _de_one_slide(slide_id, min_cells=5):
    adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_{slide_id}_adata.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    healthy_ids  = _get_healthy_ref_ids(slide_id)
    tumor_ids    = _get_tumor_ref_ids(slide_id)
    healthy_mask = np.asarray(adata.obs_names.isin(healthy_ids))
    tumor_mask   = np.asarray(adata.obs_names.isin(tumor_ids))

    X   = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    X_h = X[healthy_mask]
    X_t = X[tumor_mask]
    genes = adata.var_names.tolist()
    n_genes = len(genes)

    print(f"  {slide_id}: {healthy_mask.sum()} healthy, {tumor_mask.sum()} tumor ref cells")

    pvals  = np.full(n_genes, np.nan)
    log2fc = np.full(n_genes, np.nan)
    for i in range(n_genes):
        h, t = X_h[:, i], X_t[:, i]
        # require detection in EITHER group (not both)
        if (h > 0).sum() < min_cells and (t > 0).sum() < min_cells:
            continue
        # proper log2FC on linear scale
        mean_h = np.expm1(h).mean()
        mean_t = np.expm1(t).mean()
        log2fc[i] = np.log2((mean_t + 1) / (mean_h + 1))
        # non-parametric test
        try:
            _, pvals[i] = mannwhitneyu(t, h, alternative='two-sided')
        except ValueError:
            pvals[i] = 1.0  # all-equal case

    tested = ~np.isnan(pvals)
    padj = np.full(n_genes, np.nan)
    if tested.sum() > 0:
        _, padj[tested], _, _ = multipletests(pvals[tested], method="fdr_bh")

    return pd.DataFrame({"gene": genes, "log2fc": log2fc, "padj": padj}).set_index("gene")


def _volcano_ax(ax, log2fc, neg_log10, colors, title, fc_thresh, fdr_thresh, genes, label_idx):
    ax.scatter(log2fc, neg_log10, c=colors, s=8, alpha=0.6, linewidths=0)
    ax.axhline(-np.log10(fdr_thresh), color="black", lw=1, ls="--")
    ax.axvline( fc_thresh, color="black", lw=1, ls="--")
    ax.axvline(-fc_thresh, color="black", lw=1, ls="--")
    for i in label_idx:
        ax.text(log2fc[i], neg_log10[i], genes[i], fontsize=6, alpha=0.8)
    ax.set_xlabel("log2 Fold-Change (Tumor − Healthy)")
    ax.set_ylabel("−log10(FDR-adjusted p-value)")
    ax.set_title(title)


def plot_volcano(slide_ids=("L321", "L34"), min_cells=5, fc_thresh=1.0, fdr_thresh=0.05):
    dfs = {sid: _de_one_slide(sid, min_cells) for sid in slide_ids}

    common_genes = dfs[slide_ids[0]].index.intersection(dfs[slide_ids[1]].index)
    df = {sid: dfs[sid].loc[common_genes] for sid in slide_ids}

    def _sig(d):
        return (d["padj"] < fdr_thresh) & (d["log2fc"].abs() > fc_thresh)

    sig = {sid: _sig(df[sid]) for sid in slide_ids}
    concordant = (
        sig[slide_ids[0]] & sig[slide_ids[1]]
        & (np.sign(df[slide_ids[0]]["log2fc"]) == np.sign(df[slide_ids[1]]["log2fc"]))
    )

    genes = common_genes.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for ax, sid in zip(axes[:2], slide_ids):
        lfc       = df[sid]["log2fc"].values
        neg_log10 = -np.log10(df[sid]["padj"].fillna(1).values + 1e-300)
        this_sig  = sig[sid].values
        conc      = concordant.values

        colors = np.where(
            conc & (lfc > 0), "firebrick",
            np.where(conc & (lfc < 0), "steelblue",
            np.where(this_sig & ~conc & (lfc > 0), "#FFAAAA",
            np.where(this_sig & ~conc & (lfc < 0), "#AACFFF",
            "lightgray"))))

        conc_idx     = np.where(conc)[0]
        specific_idx = np.where(this_sig & ~conc)[0]
        top_conc     = conc_idx[np.argsort(neg_log10[conc_idx])[-10:]]     if len(conc_idx)     else np.array([], dtype=int)
        top_specific = specific_idx[np.argsort(neg_log10[specific_idx])[-10:]] if len(specific_idx) else np.array([], dtype=int)
        label_idx    = np.concatenate([top_conc, top_specific])

        n_conc = int(conc.sum())
        n_spec = int((this_sig & ~conc).sum())
        _volcano_ax(
            ax, lfc, neg_log10, colors,
            f"{sid} — {n_conc} concordant (solid), {n_spec} slide-specific (light)",
            fc_thresh, fdr_thresh, genes, label_idx,
        )

    # intersection plot
    sid0, sid1 = slide_ids
    lfc       = ((df[sid0]["log2fc"] + df[sid1]["log2fc"]) / 2).values
    padj_max  = np.fmax(df[sid0]["padj"].fillna(1).values, df[sid1]["padj"].fillna(1).values)
    neg_log10 = -np.log10(padj_max + 1e-300)
    conc      = concordant.values

    colors = np.where(
        conc & (lfc > 0), "firebrick",
        np.where(conc & (lfc < 0), "steelblue", "lightgray"),
    )
    conc_idx  = np.where(conc)[0]
    top_idx   = conc_idx[np.argsort(neg_log10[conc_idx])[-20:]] if len(conc_idx) else np.array([], dtype=int)
    _volcano_ax(
        axes[2], lfc, neg_log10, colors,
        f"Concordant across both slides — {int(conc.sum())} genes",
        fc_thresh, fdr_thresh, genes, top_idx,
    )

    plt.suptitle("Tumor vs Healthy Reference Cells", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return pd.DataFrame({
        f"log2fc_{sid0}": df[sid0]["log2fc"],
        f"padj_{sid0}":   df[sid0]["padj"],
        f"log2fc_{sid1}": df[sid1]["log2fc"],
        f"padj_{sid1}":   df[sid1]["padj"],
        "concordant":     concordant,
    }, index=common_genes)


def _get_volcano_genes(result_df):
    result_df = result_df.copy()
    result_df["mean_log2fc"] = (result_df["log2fc_L321"] + result_df["log2fc_L34"]) / 2
    result_df["max_padj"] = np.fmax(result_df["padj_L321"].fillna(1),
                                    result_df["padj_L34"].fillna(1))

    # Combined score: significance × magnitude
    result_df["score"] = result_df["mean_log2fc"].abs() * -np.log10(result_df["max_padj"] + 1e-300)

    # Filter to concordant hits and sort
    ranked = (result_df[result_df["concordant"]]
              .sort_values("score", ascending=False))

    # Just the gene names, ordered
    gene_list = ranked.index.tolist()
    print(gene_list)

def _get_per_slide_genes(result_df):
    fc_thresh, fdr_thresh = 1.0, 0.05

    # Per-slide significance
    sig_L321 = (result_df["padj_L321"] < fdr_thresh) & (result_df["log2fc_L321"].abs() > fc_thresh)
    sig_L34 = (result_df["padj_L34"] < fdr_thresh) & (result_df["log2fc_L34"].abs() > fc_thresh)

    # Slide-specific = significant in this slide but NOT concordant across both
    specific_L321 = result_df[sig_L321 & ~result_df["concordant"]]
    specific_L34 = result_df[sig_L34 & ~result_df["concordant"]]

    # Concordant (for comparison)
    concordant = result_df[result_df["concordant"]]

    print(f"L321-specific: {len(specific_L321)} genes")
    print(f"L34-specific:  {len(specific_L34)} genes")
    print(f"Concordant:    {len(concordant)} genes")
    l321_specific_genes = (specific_L321
                           .sort_values("padj_L321")
                           .index.tolist())

    l34_specific_genes = (specific_L34
                          .sort_values("padj_L34")
                          .index.tolist())

    print("L321-specific:", l321_specific_genes)
    print("L34-specific:", l34_specific_genes)


def _plot_ref_cells_spatial(slide_id):
    adata = ad.read_h5ad(rf"{BASE_DIR}/resources/cache/sample_{slide_id}_adata.h5ad")

    healthy_ids = _get_healthy_ref_ids(slide_id)
    tumor_ids   = _get_tumor_ref_ids(slide_id)

    healthy_mask = np.asarray(adata.obs_names.isin(healthy_ids))
    tumor_mask   = np.asarray(adata.obs_names.isin(tumor_ids))
    background   = ~(healthy_mask | tumor_mask)

    obs = adata.obs
    plt.figure(figsize=(12, 12))

    # Background — all other cells
    plt.scatter(
        obs.loc[background, CENTER_X_GLOBAL_PX],
        obs.loc[background, CENTER_Y_GLOBAL_PX],
        c="#D9D9D9", s=0.5, alpha=0.3, edgecolors="none",
    )
    # Healthy reference cells
    plt.scatter(
        obs.loc[healthy_mask, CENTER_X_GLOBAL_PX],
        obs.loc[healthy_mask, CENTER_Y_GLOBAL_PX],
        c="dodgerblue", s=4, alpha=0.9, edgecolors="none",
        label=f"Healthy ref ({healthy_mask.sum()})",
    )
    # Tumor reference cells
    plt.scatter(
        obs.loc[tumor_mask, CENTER_X_GLOBAL_PX],
        obs.loc[tumor_mask, CENTER_Y_GLOBAL_PX],
        c="firebrick", s=4, alpha=0.9, edgecolors="none",
        label=f"Tumor ref ({tumor_mask.sum()})",
    )

    plt.title(f"{slide_id} — Healthy vs Tumor Reference Cells (spatial)", fontsize=14)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(markerscale=4, frameon=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _plot_ref_cells_spatial("L321")
    _plot_ref_cells_spatial("L34")
    result_df = plot_volcano()
    _get_volcano_genes(result_df)
    _get_per_slide_genes(result_df)
    run_identify_tumor_cells_joint()
