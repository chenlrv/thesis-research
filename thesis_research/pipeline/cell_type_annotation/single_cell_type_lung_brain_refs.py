import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
from scipy.spatial import cKDTree
import scanpy as sc
from anndata import AnnData
from scipy.stats import chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from thesis_research.utils.columns import CENTER_Y_GLOBAL_PX, CENTER_X_GLOBAL_PX


def plot_tumor_old():
    adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\slice_1_adata.h5ad")
    df_results = pd.read_csv(
        r"/outputs/cell_annotation/L321/05/1/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
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

    adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]

    adata.obsm["spatial"] = np.stack(
        [adata.obs[CENTER_X_GLOBAL_PX].values, adata.obs[CENTER_Y_GLOBAL_PX].values], axis=1
    )

    adata.obs["predicted_cell_type"] = adata.obs["predicted_cell_type"].astype("category")
    is_tumor = adata.obs["predicted_cell_type"] == "Tumor"

    coords = np.c_[
        adata.obs[CENTER_X_GLOBAL_PX].to_numpy(), adata.obs[CENTER_Y_GLOBAL_PX].to_numpy()
    ]

    if filter_by_surrounding:
        tumor_coords = coords[is_tumor]

        adata.obs["is_surrounded_by_tumor"] = False

        # need at least k+1 tumor cells total (because each tumor cell counts itself)
        if tumor_coords.shape[0] >= (k + 1):
            tree = cKDTree(tumor_coords)

            dists, _ = tree.query(tumor_coords, k=k + 1)  # shape (n_tumor, 6)
            # dists[:,0] ~= 0 (self). 5 other tumors => 6th nearest => index k (5)
            surrounded_tumor = dists[:, k] <= r

            # write back only for tumor cells
            adata.obs.loc[is_tumor, "is_surrounded_by_tumor"] = surrounded_tumor

        cells_to_keep = is_tumor & adata.obs["is_surrounded_by_tumor"]
        is_background = ~cells_to_keep
    else:
        cells_to_keep = adata.obs["predicted_cell_type"] == "Tumor"
        cells_to_keep = (adata.obs["predicted_cell_type"] == "Tumor") & (
            ~adata.obs_names.isin(overlap_cells)
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

    plt.title(f"L321 slice 1 spatial Mapping tumor cells", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def plot_cell_type(
    sample_id: str,
    slice_id: int,
    cell_type: str,
    color: str,
    abs_threshold: float = None,
    delta_threshold: float = None,
):
    # Use slice_id to build paths
    adata = ad.read_h5ad(rf"D:\thesis-research\resources\cache\slice_{slice_id}_adata.h5ad")
    df_results = pd.read_csv(
        rf"D:\thesis-research\outputs\cell_annotation\{sample_id}\05\slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )

    df_results = df_results[df_results["predicted_cell_type"] == cell_type]

    score_columns = ["score_brain_struct", "score_brain_immune", "score_tumor"]

    df_results["next_best_score"] = df_results[score_columns].apply(
        lambda x: sorted(x, reverse=True)[1], axis=1
    )
    df_results["delta_score"] = df_results["final_confidence"] - df_results["next_best_score"]

    df_results = df_results[(df_results["predicted_cell_type"] == cell_type)]
    if abs_threshold:
        df_results = df_results[df_results["final_confidence"] > abs_threshold]

    if delta_threshold:
        df_results = df_results[df_results["delta_score"] > delta_threshold]

    adata.obs = adata.obs.merge(
        df_results[["cell_barcode", "predicted_cell_type"]],
        left_index=True,
        right_on="cell_barcode",
        how="left",
    ).set_index("cell_barcode")

    adata.obs["predicted_cell_type"] = adata.obs["predicted_cell_type"].astype("category")

    adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]

    adata.obsm["spatial"] = np.stack(
        [adata.obs[CENTER_X_GLOBAL_PX].values, adata.obs[CENTER_Y_GLOBAL_PX].values], axis=1
    )

    is_cell_type = adata.obs["predicted_cell_type"] == cell_type
    is_background = ~is_cell_type

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
        adata.obs.loc[is_cell_type, CENTER_X_GLOBAL_PX],
        adata.obs.loc[is_cell_type, CENTER_Y_GLOBAL_PX],
        c=color,
        s=1.5,
        alpha=0.8,
        edgecolors="none",
        label="D122 Tumor Cells",
    )

    plt.title(
        f"{sample_id} slice {slice_id} control spatial Mapping - {cell_type} cells", fontsize=15
    )
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def plot_tumor_cells(
    sample_id: str,
    slice_id: int,
    abs_threshold: float = 0.2,
    delta_threshold: float = 0.08,
    k: int = 5,
    r: float = 500,
    filter_by_surrounding: bool = False,
):
    adata = ad.read_h5ad(rf"D:\thesis-research\resources\cache\slice_{slice_id}_adata.h5ad")
    df_results = pd.read_csv(
        rf"D:\thesis-research\outputs\cell_annotation\{sample_id}\05\slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    df_results = df_results[df_results["predicted_cell_type"] == "Tumor"]
    df_results["next_best_score"] = df_results[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results["delta_score"] = abs(df_results["score_tumor"] - df_results["next_best_score"])

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

    adata.obs[CENTER_Y_GLOBAL_PX] = -adata.obs[CENTER_Y_GLOBAL_PX]
    adata.obs[CENTER_X_GLOBAL_PX] = -adata.obs[CENTER_X_GLOBAL_PX]

    adata.obsm["spatial"] = np.stack(
        [adata.obs[CENTER_X_GLOBAL_PX].values, adata.obs[CENTER_Y_GLOBAL_PX].values], axis=1
    )

    adata.obs["predicted_cell_type"] = adata.obs["predicted_cell_type"].astype("category")
    is_tumor = adata.obs["predicted_cell_type"] == "Tumor"

    adata = add_is_surrounded_by_tumor_column(adata, k, r)

    if filter_by_surrounding:
        cells_to_keep = is_tumor & adata.obs["is_surrounded_by_tumor"]
        is_background = ~cells_to_keep
    else:
        cells_to_keep = adata.obs["predicted_cell_type"] == "Tumor"
        is_background = ~cells_to_keep

    generate_tumor_plot(sample_id, slice_id, adata, is_background)

    adata_rem = adata[cells_to_keep].copy()
    cluster_cells(adata_rem)

    cluster_key = "leiden_remaining"

    for cluster in adata_rem.obs[cluster_key].cat.categories:
        mask_surrounded = (
            (adata_rem.obs["predicted_cell_type"] == "Tumor")
            & (adata_rem.obs[cluster_key] == cluster)
            & (adata_rem.obs["is_surrounded_by_tumor"])
        )
        mask_single = (
            (adata_rem.obs["predicted_cell_type"] == "Tumor")
            & (adata_rem.obs[cluster_key] == cluster)
            & (~adata_rem.obs["is_surrounded_by_tumor"])
        )
        adata_single = adata_rem[mask_single].copy()
        adata_single.write_h5ad(f"{cluster}_cluster_single.h5ad")
        adata_surrounded = adata_rem[mask_surrounded].copy()
        adata_surrounded.write_h5ad(f"{cluster}_cluster_surrounded.h5ad")

    adata.obs[cluster_key] = pd.NA
    adata.obs.loc[adata_rem.obs_names, cluster_key] = adata_rem.obs[cluster_key].astype(str).values
    adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

    plot_cells_by_clusters(sample_id, slice_id, adata, adata_rem, cells_to_keep, cluster_key)


def add_is_surrounded_by_tumor_column(adata: AnnData, k: int = 5, r: float = 500) -> AnnData:
    adata_copy = adata.copy()
    is_tumor = adata_copy.obs["predicted_cell_type"] == "Tumor"

    coords = np.c_[
        adata_copy.obs[CENTER_X_GLOBAL_PX].to_numpy(), adata_copy.obs[CENTER_Y_GLOBAL_PX].to_numpy()
    ]

    tumor_coords = coords[is_tumor]

    adata_copy.obs["is_surrounded_by_tumor"] = False

    if tumor_coords.shape[0] >= (k + 1):
        tree = cKDTree(tumor_coords)
        dists, _ = tree.query(tumor_coords, k=k + 1)  # shape (n_tumor, 6)
        surrounded_tumor = dists[:, k] <= r

        adata_copy.obs.loc[is_tumor, "is_surrounded_by_tumor"] = surrounded_tumor

    return adata_copy


def generate_tumor_plot(sample_id: str, slice_id: int, adata: AnnData, is_background: np.ndarray):
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

    plt.title(f"{sample_id} slice {slice_id} spatial Mapping tumor cells", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.show()


def cluster_cells(adata: AnnData, cluster_key: str = "leiden_remaining"):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # sc.pp.highly_variable_genes(adata, n_top_genes=50, flavor="seurat_v3")
    # adata_rem = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50)

    # Graph + clustering
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.leiden(adata, resolution=0.3, key_added="leiden_remaining")

    # Optional: UMAP for visualization
    sc.tl.umap(adata)

    sc.pl.umap(adata, color="leiden_remaining", show=False)
    sc.pl.umap(adata, color=["leiden_remaining"], legend_loc="right margin")


def cluster_cells_no_pca(adata: AnnData, cluster_key: str = "leiden_remaining"):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # sc.pp.highly_variable_genes(adata, n_top_genes=50, flavor="seurat_v3")
    # adata_rem = adata[:, adata.var["highly_variable"]].copy()

    # Graph + clustering
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
    sc.tl.leiden(adata, resolution=0.3, key_added=cluster_key)

    # Optional: UMAP for visualization
    sc.tl.umap(adata)

    sc.pl.umap(adata, color="leiden_remaining", show=False)
    sc.pl.umap(adata, color=["leiden_remaining"], legend_loc="right margin")


def plot_cells_by_clusters(
    sample_id: str,
    slice_id: int,
    adata: AnnData,
    adata_rem: AnnData,
    cells_to_keep,
    cluster_key: str = "leiden_remaining",
):
    tumor = cells_to_keep.to_numpy()  # surrounded tumor cells (COLORED)
    excluded = ~tumor  # everything else (GRAY)

    # 3) Build a palette for the clusters present in excluded cells
    cats = list(adata.obs.loc[tumor, cluster_key].cat.categories)
    colors = list(adata_rem.uns[f"{cluster_key}_colors"])
    cat2color = dict(zip(cats, colors))

    cluster_colors = adata.obs.loc[tumor, cluster_key].astype(str).map(cat2color).to_numpy()

    # 4) Plot spatial: gray background + colored surrounded tumor clusters
    X, Y = CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX
    plt.figure(figsize=(10, 10), dpi=300)

    plt.scatter(
        adata.obs.loc[excluded, X],
        adata.obs.loc[excluded, Y],
        c="#CFCFCF",
        s=0.7,
        alpha=1.0,
        edgecolors="none",
        rasterized=True,
    )

    plt.scatter(
        adata.obs.loc[tumor, X],
        adata.obs.loc[tumor, Y],
        c=cluster_colors,
        s=2.0,
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
    )

    plt.title(f"{sample_id} slice {slice_id} tumor cells colored by leiden_remaining", fontsize=15)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=cat2color[c],
            markeredgecolor="none",
            label=str(c),
        )
        for c in cats
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def pca_surrounded_single_healthy_tumor(abs_threshold: float = 0.2, delta_threshold: float = 0.08):
    CELL_COL = "cell_global_id"
    adata3 = ad.read_h5ad(rf"D:\thesis-research\resources\cache\slice_3_adata.h5ad")
    df_results3 = pd.read_csv(
        rf"D:\thesis-research\outputs\cell_annotation\L321\05\slice_3_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    df_results3 = df_results3[df_results3["predicted_cell_type"] == "Tumor"]
    df_results3["next_best_score"] = df_results3[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results3["delta_score"] = abs(df_results3["score_tumor"] - df_results3["next_best_score"])

    df_results3 = df_results3[
        (df_results3["predicted_cell_type"] == "Tumor")
        & (df_results3["score_tumor"] > abs_threshold)  # New: Absolute match filter
        & (df_results3["delta_score"] > delta_threshold)
        & (df_results3["score_tumor"] > df_results3["next_best_score"])
    ]

    df_results3[CELL_COL] = df_results3["cell_barcode"]

    zero_cluster_single = ad.read_h5ad("0_cluster_single.h5ad")
    zero_cluster_surrounded = ad.read_h5ad("0_cluster_surrounded.h5ad")
    one_cluster_single = ad.read_h5ad("1_cluster_single.h5ad")
    one_cluster_surrounded = ad.read_h5ad("1_cluster_surrounded.h5ad")

    dfs = {
        "tumor_cluster0_single": zero_cluster_single.obs,
        "tumor_cluster0_surrounded": zero_cluster_surrounded.obs,
        "tumor_cluster1_single": one_cluster_single.obs,
        "tumor_cluster1_surrounded": one_cluster_surrounded.obs,
        "healthy": df_results3,
    }

    parts = []
    for name, df in dfs.items():
        tmp = df[[CELL_COL]].copy()
        tmp["group"] = name
        parts.append(tmp)

    groups = pd.concat(parts, ignore_index=True)
    groups = groups.set_index(CELL_COL)

    adata_full = ad.read_h5ad(rf"D:\thesis-research\resources\cache\sample_L321_adata.h5ad")

    keep = adata_full.obs_names.intersection(groups.index)
    adata_pca = adata_full[keep].copy()

    adata_pca.obs["group"] = groups.loc[keep, "group"].astype("category")

    # PCA
    sc.pp.normalize_total(adata_pca, target_sum=1e4)
    sc.pp.log1p(adata_pca)
    sc.pp.scale(adata_pca, max_value=10)
    sc.tl.pca(adata_pca, n_comps=50, random_state=42)

    ax = sc.pl.pca(
        adata_pca,
        color="group",
        title="PCA of tumor subsets",
        legend_loc="right margin",
        show=False,
    )
    plt.savefig("pca_groups.png", dpi=300, bbox_inches="tight")
    plt.close()

    X = adata_pca.obsm["X_pca"][:, :2]
    pc = pd.DataFrame(X, index=adata_pca.obs_names, columns=["PC1", "PC2"])

    healthy_mask = adata_pca.obs["group"] == "healthy"
    mu = pc.loc[healthy_mask].mean().values
    cov = np.cov(pc.loc[healthy_mask].values.T)
    inv = np.linalg.inv(cov)

    diff = pc.values - mu
    md2 = np.einsum("ij,jk,ik->i", diff, inv, diff)  # squared Mahalanobis

    thr = chi2.ppf(0.8, df=2)  # 95% ellipse
    overlap = (~healthy_mask.values) & (md2 <= thr)

    overlap_cells = pc.index[overlap]
    pd.Series(overlap_cells).to_csv(
        "tumor_cells_overlapping_healthy_pca.csv", index=False, header=False
    )
    print("overlap tumor cells:", overlap.sum())

    # assumes you already computed:
    # pc (DataFrame with PC1,PC2 indexed by cell ids)
    # healthy_mask (boolean Series aligned to pc)
    # overlap (boolean array aligned to pc)  # tumor points inside healthy ellipse

    ax = sc.pl.pca(adata_pca, color="group", show=False, title="PCA + overlap tumor cells")

    # overlay overlap cells as black circles (no fill)
    ax.scatter(
        pc.loc[overlap, "PC1"],
        pc.loc[overlap, "PC2"],
        s=20,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        label="tumor overlap",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    plt.tight_layout()
    plt.show()

    cells_to_remove = list(overlap_cells)

    cells_to_keep = ~adata1.obs_names.isin(cells_to_remove)
    is_background = ~cells_to_keep

    generate_tumor_plot("L321", 1, adata1, is_background)

    print()


_get_tumor_ref_mask(None)
plot_tumor_cells(
    "L321", 1, abs_threshold=0.2, delta_threshold=0.08, k=5, r=500, filter_by_surrounding=False
)
pca_surrounded_single_healthy_tumor()
# plot_cell_type("Tumor", "red", abs_threshold=0.2, delta_threshold=0.08)
# plot_tumor_filtered(k=5, r=500)
# plot_cell_type("Tumor", "red", abs_threshold=0.2, delta_threshold=0.08)
# plot_cell_type("astrocyte", "blue", delta_threshold=0.08)
# plot_cell_type("macrophage", "green", abs_threshold=0.15)
# plot_cell_type("microglial cell", "purple", abs_threshold=0.15)
