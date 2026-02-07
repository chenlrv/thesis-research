import anndata
import pandas as pd
import scanpy as sc
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt

import umap
import matplotlib.colors as mcolors

from thesis_research.utils.columns import CELL_ID_UNIQUE
from thesis_research.utils.entity_type import (
    GlobalEntityType,
    get_global_resource_path,
    get_output_dir,
)


def copy_slice_info(source_adata: AnnData):
    source_adata = anndata.read_h5ad("D:/thesis-research/resources/adata_full.h5ad")
    target_adata = anndata.read_h5ad(
        "D:/thesis-research/resources/adata_slim.h5ad"
    )  # todo load adata_new here and copy slice info from source_adata, so it will hold everything

    if (
        "cell_id_unique" not in source_adata.obs.columns
        or "cell_id_unique" not in target_adata.obs.columns
    ):
        raise KeyError("Both source and target AnnData must have 'cell_id_unique' in .obs")

    # Create a DataFrame for mapping
    mapping_df = source_adata.obs[
        [
            "cell_id_unique",
            "CenterX_global_px",
            "CenterY_global_px",
            "sample_ID",
            "slice_type",
            "slice_ID",
        ]
    ]

    target_adata.obs["CenterX_global_px"] = target_adata.obs["cell_id_unique"].map(
        mapping_df.set_index("cell_id_unique")["CenterX_global_px"]
    )
    target_adata.obs["CenterY_global_px"] = target_adata.obs["cell_id_unique"].map(
        mapping_df.set_index("cell_id_unique")["CenterY_global_px"]
    )

    target_adata.obs["sample_ID"] = target_adata.obs["cell_id_unique"].map(
        mapping_df.set_index("cell_id_unique")["sample_ID"]
    )

    target_adata.obs["slice_ID"] = target_adata.obs["cell_id_unique"].map(
        mapping_df.set_index("cell_id_unique")["slice_ID"]
    )
    target_adata.obs["slice_type"] = target_adata.obs["cell_id_unique"].map(
        mapping_df.set_index("cell_id_unique")["slice_type"]
    )

    target_adata.write_h5ad(
        "D:/thesis-research/resources/adata_with_slices.h5ad", compression="gzip"
    )

    target_adata.obs["slice_ID"] = target_adata.obs["slice_ID"].astype("category")
    sc.pl.umap(
        target_adata,
        color="slice_ID",  # "tumor" / "healthy"
    )

    sc.pl.umap(
        target_adata,
        color="slice_type",  # "tumor" / "healthy"
    )

    print("done")


def run_pca(adata: AnnData):
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=50, random_state=42)

    sc.pp.neighbors(adata, n_pcs=30, metric="cosine", random_state=42)
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(adata, resolution=0.45, key_added="cluster", random_state=42)

    sc.pl.umap(adata, color="sample_id", legend_loc="right margin")

    sc.pl.umap(adata, color="cluster", legend_loc="right margin", title="Clusters")


def run_umap(run_id: str) -> AnnData:
    print(f"🕒 Starting UMAP...")

    pca_df = pd.read_csv(
        get_global_resource_path(GlobalEntityType.X_PCA_PEARSON_BATCH_FILE), index_col=0
    )
    X_pca = pca_df.to_numpy(dtype=np.float32)

    metadata = pd.read_csv(get_global_resource_path(GlobalEntityType.METADATA_FILE), index_col=0)
    metadata = metadata.set_index(CELL_ID_UNIQUE).loc[pca_df.index]

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.01,
        metric="cosine",
        n_components=2,
        random_state=0,
        low_memory=True,
    )

    X_umap = reducer.fit_transform(X_pca)

    adata = AnnData(X=X_pca)
    adata.obs = metadata.copy()
    adata.obsm["X_umap"] = X_umap

    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X", metric="cosine")

    sc.tl.leiden(adata, resolution=1.2, key_added="cluster", random_state=0)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    sc.pl.umap(
        adata, color="cluster", legend_loc="right margin", title="Clusters", ax=ax, show=False
    )

    fig.tight_layout()
    fig.savefig(get_output_dir(run_id) / "umap_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return adata


def distinct_palette_hex(n: int) -> list[str]:
    hues = np.linspace(0, 1, n, endpoint=False).tolist()
    return [mcolors.to_hex(mcolors.hsv_to_rgb((float(h), 0.75, 0.9))) for h in hues]


def run_clustering():
    cluster_labels = {
        0: "Endothelial_Arterial",
        1: "Neuron",
        2: "T_cell",
        3: "B_cell",
        4: "Meningeal_Stromal_Ccl19",
        5: "Endothelial_Blood",
        6: "Mural_VSMC",
        7: "Endothelial_Venous",
        8: "Fibroblast",
        9: "Myeloid_APC",
        10: "Cycling",
        11: "Vascular_Remodeling",
        12: "VLMC_Ptgds",
        13: "Lymphatic_Endothelial",
        14: "pDC_like",
        15: "Myeloid_like",
        16: "Perivascular_Fibroblast_ECM",
        17: "Astrocyte_Reactive",
        18: "NK_Cytotoxic",
        19: "Pericyte",
        20: "ISG_state",
        21: "Microglia_APC",
        22: "Barrier_Epithelial",
        23: "ChoroidPlexus",
        24: "Microglia_ISG",
    }

    celltype_order = [cluster_labels[i] for i in range(25)]

    adata = anndata.read_h5ad("D:/thesis-research/resources/adata_with_slices.h5ad")
    cluster_key = "cluster"  # or "leiden"
    cats = [str(i) for i in range(25)]
    adata.obs[cluster_key] = adata.obs[cluster_key].astype(str)
    adata.obs[cluster_key] = pd.Categorical(adata.obs[cluster_key], categories=cats, ordered=True)

    adata.uns[f"{cluster_key}_colors"] = distinct_palette_hex(25)

    adata.obs["cell_type"] = adata.obs["cluster"].map(
        {str(k): v for k, v in cluster_labels.items()}
    )
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    print("Cluster counts:\n", adata.obs[cluster_key].value_counts().sort_index())
    print("Cell type counts:\n", adata.obs["cell_type"].value_counts(dropna=False))

    try:
        from colorcet import glasbey  # pip install colorcet

        palette = list(glasbey[:25])
        print("Using colorcet.glasbey palette")
    except Exception:
        palette = sc.pl.palettes.zeileis_28[:25]
        print("Using scanpy zeileis_28 palette (fallback)")

    adata.uns["cell_type_colors"] = palette

    umap_point_size = 10
    legend_dot_size = 14

    fig, ax = plt.subplots(1, 1, figsize=(20, 12), dpi=300)

    sc.pl.umap(
        adata,
        color="cell_type",
        legend_loc=None,  # no scanpy legend
        frameon=False,
        size=umap_point_size,
        ax=ax,
        show=False,
    )

    # custom legend outside axes, with big dots
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=legend_dot_size,
            markerfacecolor=palette[i],
            markeredgecolor="none",
        )
        for i in range(25)
    ]

    ax.legend(
        handles,
        celltype_order,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=10,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.subplots_adjust(right=0.72)

    out = "umap_cell_type_only_noclip_bigDots.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.show()
    print("Saved:", out)

    for sample_id in adata.obs["sample_ID"].cat.categories:
        adata_sample = adata[adata.obs["sample_ID"] == sample_id].copy()
        plot_spatial_clusters(adata_sample, f"Spatial Cell Type Plot - Sample {sample_id}")

    # sc.pl.umap(
    #     adata,
    #     color="cell_type",
    #     legend_loc="right margin",
    #     frameon=False
    # )
    # sc.pl.umap(adata, color=["cluster", "cell_type"], wspace=0.4, frameon=False)


def plot_spatial_clusters(
    adata,
    plot_title: str,
    color_col="cell_type",
    x_key="CenterX_global_px",
    y_key="CenterY_global_px",
    s=0.5,
    alpha=1.0,
    dpi=500,
    figsize=(12, 12),
    invert_y=True,
    savepath=None,
):
    adata.obs[color_col] = adata.obs[color_col].astype("category")
    cats = list(adata.obs[color_col].cat.categories)

    palette_key = f"{color_col}_colors"
    if palette_key not in adata.uns:
        raise KeyError(
            f"'{palette_key}' not found in adata.uns. "
            f"Run sc.pl.umap(adata, color='{color_col}', show=False) once."
        )

    colors = list(adata.uns[palette_key])
    cat2color = dict(zip(cats, colors))

    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()
    labels = adata.obs[color_col].astype(str).to_numpy()
    point_colors = pd.Series(labels).map(cat2color).to_numpy()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, c=point_colors, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")

    # ax.invert_xaxis()
    # ax.invert_yaxis()

    # if invert_y:
    #     ax.invert_yaxis()
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Spatial {plot_title} plot colored by {color_col}")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("Saved:", savepath)
    plt.show()


# first run clustering on the method of atmox
# compare my clusters to their clusters

# run pca, clustrting on every sample individually
# check the differences
# check if they have cell type annotation
# show the annotated cells on the spatial location

# copy_slice_info(None)
# run_clustering()
# adata = anndata.read_h5ad('D:/thesis-research/resources/clustered_adata.h5ad')
# run_clustering(adata)
