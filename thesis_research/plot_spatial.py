import anndata as ad

from thesis_research.pipeline.analysis import plot_spatial_clusters

adatas = {
    "scvi": {
        "adata": "D:\\thesis-research\\resources\\adata_scvi.h5ad",
        "color_column": "leiden_scvi",
    },
    "no_pca": {
        "adata": "D:\\thesis-research\\resources\\clustering\\adata_full_no_pca_clustered.h5ad",
        "color_column": "cluster",
    },
    "pca": {
        "adata": "D:\\thesis-research\\resources\\clustering\\adata_full_with_pca_clustered.h5ad",
        "color_column": "cluster",
    },
}


def plot_spatial(adata_path: str, color_column: str) -> None:
    adata = ad.read_h5ad(adata_path)
    for sample_id in adata.obs["sample_id"].cat.categories:
        adata_sample = adata[adata.obs["sample_id"] == sample_id].copy()
        plot_spatial_clusters(
            adata_sample, f"Spatial Cell Type Plot - Sample {sample_id}", color_col=color_column
        )


plot_spatial(adatas["pca"]["adata"], adatas["pca"]["color_column"])
