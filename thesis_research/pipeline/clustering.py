import anndata as ad
import scanpy as sc
from anndata import AnnData
from anndata import AnnData
from matplotlib import pyplot as plt
import colorcet as cc


def clustering_no_pca():
    adata = ad.read_h5ad("D:\\thesis-research\\resources\\adata_full.h5ad")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # optional: try both with and without scaling
    # sc.pp.scale(adata, max_value=10)

    sc.pp.neighbors(
        adata,
        use_rep="X",
        metric="cosine",
        n_neighbors=30,
    )
    print("leiden...")
    sc.tl.leiden(adata, resolution=0.45, key_added="cluster", random_state=42)
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    print("umap...")
    sc.tl.umap(adata, random_state=42)

    sc.pl.umap(
        adata,
        color="cluster",
        palette=cc.glasbey,
        legend_loc="right margin",
        size=2.0,
        alpha=0.8,
        title="Clusters (no PCA)",
        show=False,
    )
    print("plotting...")
    plt.savefig(
        "D:\\thesis-research\\resources\\umap_clusters_no_pca.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# def run_clustering_pipeline():
#     print(f"🔵 Starting clustering pipeline...")
#     adata = _load_merged_adata()
#
#
#
#
#
#
#
#
#
# def _load_slice_adatas() -> list[AnnData]:
#     slice_adatas = []
#     for slice_adata_path in sorted(CACHE_DIR_PATH.glob("slice_*_adata.h5ad")):
#         slice_id = slice_adata_path.stem.split("_")[1]
#         print(f"🕒 Loading adata for slice {slice_id} from cache...")
#         adata = read_h5ad(slice_adata_path)
#         adata.uns[SLICE_ID] = slice_id
#         slice_adatas.append(adata)
#     return slice_adatas
#
#
#
# def _load_merged_adata() -> AnnData:
#     print(f"🕒 Loading merged adata from cache...")
#     adata = ad.read_h5ad(CACHE_DIR_PATH / "merged_adata.h5ad")
#     return adata
#
#
# def run_clustering_on_adata(adata: AnnData) -> None:
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#
#     sc.pp.scale(adata, max_value=10)
#
#     sc.tl.pca(adata, n_comps=50, random_state=42)
#
#     sc.pp.neighbors(adata, n_pcs=50, metric='cosine', random_state=42)
#     sc.tl.leiden(adata, resolution=0.45, key_added="cluster", random_state=42)
#     sc.tl.umap(adata, random_state=42)
#
#     sc.pl.umap(adata, color="sample_id", legend_loc="right margin")
#
#     sc.pl.umap(
#         adata,
#         color="cluster",
#         legend_loc="right margin",
#         title="Clusters"
#     )

clustering_no_pca()
