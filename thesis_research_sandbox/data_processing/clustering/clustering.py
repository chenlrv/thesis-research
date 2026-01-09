import scanpy as sc
from anndata import AnnData

from data_processing.clustering.allen_brain_atlas_cell_type import get_allen_brain_atlas_cell_type_markers
from data_processing.clustering.mouse_brain_org_cell_type import get_cell_type_marker_genes
from data_processing.imaging.plot_spatial_clusters import plot_spatial_clusters


def cluster(adata: AnnData) -> None:
    print(f'Starting clustering process for {len(adata.obs)} cells and {len(adata.var)} genes.')
    _create_clusters(adata)

    adata_allen_labeling = adata.copy()
    _label_clusters_by_allen_brain_atlas(adata_allen_labeling)
    plot_spatial_clusters(adata_allen_labeling, 'allen brain atlas')

    adata_mouse_brain_labeling = adata.copy()
    _label_clusters_by_mouse_brain_org(adata_mouse_brain_labeling)
    plot_spatial_clusters(adata_mouse_brain_labeling, 'mouse brain org')

    # _filter_microglia(adata)


def _create_clusters(adata: AnnData) -> None:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=50, random_state=42)

    sc.pp.neighbors(adata, n_pcs=30, metric='cosine', random_state=42)
    sc.tl.umap(adata, random_state=42)
    sc.tl.leiden(adata, resolution=0.45, key_added="cluster", random_state=42)

    sc.pl.umap(adata, color="sample_id", legend_loc="right margin")

    sc.pl.umap(
        adata,
        color="cluster",
        legend_loc="right margin",
        title="Clusters"
    )


def _label_clusters_by_allen_brain_atlas(adata: AnnData) -> None:
    markers = get_allen_brain_atlas_cell_type_markers()
    _label_clusters(adata, markers)


def _label_clusters_by_mouse_brain_org(adata: AnnData) -> None:
    markers = get_cell_type_marker_genes()
    _label_clusters(adata, markers)


def _label_clusters(adata: AnnData, markers: dict[str, list]):
    var_names = {gene.strip().lower() for gene in adata.var_names}

    for cell_type, genes in markers.items():
        if not genes or not any(gene.strip().lower() in var_names for gene in genes):
            continue
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"{cell_type}_score")

    cluster_labels = {}
    score_cols = [f'{cell_type}_score' for cell_type in markers if f'{cell_type}_score' in adata.obs.columns]

    for cluster in sorted(adata.obs["cluster"].unique(), key=int):
        sub = adata[adata.obs["cluster"] == cluster]
        mean_scores = sub.obs[score_cols].mean()
        cluster_labels[cluster] = mean_scores.idxmax().replace("_score", "")

    adata.obs["celltype"] = adata.obs["cluster"].map(cluster_labels)

    sc.pl.umap(adata, color="celltype", legend_loc="on data")


def _filter_microglia(adata: AnnData):
    micro = adata[adata.obs["celltype"] == "Microglia"].copy()
    print("Microglia shape before QC:", micro.shape)

    print("micro shape before PCA:", micro.shape)

    neg_probes = [g for g in adata.var_names
                  if "neg" in g.lower()]

    sc.tl.pca(micro, n_comps=30)
    sc.pp.neighbors(micro, n_pcs=15)
    sc.tl.umap(micro)
    sc.tl.leiden(micro, resolution=0.4, key_added="micro_cluster")

    sc.pl.umap(micro, color="micro_cluster", legend_loc="on data")

    df = micro.to_df()

    cluster_means = (
        df[neg_probes]
        .groupby(micro.obs["micro_cluster"])
        .mean()
    )

    bad_cluster = cluster_means.mean(axis=1).idxmax()

    bad_cells = micro.obs[micro.obs["micro_cluster"] == bad_cluster].index
    len(bad_cells)

    adata_clean = adata[~adata.obs.index.isin(bad_cells)].copy()
    adata_clean
