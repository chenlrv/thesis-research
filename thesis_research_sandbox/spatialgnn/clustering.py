from anndata import AnnData
import scanpy as sc

from common.get_marker_genes import get_marker_genes


def cluster_from_gnn(
        adata: AnnData,
        gnn_key: str = "X_gnn_dgi",
        n_neighbors: int = 15,
        resolution: float = 0.6,
        plot: bool = True,
):
    """
    Cluster ALL cells using GNN embeddings.
    """
    if gnn_key not in adata.obsm:
        raise KeyError(f"{gnn_key} not found in adata.obsm.")

    adata.obsm["X_gnn_used"] = adata.obsm[gnn_key]

    sc.pp.neighbors(adata, use_rep="X_gnn_used", n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=resolution, key_added="gnn_cluster")

    if plot:
        sc.pl.umap(adata, color="gnn_cluster")

    return adata


def label_gnn_clusters(adata: AnnData,
                       cluster_key: str = "gnn_cluster",
                       celltype_col: str = "celltype") -> None:
    markers = get_marker_genes()

    for cell_type, genes in markers.items():
        if not genes or not any(gene in adata.var_names for gene in genes):
            continue
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"{cell_type}_score")

    cluster_labels = {}
    score_cols = [f'{cell_type}_score' for cell_type in markers if f'{cell_type}_score' in adata.obs.columns]

    for cluster in sorted(adata.obs[cluster_key].unique(), key=int):
        sub = adata[adata.obs[cluster_key] == cluster]
        mean_scores = sub.obs[score_cols].mean()
        cluster_labels[cluster] = mean_scores.idxmax().replace("_score", "")

    adata.obs[celltype_col] = adata.obs[cluster_key].map(cluster_labels)

    sc.pl.umap(adata, color=celltype_col, legend_loc="on data")
