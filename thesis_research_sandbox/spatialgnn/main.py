import anndata as ad
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from data_processing.imaging.plot_spatial_clusters import plot_spatial_clusters
from spatialgnn.clustering import cluster_from_gnn, label_gnn_clusters
from spatialgnn.embeddings import add_gnn_embeddings
from spatialgnn.training import train_dgi


def prepare_node_features(adata: AnnData, n_pcs: int) -> np.ndarray:
    """
    Take AnnData, run PCA, return node features (cells × n_pcs).
    Stores results in adata so we can reuse.
    """
    adata_tmp = adata.copy()

    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)

    sc.pp.scale(adata_tmp, max_value=10)
    sc.tl.pca(adata_tmp, n_comps=n_pcs)

    adata.obsm["X_pca_gnn"] = adata_tmp.obsm["X_pca"]

    return adata.obsm["X_pca_gnn"]


def build_spatial_knn_graph(adata: AnnData, k: int = 8,
                            x_key: str = "CenterX_global_px",
                            y_key: str = "CenterY_global_px") -> torch.LongTensor:
    """
    Build a kNN graph based on spatial coordinates.
    Returns edge_index (2 × num_edges) as torch.LongTensor.
    """
    coords = np.vstack([adata.obs[x_key].values, adata.obs[y_key].values]).T  # (n_cells, 2)
    n_cells = coords.shape[0]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")  # +1 for self
    nn.fit(coords)
    dists, indices = nn.kneighbors(coords)  # indices: (n_cells, k+1)

    # Build edge list (i -> j)
    # skip index 0 because it's the cell itself
    edge_src = []
    edge_dst = []
    for i in range(n_cells):
        neighbors = indices[i, 1:]  # ignore self
        for j in neighbors:
            edge_src.append(i)
            edge_dst.append(j)

    edge_index = np.vstack([edge_src, edge_dst])  # shape (2, num_edges)

    # Make it undirected by adding reverse edges
    edge_index_rev = edge_index[::-1, :]
    edge_index = np.hstack([edge_index, edge_index_rev])

    # Optionally, remove duplicates
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)  # unique columns

    return edge_index


def adata_to_pyg_data(adata: AnnData,
                      feature_key: str = "X_pca_gnn",
                      x_key: str = "CenterX_global_px",
                      y_key: str = "CenterY_global_px",
                      k: int = 8) -> Data:
    """
    Convert AnnData to PyG Data object:
    - x = node features (PCA)
    - edge_index = spatial kNN graph
    - pos = spatial coordinates (for potential spatial encodings)
    """
    if feature_key not in adata.obsm:
        raise ValueError(f"{feature_key} not found in adata.obsm. Run prepare_node_features first.")

    x = torch.as_tensor(adata.obsm[feature_key], dtype=torch.float32)
    edge_index = build_spatial_knn_graph(adata, k=k, x_key=x_key, y_key=y_key)

    pos = torch.as_tensor(
        np.vstack([adata.obs[x_key].values,
                   adata.obs[y_key].values]).T,
        dtype=torch.float32
    )

    data = Data(x=x, edge_index=edge_index, pos=pos)
    return data


def run():
    adata = ad.read_h5ad('../resources/final_adata.h5ad')
    x_pca = prepare_node_features(adata, n_pcs=20)
    print(x_pca.shape)

    data = adata_to_pyg_data(adata)
    encoder = train_dgi(data, hidden_dim=128, epochs=200, device="cpu")

    add_gnn_embeddings(adata, data, encoder, key="X_gnn_dgi", device="cpu")

    mg = cluster_from_gnn(
        adata,
        gnn_key="X_gnn_dgi",
        resolution=0.6,
        plot=True
    )

    label_gnn_clusters(adata)
    plot_spatial_clusters(adata, 'gnn_cluster')


run()
