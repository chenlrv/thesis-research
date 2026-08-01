from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from typing import Tuple

import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


def build_spatial_graph(adata_slice, label_col, is_slice_b=False, k=8):
    # 1. Node Features: Use PCA for a dense, denoised representation
    x = torch.tensor(adata_slice.obsm["X_pca"], dtype=torch.float)

    # 2. Edges: Compute k-NN based on global spatial coordinates
    coords = adata_slice.obs[["CenterX_global_px", "CenterY_global_px"]].values
    knn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    distances, indices = knn.kneighbors(coords)

    # Create edge_index (exclude self-loops)
    edge_list = []
    for i in range(len(indices)):
        for j in indices[i][1:]:  # Skip the first one (it's the node itself)
            edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 3. Labels:
    # If it's Slice B, ALL cells are 0 (Healthy), regardless of SingleR
    if is_slice_b:
        y = torch.zeros(len(adata_slice), dtype=torch.long)
    else:
        # For Slice A, use your high-confidence labels (0=Healthy, 1=Tumor)
        y = torch.tensor(adata_slice.obs[label_col].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


# Construct your subgraphs
data_a = build_spatial_graph(slice_a, "gnn_label", is_slice_b=False)
data_b = build_spatial_graph(slice_b, "gnn_label", is_slice_b=True)


import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TumorGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(TumorGAT, self).__init__()
        # First layer: Multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        # Second layer: Final classification (concatenating heads)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Initialize: in_channels = number of PCs (e.g., 50), out_channels = 2
model = TumorGAT(in_channels=50, hidden_channels=32, out_channels=2)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    total_loss = 0
    # Train on both graphs in each epoch
    for data in [data_a, data_b]:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / 2


def prepare_expression_features(
    adata: AnnData,
    n_hvg: int = 2000,
    n_pcs: int = 50,
) -> Tuple[AnnData, np.ndarray]:
    """
    Prepare PCA-based node features from expression.

    Returns
    -------
    adata_proc : AnnData
        Processed AnnData object.
    X_pca : np.ndarray
        PCA feature matrix of shape (n_cells, n_pcs).
    """
    adata_proc = adata.copy()

    # standard preprocessing
    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    # sc.pp.highly_variable_genes(adata_proc, n_top_genes=n_hvg, subset=True)
    sc.pp.scale(adata_proc, max_value=10)
    sc.tl.pca(adata_proc, n_comps=n_pcs)

    X_pca = np.asarray(adata_proc.obsm["X_pca"], dtype=np.float32)
    return adata_proc, X_pca


def extract_spatial_coordinates(
    adata: AnnData,
    x_key: str = "CenterX_global_px",
    y_key: str = "CenterY_global_px",
) -> np.ndarray:
    """
    Extract 2D spatial coordinates from adata.obs.

    Returns
    -------
    coords : np.ndarray
        Array of shape (n_cells, 2)
    """
    if x_key not in adata.obs.columns or y_key not in adata.obs.columns:
        raise KeyError(f"Missing coordinate columns: {x_key}, {y_key}")

    coords = np.column_stack(
        [
            adata.obs[x_key].to_numpy(),
            adata.obs[y_key].to_numpy(),
        ]
    ).astype(np.float32)

    return coords


def build_pseudo_labels_from_cell_ids(
    adata: AnnData,
    tumor_cell_ids: list[str],
    healthy_cell_ids: list[str],
    cell_id_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create pseudo-labels and training mask from trusted cell IDs.
    """
    tumor_cell_ids = set(map(str, tumor_cell_ids))
    healthy_cell_ids = set(map(str, healthy_cell_ids))

    obs_ids = adata.obs[cell_id_col].astype(str).to_numpy()

    obs_id_set = set(obs_ids)
    missing_tumor = tumor_cell_ids - obs_id_set
    missing_healthy = healthy_cell_ids - obs_id_set

    tumor_mask = np.isin(obs_ids, list(tumor_cell_ids))
    healthy_mask = np.isin(obs_ids, list(healthy_cell_ids))

    y = np.full(adata.n_obs, -1, dtype=np.int64)
    train_mask = np.zeros(adata.n_obs, dtype=bool)

    y[tumor_mask] = 1
    y[healthy_mask] = 0
    train_mask[tumor_mask | healthy_mask] = True

    print(f"Labeled tumor cells: {tumor_mask.sum()}")
    print(f"Labeled healthy cells: {healthy_mask.sum()}")
    print(f"Total labeled cells: {train_mask.sum()}")

    return y, train_mask


if __name__ == "__main__":
    # Example usage:
    # adata = sc.read_h5ad("your_file.h5ad")
    # adata_proc, X = prepare_expression_features(adata, n_hvg=2000, n_pcs=50)
    # coords = extract_spatial_coordinates(adata_proc)
    # y, train_mask = build_pseudo_labels_from_logreg(
    #     adata_proc,
    #     prob_col="tumor_prob_logreg",
    #     tumor_threshold=0.9,
    #     non_tumor_threshold=0.1,
    # )
    # print(X.shape, coords.shape, y.shape, train_mask.sum())
    pass
