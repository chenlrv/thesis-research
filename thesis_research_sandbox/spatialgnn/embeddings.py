import torch
from anndata import AnnData
from torch_geometric.data import Data
from torch import nn

def add_gnn_embeddings(adata: AnnData, data: Data, encoder: nn.Module,
                       key: str = "X_gnn_dgi",
                       device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    encoder.eval()
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    with torch.no_grad():
        z = encoder(x, edge_index)  # shape: [n_cells, hidden_dim]

    z = z.cpu().numpy()
    adata.obsm[key] = z
    print(f"Stored GNN embeddings in adata.obsm['{key}'] with shape {z.shape}")
