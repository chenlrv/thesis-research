import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, to_undirected, coalesce
import scanpy as sc
import pandas as pd
import anndata as ad

from thesis_research.utils.columns import CELL_GLOBAL_ID

BASE_DIR = r"D:/thesis-research/"

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def prepare_gnn_labels(adata_slice_1, adata_slice_3, df_results_slice_1):
    # --- SLICE 1 LOGIC ---
    # 1. Identify High-Confidence Tumors using your specific criteria
    df_results_slice_1 = df_results_slice_1.copy()
    df_results_slice_1["next_best_score"] = df_results_slice_1[["score_brain_struct", "score_brain_immune"]].max(
        axis=1
    )
    df_results_slice_1["delta_score"] = abs(df_results_slice_1["score_tumor"] - df_results_slice_1["next_best_score"])
    df_results_slice_1["best_healthy_score"] = df_results_slice_1[
        ["score_brain_struct", "score_brain_immune"]
    ].max(axis=1)

    tumor_mask = (
        (df_results_slice_1["predicted_cell_type"] == "Tumor")
        & (df_results_slice_1["score_tumor"] > 0.5)
        & (df_results_slice_1["delta_score"] > 0.08)
    )
    high_conf_tumor_barcodes = set(df_results_slice_1.loc[tumor_mask, "cell_barcode"])

    # 2. Identify High-Confidence Healthy in Slice 1
    # A cell is "Healthy Anchor" only if SingleR is very sure it's NOT a tumor
    healthy_mask = (
            (df_results_slice_1["predicted_cell_type"] != "Tumor")
            & (df_results_slice_1["best_healthy_score"] > 0.5)
            & (df_results_slice_1["best_healthy_score"] - df_results_slice_1["score_tumor"] > 0.08)
    )
    high_conf_healthy_barcodes = set(df_results_slice_1.loc[healthy_mask, "cell_barcode"])

    # Apply to adata_slice_1
    adata_slice_1.obs["gnn_label"] = -1  # Default to ignore
    adata_slice_1.obs["gnn_train_mask"] = False  # Default to ignore

    # Set Tumor Anchors
    is_tumor = adata_slice_1.obs[CELL_GLOBAL_ID].isin(high_conf_tumor_barcodes)
    adata_slice_1.obs.loc[is_tumor, "gnn_label"] = 1
    adata_slice_1.obs.loc[is_tumor, "gnn_train_mask"] = True

    # Set Healthy Anchors (ONLY high confidence ones)
    is_healthy = adata_slice_1.obs[CELL_GLOBAL_ID].isin(high_conf_healthy_barcodes)
    adata_slice_1.obs.loc[is_healthy, "gnn_label"] = 0
    adata_slice_1.obs.loc[is_healthy, "gnn_train_mask"] = True

    df_results_slice_1["singler_label"] = (
            df_results_slice_1["predicted_cell_type"] == "Tumor"
    ).astype(int)

    df_results_slice_1["singler_conf"] = np.where(
        df_results_slice_1["predicted_cell_type"] == "Tumor",
        df_results_slice_1["score_tumor"],
        df_results_slice_1["best_healthy_score"]
    )

    barcode_to_singler_label = dict(
        zip(df_results_slice_1["cell_barcode"], df_results_slice_1["singler_label"])
    )

    barcode_to_singler_conf = dict(
        zip(df_results_slice_1["cell_barcode"], df_results_slice_1["singler_conf"])
    )


    adata_slice_1.obs["singler_label"] = (
        adata_slice_1.obs[CELL_GLOBAL_ID]
        .map(barcode_to_singler_label)
        .fillna(0)
        .astype(int)
    )

    adata_slice_1.obs["singler_conf"] = (
        adata_slice_1.obs[CELL_GLOBAL_ID]
        .map(barcode_to_singler_conf)
        .fillna(0.0)
        .astype(float)
    )

    # --- SLICE 3 LOGIC --- #TODO for now not including slice 3 in training
    # Everything in Slice B is a Healthy Anchor (Label 0)
    # adata_slice_3.obs["singler_conf"] = 0.0 #TODO change
    # adata_slice_3.obs["singler_label"] = 0
    # adata_slice_3.obs["gnn_label"] = 0
    # adata_slice_3.obs["gnn_train_mask"] = True


    adata_slice_1 = _run_pca(adata_slice_1)
    # adata_slice_3 = _run_pca(adata_slice_3)

    print(f"Mapped {adata_slice_1.obs['gnn_train_mask'].sum()} slice 1 anchors out of {len(adata_slice_1)} cells.")
    # print(f"Mapped {adata_slice_3.obs['gnn_train_mask'].sum()} slice 3 anchors out of {len(adata_slice_3)} cells.")

    return adata_slice_1, adata_slice_3


# -----------------------------
# Graph construction
# -----------------------------
def create_advanced_graph(
    adata_slice,
    is_slice_3: bool = False,
    spatial_k: int = 6,
    feature_k: int = 10,
    pca_key: str = "X_pca",
    x_key: str = "CenterX_global_px",
    y_key: str = "CenterY_global_px",
    label_col: str = "gnn_label",
    train_mask_col: str = "gnn_train_mask",
):
    if pca_key not in adata_slice.obsm:
        raise KeyError(f"{pca_key} not found in adata_slice.obsm")
    if x_key not in adata_slice.obs.columns or y_key not in adata_slice.obs.columns:
        raise KeyError(f"Missing spatial coordinate columns: {x_key}, {y_key}")
    if not is_slice_3 and label_col not in adata_slice.obs.columns:
        raise KeyError(f"{label_col} not found in adata_slice.obs")
    if train_mask_col not in adata_slice.obs.columns:
        raise KeyError(f"{train_mask_col} not found in adata_slice.obs")

    x_np = np.asarray(adata_slice.obsm[pca_key], dtype=np.float32)
    if not np.isfinite(x_np).all():
        raise ValueError(f"{pca_key} contains NaN or inf values")

    coords_np = adata_slice.obs[[x_key, y_key]].to_numpy(dtype=np.float32)

    x = torch.tensor(x_np, dtype=torch.float)

    def get_knn_edges(data: np.ndarray, k: int) -> torch.Tensor:
        knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        knn.fit(data)
        adj = knn.kneighbors_graph(data, mode="connectivity")
        coo = adj.tocoo()
        edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        return edge_index

    # Hybrid graph: spatial + feature neighbors
    edge_s = get_knn_edges(coords_np, k=spatial_k)
    edge_f = get_knn_edges(x_np, k=feature_k)

    edge_index = torch.cat([edge_s, edge_f], dim=1)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index = coalesce(edge_index)

    n = len(adata_slice)

    is_anchor = adata_slice.obs[train_mask_col].to_numpy().astype(bool)
    is_ambiguous = ~is_anchor

    train_mask = torch.tensor(is_anchor, dtype=torch.bool)
    ambiguous_mask = torch.tensor(is_ambiguous, dtype=torch.bool)


    y_np = adata_slice.obs[label_col].to_numpy()
    labeled_y = y_np[is_anchor]
    if not np.isin(labeled_y, [0, 1]).all():
        raise ValueError(f"{label_col} contains invalid values among anchor cells")

    y = torch.tensor(y_np, dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        ambiguous_mask=ambiguous_mask,
    )
    return data


# -----------------------------
# Model
# -----------------------------
class TumorGAT(torch.nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2, heads: int = 4
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.0
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# -----------------------------
# Training
# -----------------------------
def build_class_weights(data_a, device):
    all_train_y = data_a.y[data_a.train_mask]
    # all_train_y = torch.cat([data_a.y[data_a.train_mask], data_b.y[data_b.train_mask]])

    num_healthy = int((all_train_y == 0).sum())
    num_tumor = int((all_train_y == 1).sum())

    if num_healthy == 0 or num_tumor == 0:
        raise ValueError(
            f"Invalid class counts for weighted loss: healthy={num_healthy}, tumor={num_tumor}"
        )

    weights = torch.tensor([1.0 / num_healthy, 1.0 / num_tumor], dtype=torch.float, device=device)
    weights = weights / weights.sum()
    return weights


def train_model(model, data_a, optimizer, criterion, device, epochs=300, print_every=20):
    data_a = data_a.to(device)
    # data_b = data_b.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out_a = model(data_a.x, data_a.edge_index)
        # out_b = model(data_b.x, data_b.edge_index)

        loss_a = criterion(out_a[data_a.train_mask], data_a.y[data_a.train_mask])
        # loss_b = criterion(out_b[data_b.train_mask], data_b.y[data_b.train_mask])
        loss = loss_a

        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            train_acc_a = _evaluate_mask_accuracy(model, data_a, data_a.train_mask)
            # train_acc_b = _evaluate_mask_accuracy(model, data_b, data_b.train_mask)
            # correction_b = _validate_ambiguous_slice_3(model, data_b)

            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Train Acc 1: {train_acc_a:.4f} | "
                # f"Train Acc 3: {train_acc_b:.4f} | "
                # f"Slice 3 ambiguous correction: {correction_b:.4f}"
            )

    return model, data_a


def get_predictions(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)
        preds = probs.argmax(dim=1)
    return preds.cpu().numpy(), probs[:, 1].cpu().numpy()


# -----------------------------
# Metrics
# -----------------------------
def _run_pca(adata_slice):
    adata_copy = adata_slice.copy()
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)
    # sc.pp.highly_variable_genes(adata_slice, n_top_genes=2000, subset=True)
    sc.pp.scale(adata_copy, max_value=10)
    sc.tl.pca(adata_copy, n_comps=50)

    return adata_copy


def _evaluate_mask_accuracy(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        total = int(mask.sum())
        if total == 0:
            return np.nan
        correct = int((pred[mask] == data.y[mask]).sum())
        return correct / total


def _validate_ambiguous_slice_3(model, data_3):
    """
    Since slice 3 is assumed healthy, this checks how many ambiguous slice-B cells
    are predicted as class 0 (non-tumor).
    """
    model.eval()
    with torch.no_grad():
        out = model(data_3.x, data_3.edge_index)
        preds = out.argmax(dim=1)
        total = int(data_3.ambiguous_mask.sum())
        if total == 0:
            return np.nan
        correct = int((preds[data_3.ambiguous_mask] == 0).sum())
        return correct / total


def run_flow():
    # 1. Load Data
    # Use the same variable names throughout to avoid confusion
    adata_a = ad.read_h5ad(r"D:\thesis-research\resources\cache\slice_1_adata.h5ad")
    adata_b = ad.read_h5ad(r"D:\thesis-research\resources\cache\slice_3_adata.h5ad")

    df_results_a = pd.read_csv(
        rf"{BASE_DIR}/outputs/cell_annotation/L321/05/slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )

    # 2. Preprocess & Label (The Tri-State Logic)
    # Adds 'gnn_label' and 'gnn_train_mask' to .obs
    adata_a, adata_b = prepare_gnn_labels(adata_a, adata_b, df_results_a)

    # 3. Graph Construction
    data_a = create_advanced_graph(adata_a)
    # data_b = create_advanced_graph(adata_b)

    # 4. Setup Hardware & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_a.num_features automatically detects your PCA dim
    model = TumorGAT(in_channels=data_a.num_features, hidden_channels=32, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # 5. Handle Imbalance
    class_weights = build_class_weights(data_a, device=device)
    criterion = torch.nn.NLLLoss(weight=class_weights)

    # 6. Train
    # Note: train_model moves data to device internally based on your code
    model, data_a_dev = train_model(
        model=model,
        data_a=data_a,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=300,
        print_every=20,
    )

    # 7. Inference (The "Magic" Step)
    preds_a, probs_a = get_predictions(model, data_a_dev)
    # preds_b, probs_b = get_predictions(model, data_b_dev)

    # 8. Map back to AnnData
    # IMPORTANT: Since we used numpy arrays in get_predictions,
    # we assume the order is preserved (which it is in PyG Data objects).
    adata_a.obs["GNN_prediction"] = preds_a
    adata_a.obs["GNN_tumor_prob"] = probs_a

    # adata_b.obs["GNN_prediction"] = preds_b
    # adata_b.obs["GNN_tumor_prob"] = probs_b

    print("Flow Complete. Results stored in adata_a.obs and adata_b.obs.")
    return adata_a, adata_b


run_flow()