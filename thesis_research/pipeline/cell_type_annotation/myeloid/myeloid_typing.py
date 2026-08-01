"""
Myeloid cell type annotation for mouse brain CosMx data.

Pipeline stages:
  1. Reporter gating   — GFP / tdTomato expression → MDM vs. brain-resident myeloid
  2. Marker scoring    — sc.tl.score_genes per cell type
  3. Vascular proximity— KD-tree distance to endothelial cells (spatial BAM prior)
  4. Anchor labelling  — high-confidence cells from stages 1-3 become GAT training labels
  5. Multi-class GAT   — propagates labels to ambiguous cells via spatial+feature graph
  6. scGPT (optional)  — transformer embeddings as extra node features (use_scgpt=True)

Entry point:
    adata = annotate_myeloid_cells(adata, use_scgpt=False, scgpt_model_dir=None)

Output columns added to adata.obs:
    myeloid_reporter_label   — "MDM" / "brain_resident_myeloid" / "non_myeloid"
    myeloid_gfp_hi           — bool
    myeloid_tdt_hi           — bool
    score_microglia_h        — marker gene score
    score_microglia_dam      — marker gene score
    score_mdm                — marker gene score
    score_bam                — marker gene score
    score_astrocyte          — marker gene score
    myeloid_vascular_prox    — float [0,1], proximity to endothelial cells
    myeloid_anchor_label     — int label used for GAT training (-1 = unlabeled)
    myeloid_cell_type        — final annotation string
    myeloid_cell_type_prob   — GAT confidence [0,1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.spatial import cKDTree
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX

# ─── Class labels ────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "MDM",                    # 0
    "BAM",                    # 1
    "Microglia_Homeostatic",  # 2
    "Microglia_DAM",          # 3
    "Astrocyte",              # 4
    "Other",                  # 5
]
N_CLASSES = len(CLASS_NAMES)
LABEL_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
UNLABELED = -1

CLASS_MDM          = 0
CLASS_BAM          = 1
CLASS_MICRO_H      = 2
CLASS_MICRO_DAM    = 3
CLASS_ASTROCYTE    = 4
CLASS_OTHER        = 5

# ─── Marker gene sets (panel-verified) ───────────────────────────────────────

MARKER_SETS: dict[str, list[str]] = {
    "microglia_h": [
        "Cx3cr1", "TMEM119", "Aif1", "C1qa", "C1qb", "C1qc", "Tyrobp", "Csf1r",
    ],
    "microglia_dam": [
        "Cst7", "Gpnmb", "Spp1", "Apoe", "Trem2", "Lgals3", "Cd9", "Cd63",
    ],
    "mdm": [
        "Ccr2", "S100a8", "S100a9", "Cd14", "Mmp12", "Mmp9", "Ccl2",
    ],
    "bam": [
        "Lyve1", "Mrc1", "Cd163", "Clec12a", "Clec7a", "Marco", "Itgax", "Mertk",
    ],
    "astrocyte": [
        "GFAP", "S100b", "Sox9", "Vim", "Sparcl1", "Glul",
    ],
    "endothelial": [
        "Pecam1", "Cdh5", "Kdr", "Tek", "Vwf", "Esam", "Eng",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Reporter gating
# ─────────────────────────────────────────────────────────────────────────────

def _gmm_threshold(expr: np.ndarray, n_components: int = 2) -> float:
    """Fit a 2-component GMM on all cells (zeros + expressing) and return the midpoint.

    Fitting on the full distribution captures the true bimodal structure:
    component 1 = zero/background cells, component 2 = genuine reporter+ cells.
    """
    if (expr > 0).sum() < 20:
        return float(np.percentile(expr, 90))
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(expr.reshape(-1, 1))
    means = sorted(gmm.means_.ravel())
    return float(np.mean(means))


def gate_by_reporters(
    adata: AnnData,
    gfp_gene: str = "GFP",
    tdt_gene: str = "tdTomato",
) -> AnnData:
    """
    Stage 1: gate myeloid identity from TRANSCRIPTOME markers (not lineage tags).

    The custom GFP probe sits at the noise floor (S/N ~1.7) and tdTomato is broad /
    non-specific, so reporter-tag anchoring mis-routes the downstream GAT (see
    agents/outputs/annotation/report.md). We anchor on panel transcriptome markers and
    keep GFP/tdTomato only as informational columns for post-hoc consistency:
        non_myeloid            — < 2 pan-myeloid markers detected
        MDM                    — pan-myeloid AND an MDM-specific marker (Ccr2/Plac8)
        brain_resident_myeloid — pan-myeloid without an MDM-specific marker

    Adds to adata.obs:
        myeloid_gfp_hi, myeloid_tdt_hi (informational), myeloid_reporter_label
    """
    panel = {g.lower(): g for g in adata.var_names}

    def _get_expr(gene: str) -> np.ndarray:
        key = panel.get(gene.lower())
        if key is None:
            raise KeyError(
                f"Gene '{gene}' not found in adata.var_names. "
                f"Check spelling / case."
            )
        x = adata[:, key].X
        return x.toarray().ravel() if issparse(x) else np.asarray(x).ravel()

    gfp_expr = _get_expr(gfp_gene)
    tdt_expr = _get_expr(tdt_gene)

    gfp_thresh = _gmm_threshold(gfp_expr)
    tdt_thresh = _gmm_threshold(tdt_expr)

    print(f"[Reporter gating] GFP threshold: {gfp_thresh:.3f} | tdTomato threshold: {tdt_thresh:.3f}")
    
    # Lineage tags kept as informational columns ONLY (GFP dead, tdTomato non-specific).
    adata.obs["myeloid_gfp_hi"] = gfp_expr > gfp_thresh
    adata.obs["myeloid_tdt_hi"] = tdt_expr > tdt_thresh

    # Transcriptome anchoring (replaces the old GFP/tdTomato gate).
    pan_myeloid_genes  = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
    mdm_specific_genes = ["Ccr2", "Plac8"]
    n_pan = np.zeros(adata.n_obs, dtype=int)
    for g in pan_myeloid_genes:
        if g.lower() in panel:
            n_pan += (_get_expr(g) > 0).astype(int)
    is_pan_myeloid = n_pan >= 2
    has_mdm_specific = np.zeros(adata.n_obs, dtype=bool)
    for g in mdm_specific_genes:
        if g.lower() in panel:
            has_mdm_specific |= _get_expr(g) > 0

    reporter_label = pd.array(["non_myeloid"] * adata.n_obs, dtype="object")
    reporter_label[is_pan_myeloid & ~has_mdm_specific] = "brain_resident_myeloid"
    reporter_label[is_pan_myeloid &  has_mdm_specific] = "MDM"
    adata.obs["myeloid_reporter_label"] = reporter_label

    counts = pd.Series(reporter_label).value_counts()
    print(f"[Transcriptome gating] {counts.to_dict()}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Marker gene scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_marker_genes(adata: AnnData) -> AnnData:
    """
    Stage 2: compute sc.tl.score_genes for each cell type marker set.

    Uses only genes present in adata.var_names; silently skips absent genes.
    Adds score_<name> columns to adata.obs.
    """
    panel_lower = {g.lower(): g for g in adata.var_names}

    for name, genes in MARKER_SETS.items():
        valid = [panel_lower[g.lower()] for g in genes if g.lower() in panel_lower]
        if len(valid) < 2:
            print(f"[Marker scoring] '{name}': only {len(valid)} genes on panel, skipping")
            adata.obs[f"score_{name}"] = 0.0
            continue

        sc.tl.score_genes(adata, valid, score_name=f"score_{name}", use_raw=False)
        print(f"[Marker scoring] '{name}': scored with {len(valid)}/{len(genes)} genes: {valid}")

    # Normalise each score to [0, 1] across cells for comparable thresholding
    score_cols = [f"score_{k}" for k in MARKER_SETS if k != "endothelial"]
    for col in score_cols:
        s = adata.obs[col].to_numpy(dtype=float)
        lo, hi = s.min(), s.max()
        adata.obs[col] = (s - lo) / (hi - lo + 1e-9)

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Vascular proximity
# ─────────────────────────────────────────────────────────────────────────────

def compute_vascular_proximity(
    adata: AnnData,
    endothelial_score_thresh: float = 0.4,
    x_key: str = CENTER_X_GLOBAL_PX,
    y_key: str = CENTER_Y_GLOBAL_PX,
) -> AnnData:
    """
    Stage 3: compute each cell's proximity to the endothelial cell niche.

    Endothelial cells are identified by score_endothelial > endothelial_score_thresh.
    Proximity = exp(-dist / median_dist), so values near 1 → close to vessels.

    Adds to adata.obs:
        myeloid_dist_to_endo, myeloid_vascular_prox
    """
    if "score_endothelial" not in adata.obs:
        raise RuntimeError("Run score_marker_genes() before compute_vascular_proximity().")

    endo_score = adata.obs["score_endothelial"].to_numpy(dtype=float)
    endo_score_norm = (endo_score - endo_score.min()) / (endo_score.max() - endo_score.min() + 1e-9)
    endo_mask = endo_score_norm > endothelial_score_thresh

    n_endo = endo_mask.sum()
    if n_endo < 10:
        print(f"[Vascular proximity] Only {n_endo} endothelial cells found — lowering threshold.")
        endo_mask = endo_score_norm > np.percentile(endo_score_norm, 85)
        n_endo = endo_mask.sum()

    print(f"[Vascular proximity] {n_endo} endothelial reference cells.")

    coords = adata.obs[[x_key, y_key]].to_numpy(dtype=float)
    endo_coords = coords[endo_mask]

    tree = cKDTree(endo_coords)
    dist, _ = tree.query(coords, k=1)

    median_dist = float(np.median(dist))
    proximity = np.exp(-dist / (median_dist + 1e-9))

    adata.obs["myeloid_dist_to_endo"]  = dist
    adata.obs["myeloid_vascular_prox"] = proximity
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Anchor label assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_anchor_labels(
    adata: AnnData,
    bam_score_thresh: float      = 0.4,
    micro_score_thresh: float    = 0.35,
    dam_score_thresh: float      = 0.35,
    astro_score_thresh: float    = 0.4,
    vascular_prox_thresh: float  = 0.5,
    anti_vascular_thresh: float  = 0.2,
) -> AnnData:
    """
    Stage 4: assign integer class labels to high-confidence anchor cells.

    Priority order (to resolve overlaps):
        1. MDM            — transcriptome: pan-myeloid + Ccr2/Plac8 (no lineage tags)
        2. BAM            — brain-resident myeloid + BAM score + vascular proximity
        3. Astrocyte      — non-myeloid + astrocyte score
        4. Microglia DAM  — brain-resident myeloid + DAM score + NOT vascular
        5. Microglia Hom. — brain-resident myeloid + homeostatic score + NOT vascular

    Cells that don't meet any threshold get label UNLABELED (-1).
    Adds to adata.obs: myeloid_anchor_label, myeloid_is_anchor.
    """
    for col in ["myeloid_reporter_label", "score_bam", "score_microglia_h",
                "score_microglia_dam", "score_astrocyte", "myeloid_vascular_prox"]:
        if col not in adata.obs.columns:
            raise RuntimeError(f"Column '{col}' missing — run earlier stages first.")

    reporter = adata.obs["myeloid_reporter_label"].to_numpy(dtype=str)
    is_mdm     = reporter == "MDM"
    is_myeloid = reporter == "brain_resident_myeloid"
    is_nonmyl  = reporter == "non_myeloid"

    s_bam   = adata.obs["score_bam"].to_numpy(dtype=float)
    s_mh    = adata.obs["score_microglia_h"].to_numpy(dtype=float)
    s_mdam  = adata.obs["score_microglia_dam"].to_numpy(dtype=float)
    s_ast   = adata.obs["score_astrocyte"].to_numpy(dtype=float)
    v_prox  = adata.obs["myeloid_vascular_prox"].to_numpy(dtype=float)

    labels = np.full(adata.n_obs, UNLABELED, dtype=np.int64)

    # Priority 5 → 1 (later assignments overwrite, so highest-priority goes last)
    micro_h_mask  = is_myeloid & (s_mh  >= micro_score_thresh) & (v_prox < anti_vascular_thresh) & (s_bam < bam_score_thresh)
    micro_dam_mask = is_myeloid & (s_mdam >= dam_score_thresh) & (v_prox < anti_vascular_thresh) & (s_bam < bam_score_thresh)
    astro_mask    = is_nonmyl  & (s_ast >= astro_score_thresh)
    bam_mask      = is_myeloid & (s_bam  >= bam_score_thresh)  & (v_prox >= vascular_prox_thresh)
    mdm_mask      = is_mdm

    labels[micro_h_mask]   = CLASS_MICRO_H
    labels[micro_dam_mask] = CLASS_MICRO_DAM
    labels[astro_mask]     = CLASS_ASTROCYTE
    labels[bam_mask]       = CLASS_BAM
    labels[mdm_mask]       = CLASS_MDM     # always overwrites — MDM transcriptome anchor

    is_anchor = labels != UNLABELED
    adata.obs["myeloid_anchor_label"] = labels
    adata.obs["myeloid_is_anchor"]    = is_anchor

    anchor_counts = pd.Series(labels[is_anchor]).map(LABEL_TO_CLASS).value_counts()
    print(f"[Anchor labels] {is_anchor.sum()} anchors ({(is_anchor.mean()*100):.1f}% of cells)")
    print(anchor_counts.to_string())
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 (optional): scGPT embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _patch_torchtext() -> bool:
    """
    Inject a minimal pure-Python mock for torchtext.vocab so that scGPT can be
    imported even when the compiled torchtext binary is incompatible with the
    current PyTorch version (common with dev/nightly PyTorch builds).

    The mock provides only what scGPT's GeneVocab class needs:
      - torchtext.vocab.Vocab  as a base class (plain object subclass)
    GeneVocab implements all vocab logic itself; it only inherits from Vocab for
    type-checking purposes.

    Returns True if mock was injected (or torchtext was already working).
    """
    import sys
    # If torchtext is already importable and working, do nothing
    try:
        import torchtext  # noqa: F401
        return True
    except Exception:
        pass

    if "torchtext" in sys.modules:
        # already cached (possibly a failed import) — remove stale entry
        for key in list(sys.modules.keys()):
            if key == "torchtext" or key.startswith("torchtext."):
                del sys.modules[key]

    from types import ModuleType

    class _VocabBase:
        """Minimal stand-in for torchtext.vocab.Vocab."""
        pass

    vocab_mod = ModuleType("torchtext.vocab")
    vocab_mod.Vocab = _VocabBase

    torchtext_mod = ModuleType("torchtext")
    torchtext_mod.vocab = vocab_mod
    torchtext_mod.__version__ = "0.0.0-mock"

    sys.modules["torchtext"]       = torchtext_mod
    sys.modules["torchtext.vocab"] = vocab_mod
    print("[scGPT] Injected torchtext mock (compiled torchtext binary unavailable)")
    return True


def get_scgpt_embeddings(
    adata: AnnData,
    model_dir: str,
    batch_size: int = 64,
    use_gpu: bool = True,
) -> np.ndarray | None:
    """
    Stage 6 (optional): compute scGPT cell embeddings.

    Requires: python setup_scgpt.py  (installs scgpt + downloads scGPT_mouse)
    Model:    resources/scgpt_mouse/  (best_model.pt + vocab.json + args.json)

    Input:    uses .layers["log1p"] if present (data before sc.pp.scale()),
              otherwise falls back to .X — scGPT expects log1p-normalised counts,
              NOT z-scored values, so always store the layer before scaling.

    Returns an (n_cells, emb_dim) float32 array, or None on failure.
    Note: ~958-gene panel vs. full transcriptome (~20k genes); missing genes are
    masked out by the model vocabulary, so embedding quality is slightly reduced
    but key myeloid markers (Cx3cr1, Trem2, C1q, etc.) are all on the panel.
    """
    import sys

    # Patch torchtext before attempting to import scgpt
    _patch_torchtext()

    try:
        # scgpt may be in sys.modules already from a previous (failed) import;
        # if so, we need to reload now that torchtext is patched.
        if "scgpt" in sys.modules and "scgpt.tasks" not in sys.modules:
            for key in list(sys.modules.keys()):
                if key == "scgpt" or key.startswith("scgpt."):
                    del sys.modules[key]
        from scgpt.tasks import embed_data as scgpt_embed
    except ImportError:
        print("[scGPT] scgpt not installed — run: python setup_scgpt.py")
        return None
    except Exception as e:
        print(f"[scGPT] Import failed even after torchtext patch: {e}")
        import traceback; traceback.print_exc()
        return None

    # Build input adata using log1p-normalised counts (not z-scored .X)
    if "log1p" in adata.layers:
        print("[scGPT] Using .layers['log1p'] (pre-scale expression)")
        import anndata as _ad
        adata_input = _ad.AnnData(
            X=adata.layers["log1p"],
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
    else:
        print("[scGPT] WARNING: .layers['log1p'] not found — using .X. "
              "If .X is z-scored, embeddings will be degraded.")
        adata_input = adata.copy()

    # scGPT requires a var column with gene symbols (not just var_names)
    adata_input.var["gene_name"] = adata_input.var_names.tolist()

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[scGPT] Computing embeddings on {device} from {model_dir} ...")
    print(f"[scGPT] Input: {adata_input.n_obs:,} cells x {adata_input.n_vars} genes")

    try:
        emb = scgpt_embed(
            adata_input,
            model_dir=model_dir,
            gene_col="gene_name",
            batch_size=batch_size,
            device=device,
            return_new_adata=False,
        )
        emb = np.array(emb, dtype=np.float32)
        print(f"[scGPT] Embeddings shape: {emb.shape}")
        return emb
    except Exception as e:
        print(f"[scGPT] Embedding failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Multi-class GAT
# ─────────────────────────────────────────────────────────────────────────────

class MyeloidGAT(torch.nn.Module):
    """
    Graph Attention Network for multi-class myeloid annotation.

    Same architecture as TumorGAT but defaults to N_CLASSES output channels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = N_CLASSES,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.0
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def _build_knn_edges(data: np.ndarray, k: int) -> torch.Tensor:
    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(data)
    adj = knn.kneighbors_graph(data, mode="connectivity").tocoo()
    return torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long)


def _build_graph(
    node_features: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    spatial_k: int = 6,
    feature_k: int = 10,
) -> Data:
    """Construct a hybrid spatial+feature kNN graph as a PyG Data object."""
    x = torch.tensor(node_features, dtype=torch.float)

    edge_s = _build_knn_edges(coords, k=spatial_k)
    edge_f = _build_knn_edges(node_features, k=feature_k)
    edge_index = torch.cat([edge_s, edge_f], dim=1)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index)
    edge_index = coalesce(edge_index)

    is_anchor   = labels != UNLABELED
    train_mask  = torch.tensor(is_anchor, dtype=torch.bool)
    y           = torch.tensor(labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)


def _build_class_weights(y_train: torch.Tensor, n_classes: int, device) -> torch.Tensor:
    counts = torch.bincount(y_train, minlength=n_classes).float().clamp(min=1)
    weights = 1.0 / counts
    return (weights / weights.sum() * n_classes).to(device)


def _prepare_extra_features(
    adata: AnnData,
    extra_obs_cols: list[str],
) -> np.ndarray | None:
    """
    Extract, pre-process, and standardise extra obs columns for use as GAT node features.

    Transformations applied per column:
      - *Area*  columns          : log1p before standardisation (right-skewed)
      - Circularity               : clipped to [0, 1.5] before standardisation
      - All columns               : StandardScaler (zero mean, unit variance), clipped to [-5, 5]
    """
    from sklearn.preprocessing import StandardScaler

    if not extra_obs_cols:
        return None

    missing = [c for c in extra_obs_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Extra feature columns not found in adata.obs: {missing}")

    mat = adata.obs[extra_obs_cols].to_numpy(dtype=np.float64).copy()

    for i, col in enumerate(extra_obs_cols):
        col_lower = col.lower()
        if "area" in col_lower:
            mat[:, i] = np.log1p(mat[:, i])
        if "circularity" in col_lower:
            mat[:, i] = np.clip(mat[:, i], 0.0, 1.5)

    scaler = StandardScaler()
    mat    = np.clip(scaler.fit_transform(mat), -5.0, 5.0)
    return mat.astype(np.float32)


def _build_node_features(
    adata: AnnData,
    scgpt_embeddings: np.ndarray | None,
    extra_obs_cols: list[str] | None = None,
    pca_key: str = "X_pca",
) -> np.ndarray:
    """Concatenate: PCA + marker scores + vascular proximity [+ extra obs features] [+ scGPT]."""
    parts = []

    if pca_key not in adata.obsm:
        raise KeyError(f"'{pca_key}' not in adata.obsm — run sc.tl.pca() first.")
    parts.append(adata.obsm[pca_key].astype(np.float32))

    score_cols = [f"score_{k}" for k in MARKER_SETS if k != "endothelial"]
    score_mat  = adata.obs[score_cols].to_numpy(dtype=np.float32)
    parts.append(score_mat)

    prox = adata.obs["myeloid_vascular_prox"].to_numpy(dtype=np.float32).reshape(-1, 1)
    parts.append(prox)

    feature_desc = "PCA + markers + proximity"

    if extra_obs_cols:
        extra_mat = _prepare_extra_features(adata, extra_obs_cols)
        if extra_mat is not None:
            parts.append(extra_mat)
            feature_desc += f" + extra({len(extra_obs_cols)})"

    if scgpt_embeddings is not None:
        parts.append(scgpt_embeddings.astype(np.float32))
        feature_desc += " + scGPT"

    print(f"[GAT] Node features: {feature_desc}")
    node_features = np.concatenate(parts, axis=1)
    print(f"[GAT] Node feature dim: {node_features.shape[1]}")
    return node_features


def _select_gat_subset(
    adata: AnnData,
    max_cells: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a boolean mask of cells to include in the GAT graph.

    Priority:
      1. All GFP+ cells (reporter-confirmed myeloid)
      2. All non-Astrocyte anchor cells (BAM, microglia, MDM)
      3. Top cells by max myeloid marker score (ambiguous candidates)
      4. Random sample of astrocyte anchors for class balance
    All remaining budget filled from high-scoring non-anchor cells.
    """
    n = adata.n_obs
    mask = np.zeros(n, dtype=bool)

    # Mandatory: all GFP+ cells
    gfp_hi = adata.obs["myeloid_gfp_hi"].to_numpy(dtype=bool)
    mask |= gfp_hi

    # Mandatory: all myeloid-type anchors (BAM, microglia, MDM — not astrocyte/Other)
    anchor_labels = adata.obs["myeloid_anchor_label"].to_numpy(dtype=np.int64)
    myeloid_anchors = np.isin(anchor_labels, [CLASS_MDM, CLASS_BAM, CLASS_MICRO_H, CLASS_MICRO_DAM])
    mask |= myeloid_anchors

    # Fill budget with top myeloid-scoring cells (catches GFP- myeloid spillover)
    myeloid_score_cols = ["score_microglia_h", "score_microglia_dam", "score_mdm", "score_bam"]
    myeloid_scores = np.stack(
        [adata.obs[c].to_numpy(dtype=float) for c in myeloid_score_cols], axis=1
    ).max(axis=1)
    myeloid_scores[mask] = -1.0  # already included
    budget = max(0, max_cells // 3 - mask.sum())
    if budget > 0:
        top_idx = np.argpartition(myeloid_scores, -min(budget, (myeloid_scores > 0).sum()))[-budget:]
        mask[top_idx] = True

    # Add a capped sample of astrocyte anchors for class balance in the GAT
    astro_anchors = np.where(anchor_labels == CLASS_ASTROCYTE)[0]
    astro_budget  = min(len(astro_anchors), max(500, max_cells // 6))
    if len(astro_anchors) > astro_budget:
        astro_anchors = rng.choice(astro_anchors, astro_budget, replace=False)
    mask[astro_anchors] = True

    # Fill any remaining budget with highest astrocyte-scoring non-anchor cells
    remaining = max_cells - mask.sum()
    if remaining > 0:
        astro_scores = adata.obs["score_astrocyte"].to_numpy(dtype=float).copy()
        astro_scores[mask] = -1.0
        top_ast = np.argpartition(astro_scores, -min(remaining, (astro_scores > 0).sum()))[-remaining:]
        mask[top_ast] = True

    print(f"[GAT] Subset: {mask.sum():,} / {n:,} cells selected for graph construction")
    return mask


def train_myeloid_gat(
    adata: AnnData,
    scgpt_embeddings: np.ndarray | None = None,
    extra_obs_cols: list[str] | None = None,
    spatial_k: int = 6,
    feature_k: int = 10,
    hidden_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.2,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    epochs: int = 400,
    print_every: int = 50,
    max_gat_cells: int = 20_000,
    x_key: str = CENTER_X_GLOBAL_PX,
    y_key: str = CENTER_Y_GLOBAL_PX,
    pca_key: str = "X_pca",
    seed: int = 42,
) -> tuple[AnnData, MyeloidGAT]:
    """
    Stage 5: build graph, train multi-class GAT, write predictions to adata.obs.

    For large datasets (>max_gat_cells), GAT runs on a myeloid-focused subset.
    Non-subset cells receive their anchor label if available, else "Other".

    Adds to adata.obs:
        myeloid_cell_type      — predicted class name string
        myeloid_cell_type_prob — max-class softmax probability
    """
    for col in ["myeloid_anchor_label", "myeloid_vascular_prox"]:
        if col not in adata.obs.columns:
            raise RuntimeError(f"'{col}' missing — run assign_anchor_labels() first.")

    rng = np.random.default_rng(seed)

    # Pre-initialise output arrays (fallback = anchor label or Other)
    anchor_labels = adata.obs["myeloid_anchor_label"].to_numpy(dtype=np.int64)
    final_preds = np.where(anchor_labels != UNLABELED, anchor_labels, CLASS_OTHER)
    final_conf  = np.where(anchor_labels != UNLABELED, 0.7, 0.3).astype(np.float32)

    # Select subset for GAT
    if adata.n_obs > max_gat_cells:
        sub_mask = _select_gat_subset(adata, max_gat_cells, rng)
    else:
        sub_mask = np.ones(adata.n_obs, dtype=bool)
        print(f"[GAT] Using all {adata.n_obs:,} cells")

    sub_idx = np.where(sub_mask)[0]
    adata_sub = adata[sub_mask]

    scgpt_sub = scgpt_embeddings[sub_mask] if scgpt_embeddings is not None else None
    node_features = _build_node_features(adata_sub, scgpt_sub, extra_obs_cols=extra_obs_cols, pca_key=pca_key)
    coords        = adata_sub.obs[[x_key, y_key]].to_numpy(dtype=np.float32)
    labels        = adata_sub.obs["myeloid_anchor_label"].to_numpy(dtype=np.int64)

    data   = _build_graph(node_features, coords, labels, spatial_k=spatial_k, feature_k=feature_k)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAT] Training on {device} | {int(data.train_mask.sum())} anchor cells in subset")

    model = MyeloidGAT(
        in_channels=node_features.shape[1],
        hidden_channels=hidden_channels,
        out_channels=N_CLASSES,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = _build_class_weights(data.y[data.train_mask], N_CLASSES, device)
    criterion     = torch.nn.NLLLoss(weight=class_weights)

    data = data.to(device)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                acc  = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Anchor acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs  = torch.exp(logits)
        preds  = probs.argmax(dim=1).cpu().numpy()
        conf   = probs.max(dim=1).values.cpu().numpy()

    # Map GAT predictions back to full adata index
    final_preds[sub_idx] = preds
    final_conf[sub_idx]  = conf

    adata.obs["myeloid_cell_type"]      = pd.Categorical([LABEL_TO_CLASS[p] for p in final_preds])
    adata.obs["myeloid_cell_type_prob"] = final_conf

    print("\n[GAT] Final predictions:")
    print(adata.obs["myeloid_cell_type"].value_counts().to_string())
    return adata, model


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def annotate_myeloid_cells(
    adata: AnnData,
    # Reporter genes
    gfp_gene: str             = "GFP",
    tdt_gene: str             = "tdTomato",
    # Anchor thresholds
    bam_score_thresh: float   = 0.4,
    micro_score_thresh: float = 0.35,
    dam_score_thresh: float   = 0.35,
    astro_score_thresh: float = 0.4,
    vascular_prox_thresh: float = 0.5,
    # GAT hyperparameters
    gat_hidden: int           = 64,
    gat_heads: int            = 4,
    gat_epochs: int           = 400,
    max_gat_cells: int        = 20_000,
    # scGPT (optional)
    use_scgpt: bool           = False,
    scgpt_model_dir: str | None = None,
    # Extra obs features (e.g. protein channels, morphology)
    extra_obs_cols: list[str] | None = None,
    # Graph construction
    spatial_k: int            = 6,
    feature_k: int            = 10,
    pca_key: str              = "X_pca",
) -> AnnData:
    """
    Full myeloid annotation pipeline.

    Parameters
    ----------
    adata
        AnnData object with raw or normalised counts in .X, X_pca in .obsm,
        and spatial coordinates in .obs (CenterX/Y_global_px).
    use_scgpt
        If True, compute scGPT embeddings and add to GAT node features.
        Requires: pip install scgpt, and scgpt_model_dir pointing to the
        downloaded bowang-lab/scGPT_mouse checkpoint.
    scgpt_model_dir
        Path to the scGPT mouse model directory.

    Returns
    -------
    adata with myeloid_cell_type and other columns added to .obs.
    """
    adata = adata.copy()

    print("=" * 60)
    print("Stage 1: Reporter gating")
    print("=" * 60)
    adata = gate_by_reporters(adata, gfp_gene=gfp_gene, tdt_gene=tdt_gene)

    print("\n" + "=" * 60)
    print("Stage 2: Marker gene scoring")
    print("=" * 60)
    adata = score_marker_genes(adata)

    print("\n" + "=" * 60)
    print("Stage 3: Vascular proximity")
    print("=" * 60)
    adata = compute_vascular_proximity(adata)

    print("\n" + "=" * 60)
    print("Stage 4: Anchor label assignment")
    print("=" * 60)
    adata = assign_anchor_labels(
        adata,
        bam_score_thresh=bam_score_thresh,
        micro_score_thresh=micro_score_thresh,
        dam_score_thresh=dam_score_thresh,
        astro_score_thresh=astro_score_thresh,
        vascular_prox_thresh=vascular_prox_thresh,
    )

    scgpt_emb = None
    if use_scgpt:
        print("\n" + "=" * 60)
        print("Stage 6: scGPT embeddings (optional)")
        print("=" * 60)
        if scgpt_model_dir is None:
            print("[scGPT] scgpt_model_dir not provided — skipping.")
        else:
            scgpt_emb = get_scgpt_embeddings(adata, model_dir=scgpt_model_dir)

    print("\n" + "=" * 60)
    print("Stage 5: Multi-class GAT")
    print("=" * 60)
    adata, _ = train_myeloid_gat(
        adata,
        scgpt_embeddings=scgpt_emb,
        extra_obs_cols=extra_obs_cols,
        spatial_k=spatial_k,
        feature_k=feature_k,
        hidden_channels=gat_hidden,
        heads=gat_heads,
        epochs=gat_epochs,
        max_gat_cells=max_gat_cells,
        pca_key=pca_key,
    )

    print("\n" + "=" * 60)
    print("Done. Results in adata.obs['myeloid_cell_type']")
    print("=" * 60)
    return adata
