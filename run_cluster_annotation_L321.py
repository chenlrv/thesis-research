"""Data-driven annotation of L321 Leiden clusters.

Workflow:
  1. Load X + var_names + cluster/coord obs columns straight from the .h5ad via
     h5py (anndata.read_h5ad chokes on the 'cell_global_id' obs column).
  2. Normalize + log1p if X looks like raw counts; keep raw in layers["counts"].
  3. rank_genes_groups (Wilcoxon) -> per-cluster marker genes (dotplot + CSV).
  4. score_genes for canonical cell-type signatures -> cluster x signature
     heatmap + CSV, plus an argmax label suggestion per cluster.

Nothing is written back to the cache; outputs are PNG/CSV next to the project root.
"""
import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix

CACHE = "D:/thesis-research/resources/cache/sample_L321_adata.h5ad"
CLUSTER_COL = (
    "Basic.run_Neighbor.network.expression.space.1_1"
    "_cluster_Basic.run_Leiden.Clustering.1_1"
)
X_KEY = "CenterX_global_px"
Y_KEY = "CenterY_global_px"

OUT_SPATIAL_DIR = "D:/thesis-research/L321_cluster_spatial"
OUT_DOTPLOT = "D:/thesis-research/L321_marker_dotplot.png"
OUT_MARKERS_CSV = "D:/thesis-research/L321_cluster_markers.csv"
OUT_HEATMAP = "D:/thesis-research/L321_signature_heatmap.png"
OUT_SCORES_CSV = "D:/thesis-research/L321_cluster_signature_scores.csv"
OUT_LABELS_CSV = "D:/thesis-research/L321_cluster_label_suggestions.csv"

N_TOP_MARKERS = 12

# Confidence gating for cluster -> label assignment. A cluster is labeled
# "unknown" unless BOTH conditions hold:
#   - top signature's z-score across clusters >= MIN_Z
#   - margin over runner-up signature >= MIN_MARGIN
MIN_Z = 1.0
MIN_MARGIN = 0.5
MIN_RAW_SCORE = 0.1
# Canonical signatures for the mouse brain-tumor microenvironment.
# Matched case-insensitively against the panel; missing genes are dropped.
SIGNATURES = {
    # Microglia — see caveat below. TMEM119 + Cx3cr1 are the only
    # microglia-restricted genes; C1q complement is microglia-dominant in CNS
    # but shared with border macrophages.
    "microglia":  ["TMEM119", "Cx3cr1", "C1qa", "C1qb", "C1qc"],

    # Monocyte-derived / general macrophage (recruited, CCR2+ lineage).
    # Cd68/Itgam dropped — pan-myeloid, would inflate the score on microglia clusters too.
    "macrophage_MDM": ["Ccr2", "Cd14", "Plac8", "Fcgr4"],

    # Perivascular / border-associated macrophage (tissue-resident, CD206+)
    "macrophage_BAM": ["Mrc1", "Cd163", "Lyve1", "Pf4", "Marco", "Msr1", "Cd5l"],

    # Astrocyte — GFAP is reactive-biased; Sox9/S100b/Glul/Sparcl1 add identity
    "astrocyte":  ["GFAP", "Sox9", "S100b", "Glul", "Sparcl1", "Glud1"],

    # Endothelial — competing signature so non-target clusters have somewhere to go
    # other than being force-collapsed onto microglia/macrophage/astrocyte.
    "endothelial": ["Pecam1", "Cdh5", "Flt1", "Kdr", "Esam", "Tie1", "Vwf"],
}


# ---------------------------------------------------------------------------
# Robust h5py loader
# ---------------------------------------------------------------------------
def _decode(arr):
    if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("O", "S"):
        return np.array([a.decode() if isinstance(a, bytes) else a for a in arr])
    return arr


def _read_obs_column(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        categories = _decode(node["categories"][...])
        return np.where(codes >= 0, categories[np.clip(codes, 0, None)], None)
    return _decode(node[...])


def _read_X(h5):
    node = h5["X"]
    if isinstance(node, h5py.Group):
        enc = node.attrs.get("encoding-type", "")
        shape = tuple(node.attrs["shape"])
        data = node["data"][...]
        indices = node["indices"][...]
        indptr = node["indptr"][...]
        if "csc" in str(enc):
            return csc_matrix((data, indices, indptr), shape=shape).tocsr()
        return csr_matrix((data, indices, indptr), shape=shape)
    return node[...]


def _read_var_names(h5):
    var = h5["var"]
    index_key = var.attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode()
    return _decode(var[index_key][...])


def load_minimal_adata(path):
    print(f"Loading {path} via h5py ...")
    with h5py.File(path, "r") as h5:
        if CLUSTER_COL not in h5["obs"]:
            raise KeyError(f"{CLUSTER_COL!r} not in /obs")
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        clusters_raw = _read_obs_column(h5, CLUSTER_COL)
        x = _read_obs_column(h5, X_KEY).astype(float)
        y = _read_obs_column(h5, Y_KEY).astype(float)

    # Drop cells with missing cluster assignment (categorical code == -1 -> None).
    valid = pd.notna(clusters_raw)
    n_missing = int((~valid).sum())
    if n_missing:
        print(f"  Dropping {n_missing} cells with missing cluster assignment")
        X = X[valid]
        clusters_raw = clusters_raw[valid]
        x = x[valid]
        y = y[valid]

    clusters = clusters_raw.astype(str)
    obs = pd.DataFrame({CLUSTER_COL: clusters, X_KEY: x, Y_KEY: y})
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = pd.Index(var_names, name="gene")
    adata.var_names_make_unique()
    print(f"  shape = {adata.shape}")
    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def looks_like_raw_counts(X):
    sample = X[:1000].data if hasattr(X, "data") else np.asarray(X[:1000]).ravel()
    if sample.size == 0:
        return False
    return np.allclose(sample, np.round(sample)) and sample.max() > 30


def resolve_genes(signatures, var_names):
    """Case-insensitive match of signature genes to actual panel var_names."""
    lut = {v.lower(): v for v in var_names}
    resolved, dropped = {}, {}
    for name, genes in signatures.items():
        present = [lut[g.lower()] for g in genes if g.lower() in lut]
        missing = [g for g in genes if g.lower() not in lut]
        resolved[name] = present
        if missing:
            dropped[name] = missing
    return resolved, dropped


def sort_clusters(labels):
    def key(l):
        try:
            return (0, int(l))
        except ValueError:
            return (1, l)
    return sorted(labels, key=key)


def plot_cluster_spatial(adata, out_dir):
    """One spatial plot per cluster: all cells in grey, this cluster highlighted.

    Y is flipped so the image matches the slide orientation (pixel coords grow
    downward). Each cluster gets its own PNG under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    x = adata.obs[X_KEY].to_numpy()
    y = adata.obs[Y_KEY].to_numpy()
    clusters = adata.obs[CLUSTER_COL].to_numpy()
    cats = list(adata.obs[CLUSTER_COL].cat.categories)
    print(f"Generating {len(cats)} per-cluster spatial plots -> {out_dir}")

    for cat in cats:
        mask = clusters == cat
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        ax.scatter(x[~mask], -y[~mask], s=1, c="lightgrey", linewidths=0, rasterized=True)
        ax.scatter(x[mask], -y[mask], s=3, c="crimson", linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"L321 — cluster {cat}  (n={int(mask.sum())})")
        out = os.path.join(out_dir, f"cluster_{cat}_spatial.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(cats)} plots")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    adata = load_minimal_adata(CACHE)
    adata.obs[CLUSTER_COL] = pd.Categorical(
        adata.obs[CLUSTER_COL],
        categories=sort_clusters(adata.obs[CLUSTER_COL].unique()),
        ordered=True,
    )

    # Per-cluster spatial maps before any annotation work.
    plot_cluster_spatial(adata, OUT_SPATIAL_DIR)

    # Drop negative/control probes so they can't masquerade as markers.
    keep = ~adata.var_names.str.lower().str.startswith(("neg", "negprb", "blank", "falsecode", "systemcontrol"))
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"Dropping {n_drop} control/negative probes from DE/scoring.")
        adata = adata[:, keep].copy()

    # Normalize + log1p only if X is raw counts; preserve raw in a layer.
    if looks_like_raw_counts(adata.X):
        print("X looks like raw counts -> normalize_total + log1p")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("X already looks normalized/log -> using as-is")

    # ---- 1) rank_genes_groups (Wilcoxon) ----
    print("Ranking marker genes per cluster (Wilcoxon) ...")
    sc.tl.rank_genes_groups(adata, groupby=CLUSTER_COL, method="wilcoxon")

    # Tidy marker table: names + logFC + pvals_adj + pct expressing in/out.
    rgg = adata.uns["rank_genes_groups"]
    groups = rgg["names"].dtype.names
    rows = []
    for grp in groups:
        for rank in range(N_TOP_MARKERS):
            rows.append({
                "cluster": grp,
                "rank": rank + 1,
                "gene": rgg["names"][grp][rank],
                "log2FC": rgg["logfoldchanges"][grp][rank],
                "pval_adj": rgg["pvals_adj"][grp][rank],
                "score": rgg["scores"][grp][rank],
            })
    markers_df = pd.DataFrame(rows)
    markers_df.to_csv(OUT_MARKERS_CSV, index=False)
    print(f"Saved markers -> {OUT_MARKERS_CSV}")

    top_genes = {grp: list(rgg["names"][grp][:N_TOP_MARKERS]) for grp in groups}
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=8, show=False)
    plt.savefig(OUT_DOTPLOT, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved dotplot -> {OUT_DOTPLOT}")

    # ---- 2) score_genes per signature ----
    resolved, dropped = resolve_genes(SIGNATURES, adata.var_names)
    for name, missing in dropped.items():
        print(f"  [{name}] not on panel: {missing}")
    score_cols = []
    for name, genes in resolved.items():
        if len(genes) == 0:
            print(f"  [{name}] SKIPPED — no genes on panel")
            continue
        col = f"score_{name}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=col)
        score_cols.append(col)
        print(f"  scored {name}: {genes}")

    # Mean score per cluster -> cluster x signature matrix.
    score_mat = adata.obs.groupby(CLUSTER_COL, observed=True)[score_cols].mean()
    score_mat.columns = [c.replace("score_", "") for c in score_mat.columns]
    score_mat.to_csv(OUT_SCORES_CSV)
    print(f"Saved signature scores -> {OUT_SCORES_CSV}")

    # ---- 3) label suggestion = argmax signature per cluster (with confidence gate) ----
    # z-score across clusters so a globally high signature doesn't dominate every row.
    z = (score_mat - score_mat.mean(axis=0)) / (score_mat.std(axis=0) + 1e-9)

    top_sig = z.idxmax(axis=1)
    top_z = z.max(axis=1)
    top_raw_score = pd.Series(
    [score_mat.loc[cluster, sig] for cluster, sig in top_sig.items()],
    index=score_mat.index,
    )
    runner_up = z.apply(lambda r: r.nlargest(2).index[-1], axis=1)
    runner_up_z = z.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    margin = top_z - runner_up_z

    confident = (top_z >= MIN_Z) & (margin >= MIN_MARGIN) &  (top_raw_score >= MIN_RAW_SCORE)
    label = top_sig.where(confident, other="unknown")

    suggestion = pd.DataFrame({
        "label": label,
        "top_signature": top_sig,
        "raw_score" : top_raw_score, 
        "z_score": top_z,
        "runner_up": runner_up,
        "runner_up_z": runner_up_z,
        "margin": margin,
        "confident": confident,
        "top_markers": [", ".join(top_genes[str(c)]) for c in score_mat.index],
    })
    suggestion.to_csv(OUT_LABELS_CSV)
    n_unknown = int((~confident).sum())
    print(f"Saved label suggestions -> {OUT_LABELS_CSV}")
    print(f"Confidence gate: MIN_Z={MIN_Z}, MIN_MARGIN={MIN_MARGIN} "
          f"-> {n_unknown}/{len(suggestion)} clusters labeled 'unknown'")
    print("\n=== Cluster label suggestions ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 180):
        print(suggestion)

    # Heatmap of z-scored signatures.
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(score_mat.columns) + 4),
                                    max(6, 0.4 * len(score_mat) + 2)), dpi=200)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels([f"cluster {c}" for c in z.index])
    ax.set_title("L321 — z-scored signature scores per cluster")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    plt.tight_layout()
    plt.savefig(OUT_HEATMAP, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap -> {OUT_HEATMAP}")


if __name__ == "__main__":
    main()
