
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData

# https://alleninstitute.github.io/abc_atlas_access/notebooks/cluster_annotation_tutorial.html


def attach_mapmycells(adata: AnnData, mmc_df: pd.DataFrame, cell_id_col="cell_id"):
    """
    Join MapMyCells output (mmc_df) into adata.obs by cell_id.
    Assumes adata.obs_names match mmc_df[cell_id_col].
    """
    mmc_df = mmc_df.copy()
    mmc_df = mmc_df.set_index(cell_id_col)
    left_key = "cell_id_unique"  # <-- need to regernate scvi adata correctly!!!
    adata.obs = adata.obs.join(mmc_df, on=left_key, how="left")

    return adata


def add_mmc_fallback_label(
    adata: AnnData,
    thr=0.7,
    out_col="mmc_label_fallback",
    prefer=("cluster", "supertype", "subclass", "class"),
):
    """
    Create a single per-cell label by trying fine→coarse levels, using bootstrapping_probability as confidence.
    If a level's probability < thr, we fall back to the next coarser level.
    """
    obs = adata.obs

    def level_name(level):  # e.g., "cluster" -> "cluster_name"
        return f"{level}_name"

    def level_prob(level):  # e.g., "cluster" -> "cluster_bootstrapping_probability"
        return f"{level}_bootstrapping_probability"

    label = pd.Series(index=obs.index, dtype="object")

    for lvl in prefer:
        name_col = level_name(lvl)
        if lvl == "class":
            # class is the coarsest; accept even without a probability gate
            candidate = obs[name_col]
        else:
            prob_col = level_prob(lvl)
            candidate = obs[name_col].where(obs[prob_col] >= thr)

        label = label.fillna(candidate)

    adata.obs[out_col] = label.astype("category")
    return adata


def summarize_labels_by_cluster(
    adata: AnnData,
    cluster_key="leiden_scvi",
    label_col="mmc_label_fallback",
    prob_col=None,
):
    """
    For each cluster:
      - top label (majority)
      - fraction of cells with that label (purity)
      - counts (n/N)
      - optional median probability for cells supporting that top label (if prob_col provided)
    """
    df = adata.obs[[cluster_key, label_col] + ([] if prob_col is None else [prob_col])].dropna(subset=[cluster_key, label_col])

    # counts per (cluster, label)
    counts = df.groupby([cluster_key, label_col]).size().rename("n").reset_index()

    # total per cluster
    totals = counts.groupby(cluster_key)["n"].sum().rename("N")
    counts = counts.join(totals, on=cluster_key)
    counts["fraction"] = counts["n"] / counts["N"]

    # pick top label per cluster
    top = (
        counts.sort_values([cluster_key, "fraction", "n"], ascending=[True, False, False])
              .groupby(cluster_key)
              .head(1)
              .set_index(cluster_key)
              .rename(columns={label_col: "top_label"})
    )

    out = top[["top_label", "fraction", "n", "N"]].copy()

    # optional: median prob for the top label’s cells
    if prob_col is not None:
        def median_prob(g):
            tl = out.loc[g.name, "top_label"]
            return g.loc[g[label_col] == tl, prob_col].median()

        out["median_prob"] = df.groupby(cluster_key).apply(median_prob)

    return out.sort_index()


# ---------- 4) Add a cluster-level label back onto each cell ----------
def add_cluster_level_label(
    adata,
    cluster_key="leiden_scvi",
    per_cell_label_col="mmc_label_fallback",
    out_col="leiden_scvi_mmc_label",
):
    """
    Label each Leiden cluster using the most common per-cell label in that cluster,
    then map it back to every cell.
    """
    mapping = (
        adata.obs.groupby(cluster_key)[per_cell_label_col]
        .agg(lambda x: x.value_counts().index[0])
    )
    adata.obs[out_col] = adata.obs[cluster_key].map(mapping).astype("category")
    return adata, mapping




mmc_df = pd.read_csv('D:\\thesis-research\\resources\\map_my_cells\\cell_type_mapping.csv', skiprows=4)

# Now you should have proper columns
print(mmc_df.head())
print(mmc_df.columns)

adata = ad.read_h5ad("D:\\thesis-research\\resources\\clustering\\adata_full_with_pca_clustered.h5ad")

adata = attach_mapmycells(adata, mmc_df)

# 2) make a clean per-cell label with hierarchical fallback (tune thr as you like)
adata = add_mmc_fallback_label(adata, thr=0.7, out_col="mmc_label_fallback")

# 3) summarize how cleanly your Leiden clusters map to MMC labels
summary = summarize_labels_by_cluster(
    adata,
    cluster_key="cluster",
    label_col="subclass_name",
    # optional: if you want a confidence summary, pick ONE prob col to summarize, e.g. subclass prob:
    prob_col="subclass_bootstrapping_probability",
)
print(summary)

# 4) assign one label per leiden_scvi cluster and add it back to adata.obs
adata, cluster_to_label = add_cluster_level_label(
    adata,
    cluster_key="leiden_scvi",
    per_cell_label_col="mmc_label_fallback",
    out_col="leiden_scvi_mmc_label",
)

# 5) plot
sc.pl.umap(
    adata,
    color=["leiden_scvi", "mmc_label_fallback", "leiden_scvi_mmc_label"],
    wspace=0.4
)