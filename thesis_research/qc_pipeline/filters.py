import numpy as np
import pandas as pd

# import scrublet as scr
from anndata import AnnData



def get_negative_system_probes(adata: AnnData) -> AnnData:
    """Extract negative and system control probes from the dataset."""

    print(f"Checking for negative and system control probes...")

    negative_mask = adata.var_names.str.startswith("Negative")
    system_control_mask = adata.var_names.str.startswith("SystemControl")

    combined_mask = negative_mask | system_control_mask

    probes = adata[:, combined_mask].copy()

    print(
        f"Found {probes.n_vars} negative/system-control probes: {list(probes.var_names)}"
    )

    return probes


def remove_negative_system_probes(adata: AnnData, probes: AnnData) -> AnnData:
    """Remove negative and system control probes from the dataset."""
    print(f"Removing negative and system probes...")

    if probes.n_vars > 0:
        probe_names = probes.var_names
        adata = adata[:, ~adata.var_names.isin(probe_names)].copy()
        print(
            f"Removed {len(probe_names)} probes.\n"
            f" Final var count: {adata.n_vars}"
        )

    return adata


def remove_outliers_and_doublets(adata: AnnData) -> AnnData:
    print(f"Starting QC for AnnData {get_adata_section_id(adata)}")
    adata = remove_outlier_cells_by_genes_and_transcripts(adata)
    adata = run_scrublet(adata)
    print(f"Finished QC for AnnData {get_adata_section_id(adata)}")
    return adata


def remove_outlier_cells_by_genes_and_transcripts(
    adata: AnnData,
    genes_low=5,
    genes_high=95,
    counts_low=1,
    counts_high=90,
    min_genes_floor=200,
    min_counts_floor=500,
) -> AnnData:
    """
    Filter cells based on number of genes and transcript counts.
    """
    print(f"Starting outlier removal for Anndata {get_adata_section_id(adata)}")

    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    genes_min, genes_max = choose_bounds(adata.obs["n_genes"], genes_low, genes_high)
    counts_min, counts_max = choose_bounds(adata.obs["n_counts"], counts_low, counts_high)

    plot_genes_transcripts(adata, counts_max, counts_min, genes_max, genes_min)

    filter_condition = (
        (adata.obs["n_genes"] >= genes_min)
        & (adata.obs["n_genes"] <= genes_max)
        & (adata.obs["n_counts"] >= counts_min)
        & (adata.obs["n_counts"] <= counts_max)
    )
    adata_filtered = adata[filter_condition].copy()
    print(
        f"AnnData {get_adata_section_id(adata)}: removed {adata.n_obs - adata_filtered.n_obs} cells, {adata_filtered.n_obs} cells retained.\n "
        f"(gene count range: {genes_min:.0f}–{genes_max:.0f},"
        f"UMI count range: {counts_min:.0f}–{counts_max:.0f})"
    )

    return adata_filtered


def choose_bounds(x, low=1.0, high=99.5, floor=None, ceil=None):
    low, high = np.percentile(x, [low, high])
    if floor is not None:
        low = max(low, floor)
    if ceil is not None:
        high = min(high, ceil)
    return float(low), float(high)


def run_scrublet(adata: AnnData, expected_doublet_rate=0.02, manual_threshold=None) -> AnnData:
    """Run Scrublet to identify doublets in the AnnData object."""
    print(f"Running Scrublet for Anndata {get_adata_section_id(adata)}")

    counts_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()

    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets

    plot_doublet_analysis(adata, doublet_scores)

    if manual_threshold is not None:
        adata.obs["predicted_doublet"] = adata.obs["doublet_score"] > manual_threshold
        print(f"Manual threshold set at {manual_threshold:.3f}")

    n_before = adata.n_obs
    adata = adata[~adata.obs["predicted_doublet"]].copy()
    n_after = adata.n_obs

    print(
        f"Removed from AnnData {get_adata_section_id(adata)}: {n_before - n_after} doublets ({(n_before - n_after) / n_before:.2%}"
    )
    return adata


def _get_genes_and_probes_to_remove(correlation_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    neg_probes = correlation_df[correlation_df["gene"].str.startswith("NegPrb")].copy()
    neg_probes = neg_probes.sort_values("R2", ascending=False)

    threshold_r2 = neg_probes.iloc[0]["R2"]
    threshold_gene = neg_probes.iloc[0]["gene"]

    print(f"Using R² threshold from 1st highest negative probe:")
    print(f"Gene: {threshold_gene}")
    print(f"R²: {threshold_r2:.4f}")

    genes_to_remove = correlation_df[
        (correlation_df["R2"] >= threshold_r2)
        & (~correlation_df["gene"].str.startswith("NegPrb"))  # Not a negative probe
    ]["gene"].tolist()

    return genes_to_remove, neg_probes["gene"].tolist()
