import numpy as np
import pandas as pd
import scrublet as scr
from anndata import AnnData
from scipy.stats import pearsonr

from data_processing.plots import plot_genes_transcripts, plot_doublet_analysis
from data_processing.utils.adata import get_adata_section_id


def get_negative_probes(adata: AnnData) -> AnnData:
    """Extract negative control probes from the dataset."""
    print(f'Checking for negative probes in adata {get_adata_section_id(adata)}')

    negative_mask = adata.var_names.str.startswith('NegPrb')
    negative_probes = adata[:, negative_mask].copy()

    print(f'Negative probes from adata {get_adata_section_id(adata)}: {negative_probes.var_names}')
    return negative_probes


def remove_negative_probes(adata: AnnData, negative_probes: AnnData) -> AnnData:
    """Remove negative control probes from the dataset."""

    print(f'Removing negative probes for Anndata {get_adata_section_id(adata)}')
    print(f'Initial var count {adata.n_vars}')
    if negative_probes.n_vars > 0:
        neg_names = negative_probes.var_names
        adata = adata[:, ~adata.var_names.isin(neg_names)].copy()
        print(f'Removed {len(neg_names)} negative probes from Anndata {get_adata_section_id(adata)}.\n'
              f' Final var count: {adata.n_vars}')

    return adata


def remove_outliers_and_doublets(adata: AnnData) -> AnnData:
    print(f'Starting QC for AnnData {get_adata_section_id(adata)}')
    adata = remove_outlier_cells_by_genes_and_transcripts(adata)
    adata = run_scrublet(adata)
    print(f'Finished QC for AnnData {get_adata_section_id(adata)}')
    return adata


def remove_outlier_cells_by_genes_and_transcripts(
        adata: AnnData,
        genes_low=5,
        genes_high=95,
        counts_low=1,
        counts_high=90,
        min_genes_floor=200,
        min_counts_floor=500) -> AnnData:
    """
    Filter cells based on number of genes and transcript counts.
    """
    print(f'Starting outlier removal for Anndata {get_adata_section_id(adata)}')

    adata.obs['n_counts'] = np.asarray(adata.X.sum(axis=1)).ravel()
    adata.obs['n_genes'] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    genes_min, genes_max = choose_bounds(adata.obs['n_genes'], genes_low, genes_high)
    counts_min, counts_max = choose_bounds(adata.obs['n_counts'], counts_low, counts_high)

    plot_genes_transcripts(adata, counts_max, counts_min, genes_max, genes_min)

    filter_condition = (
            (adata.obs['n_genes'] >= genes_min) & (adata.obs['n_genes'] <= genes_max) &
            (adata.obs['n_counts'] >= counts_min) & (adata.obs['n_counts'] <= counts_max)
    )
    adata_filtered = adata[filter_condition].copy()
    print(
        f'AnnData {get_adata_section_id(adata)}: removed {adata.n_obs - adata_filtered.n_obs} cells, {adata_filtered.n_obs} cells retained.\n '
        f'(gene count range: {genes_min:.0f}–{genes_max:.0f},'
        f'UMI count range: {counts_min:.0f}–{counts_max:.0f})'
    )

    return adata_filtered


def choose_bounds(x, low=1.0, high=99.5, floor=None, ceil=None):
    low, high = np.percentile(x, [low, high])
    if floor is not None: low = max(low, floor)
    if ceil is not None: high = min(high, ceil)
    return float(low), float(high)


def run_scrublet(adata: AnnData, expected_doublet_rate=0.02, manual_threshold=None) -> AnnData:
    """ Run Scrublet to identify doublets in the AnnData object."""
    print(f'Running Scrublet for Anndata {get_adata_section_id(adata)}')

    counts_matrix = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets()

    adata.obs['doublet_score'] = doublet_scores
    adata.obs['predicted_doublet'] = predicted_doublets

    plot_doublet_analysis(adata, doublet_scores)

    if manual_threshold is not None:
        adata.obs['predicted_doublet'] = adata.obs['doublet_score'] > manual_threshold
        print(f'Manual threshold set at {manual_threshold:.3f}')

    n_before = adata.n_obs
    adata = adata[~adata.obs['predicted_doublet']].copy()
    n_after = adata.n_obs

    print(
        f'Removed from AnnData {get_adata_section_id(adata)}: {n_before - n_after} doublets ({(n_before - n_after) / n_before:.2%}')
    return adata


def remove_plaque_correlated_genes(adata: AnnData) -> AnnData:
    """
    Find genes correlated with plaque overlap and removes them from the AnnData object.
    """
    print(f'Anndata has {adata.n_obs} before filtering plaque-overlapping cells')

    correlation_df = _calc_plaque_correlations(adata)
    genes_to_remove, neg_probes = _get_genes_and_probes_to_remove(correlation_df)

    print(f'Removing {len(genes_to_remove)} genes and {len(neg_probes)} neg probes from AnnData...')
    print(f'Before: {adata.n_obs} cells × {adata.n_vars} genes')

    genes_to_keep = [g for g in adata.var_names if g not in set(genes_to_remove) | set(neg_probes)]
    adata_clean = adata[:, genes_to_keep].copy()

    print(f'After:  {adata_clean.n_obs} cells × {adata_clean.n_vars} genes')
    print(f'Removed: {adata.n_vars - adata_clean.n_vars} genes')

    return adata_clean


def _calc_plaque_correlations(adata: AnnData) -> pd.DataFrame:
    print(f'Using {adata.n_obs} plaque-overlapping cells for correlation')

    plaque_col = 'Mean.BetaAmyloid'
    mask = adata.obs[plaque_col] > 0
    adata = adata[mask].copy()

    correlations = []
    mean_plaque_values = adata.obs[plaque_col].values

    for gene in adata.var_names:
        gene_counts = adata[:, gene].X.toarray().ravel()
        if np.std(gene_counts) > 0 and np.std(mean_plaque_values) > 0:
            r, _ = pearsonr(gene_counts, mean_plaque_values)
            correlations.append((gene, r ** 2))
        else:
            correlations.append((gene, 0))

    correlation_df = pd.DataFrame(correlations, columns=['gene', 'R2']).sort_values('R2', ascending=False)

    print('Top 10 correlated genes (including NegPrb):')
    print(correlation_df.head(10))

    return correlation_df


def _get_genes_and_probes_to_remove(correlation_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    neg_probes = correlation_df[correlation_df['gene'].str.startswith('NegPrb')].copy()
    neg_probes = neg_probes.sort_values('R2', ascending=False)

    threshold_r2 = neg_probes.iloc[0]['R2']
    threshold_gene = neg_probes.iloc[0]['gene']

    print(f'Using R² threshold from 1st highest negative probe:')
    print(f'Gene: {threshold_gene}')
    print(f'R²: {threshold_r2:.4f}')

    genes_to_remove = correlation_df[
        (correlation_df['R2'] >= threshold_r2) &
        (~correlation_df['gene'].str.startswith('NegPrb'))  # Not a negative probe
        ]['gene'].tolist()

    return genes_to_remove, neg_probes['gene'].tolist()
