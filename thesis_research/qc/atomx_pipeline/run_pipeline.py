import logging

import anndata as ad
from anndata import AnnData

from thesis_research.qc.atomx_pipeline.arrangement_plots import generate_fov_arrangement_plot
from thesis_research.qc.atomx_pipeline.qc_plots import _generate_violin_plots, _generate_scatter_plots, \
    _generate_histograms

LOGGER = logging.getLogger(__name__)
DATA_PATH_TEMPLATE = 'D:/thesis-research/resources/{section_id}'


def initial_data(section_id: str, path: str) -> AnnData:
    if path:
        adata = ad.read_h5ad(path)
    else:
        adata = _load_adata(section_id)

    df = adata.obs[[
        "slide_ID",
        "fov",
        "nCount_RNA",
        "nFeature_RNA",
        "nCount_negprobes"
    ]].copy()
    df["fov"] = df["fov"].astype(int)
    fov_stats = (
        df.groupby(["slide_ID", "fov"])
        .agg(
            mean_transcripts_per_cell=("nCount_RNA", "mean"),
            mean_unique_genes_per_cell=("nFeature_RNA", "mean"),
            p10_transcripts_per_cell=("nCount_RNA", lambda x: x.quantile(0.10)),
            p90_transcripts_per_cell=("nCount_RNA", lambda x: x.quantile(0.90)),
            mean_negprobe_counts_per_cell=("nCount_negprobes", "mean"),
            n_cells=("nCount_RNA", "size"),
        )
        .reset_index()
    )
    fov_stats_sorted = fov_stats.sort_values(["slide_ID", "fov"]).reset_index(drop=True)
    fov_stats_sorted.head(10)

    return adata


def qc(adata: AnnData) -> None:
    adata.obs["Identity"] = "C"
    x = adata.obs["propNegative"].astype(float)
    adata.obs["percent.NegPrb"] = x * 100 if x.max() <= 1.0 else x

    keys = ["nFeature_RNA", "nCount_RNA", "percent.NegPrb", "Area"]

    _generate_histograms(adata, keys)
    _generate_violin_plots(adata, keys)
    _generate_scatter_plots(adata)


generate_fov_arrangement_plot('D:\\thesis-research\\resources\\L34\\L34_fov_positions_file.csv')
adata = initial_data('L34', path='D:\\thesis-research\\resources\\L34\\adata.h5ad')
qc(adata)

# At the cell level, look for transcripts per cell > ~200. If too many cells are flagged (30% or more), consider
# reducing this threshold. Transcripts per cell depends on many factors in study design and sample biology


# Normalization
# l Total Counts Normalization is generally recommended as it keeps the data on a linear scale, is easily
# interpretable, and is quick to run.
# l Other transformations (log1p, Pearson, sctransform) are possible and may improve visualizations in some
# datasets. However, the Pearson method is very resource- and time-intensive, so it is only recommended for
# smaller datasets (using lower plex panels than WTX).


# 4. PCA
# l Calculate 50 principal components from normalized counts.
# 5. UMAP
# l Recommended parameters (optimal parameters are project-dependent; these are suggested as a starting
# point):
# o Minimum distance = 0.01; lower minimum distance generates more clusters.
# o Spread = 5 or 2; higher spread yields more separation of clusters.
# o Neighbors = 30; keep between 20-40; higher value yields more distinct clusters.
# o Metric = cosine.
# o Use between 15-50 principal components.
