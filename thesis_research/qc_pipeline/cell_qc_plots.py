from pathlib import Path

import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np

from thesis_research.qc_pipeline.sample_slice import SampleSlice
from thesis_research.utils.columns import N_COUNT_RNA, LOW_RNA_COUNT
from thesis_research.utils.entity_type import get_output_dir


def run_cell_qc(adata: AnnData, sample_id: str, sample_slice: SampleSlice, run_id: str) -> AnnData:
    print("Generating cell QC plots...")
    adata_counts = adata.obs[N_COUNT_RNA]
    threshold, flag = _low_count_flag(adata_counts)
    _generate_log2_count_cutoff_histogram(
        sample_id, sample_slice, adata_counts, threshold, flag, get_output_dir(sample_id, run_id)
    )

    num_cells_to_remove = flag.sum()
    total_cells = len(flag)
    print(
        f"Filtering out {num_cells_to_remove} of {total_cells} cells by count threshold ({threshold:.2f}), ({((num_cells_to_remove / total_cells) * 100):.2f}%) based on low RNA counts."
    )

    adata.obs[LOW_RNA_COUNT] = flag
    return adata[~adata.obs[LOW_RNA_COUNT]].copy()


def _low_count_flag(
    counts: np.ndarray, quantile: float = 0.05, cap: int = 20
) -> tuple[float, np.ndarray]:
    """
    Generate a flag for low count cells.
    """
    counts = np.asarray(counts, dtype=float)
    threshold = min(cap, np.nanquantile(counts, quantile))
    flag = counts < threshold
    return threshold, flag


def _generate_log2_count_cutoff_histogram(
    sample_id: str,
    sample_slice: SampleSlice,
    adata_counts: pd.Series,
    threshold: float,
    flag: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Generate a histogram of the log2 counts per cell.
    """
    fig = plt.figure(figsize=(3.0, 2.2), dpi=300)
    num_cells_to_remove = flag.sum()
    total_cells = len(flag)
    textstr = f"Cells to filter: {num_cells_to_remove}/{total_cells} ({(num_cells_to_remove / total_cells) * 100:.1f}%)"
    plt.gca().text(
        0.5,
        -0.25,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.hist(np.log2(adata_counts[adata_counts > 0]), bins=100)
    plt.axvline(np.log2(threshold), color="red")
    plt.xlabel("Log2 counts per cell")
    plt.ylabel("N cells")
    plt.title(f"{sample_id} slice {sample_slice.slice_id} Log2 counts per cell histogram")
    plt.legend([f"{flag.mean() * 100:.1f}% rejected"], loc="upper left")
    plt.savefig(
        output_dir / f"slice_{sample_slice.slice_id}_log2_count_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# qcFlagsCellCounts - Cell failed QC based on RNA count thresholds
# qcFlagsCellPropNeg - Cell failed QC due to high negative-probe proportion
# qcFlagsCellComplex - Cell failed QC due to low complexity
# qcFlagsCellArea - Cell failed QC due to abnormal area


# qc(adata)


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
