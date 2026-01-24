from pathlib import Path

import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np


from thesis_research.utils.columns import N_COUNT_RNA
from thesis_research.utils.entity_type import get_output_dir


def low_count_flag(counts, q=0.05, cap=20):
    counts = np.asarray(counts, dtype=float)
    thr = min(cap, np.nanquantile(counts, q))
    flag = counts < thr
    return thr, flag


def run_cell_qc(adata: AnnData, sample_id:str, sample_slice: str, run_id: str) -> AnnData:
    print("Generating cell QC plots...")
    adata_counts = adata.obs[N_COUNT_RNA]
    threshold, flag = low_count_flag(adata_counts)
    _generate_log2_count_cutoff_histogram(sample_id, sample_slice, adata_counts, threshold, flag, get_output_dir(sample_id, run_id))

    adata.obs["low_count"] = flag
    return adata[~adata.obs["low_count"]].copy()



def _generate_log2_count_cutoff_histogram(
        sample_id: str,
        sample_slice: str,
        adata_counts:
        pd.Series,
        threshold: float,
        flag: np.ndarray,
        output_dir: Path):
    fig = plt.figure(figsize=(3.0, 2.2), dpi=300)

    plt.hist(np.log2(adata_counts[adata_counts > 0]), bins=100)
    plt.axvline(np.log2(threshold), color="red")
    plt.xlabel("Log2 counts per cell")
    plt.ylabel("N cells")
    plt.title(f"{sample_id} slice {sample_slice} Log2 counts per cell histogram")
    plt.legend([f"{flag.mean() * 100:.1f}% rejected"], loc="upper left")
    plt.savefig(output_dir / f"slice_{sample_slice}_log2_count_histogram.png", dpi=300, bbox_inches="tight")

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
