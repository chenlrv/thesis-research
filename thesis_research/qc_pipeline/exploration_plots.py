from pathlib import Path

from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from thesis_research.utils.columns import (
    PROP_NEGATIVE,
    PERCENT_NEGPRB,
    N_FEATURE_RNA,
    N_COUNT_RNA,
    AREA,
    SAMPLE_ID,
    SLIDE_ID,
)
from thesis_research.utils.entity_type import get_output_dir


def run_exploration(adata: AnnData, sample_id: str, sample_slice: str, run_id: str) -> None:
    print("Generating exploration plots...")
    x = adata.obs[PROP_NEGATIVE].astype(float)
    adata.obs[PERCENT_NEGPRB] = x * 100 if x.max() <= 1.0 else x
    features = [N_FEATURE_RNA, N_COUNT_RNA, PERCENT_NEGPRB, AREA]

    output_dir = get_output_dir(sample_id, run_id)
    for feature in features:
        _generate_histogram(adata, sample_id, sample_slice, feature, output_dir)
        _generate_violin_plot(adata, sample_id, feature, output_dir)

    _generate_scatter_plots(adata)


def _generate_histogram(
    adata: AnnData, sample_id: str, sample_slice: str, feature: str, output_dir: Path
):
    x = adata.obs[feature].to_numpy(dtype=float)

    fig = plt.figure()
    plt.hist(x, bins=100)
    plt.xlabel(feature)
    plt.ylabel("Number of cells")
    plt.title(f"{sample_id} slice {sample_slice} {feature} Histogram")
    plt.savefig(
        output_dir / f"sample_{sample_slice}_{feature}_histogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def _generate_violin_plot(adata: AnnData, sample_id: str, feature: str, output_dir: Path):
    sc.pl.violin(
        adata,
        keys=feature,
        groupby=SLIDE_ID,
        stripplot=False,
        color="#F8766D",
    )
    plt.title(f"{sample_id} {feature} Violin Plot")
    plt.savefig(output_dir / f"{feature}_violit_plot.png", dpi=300, bbox_inches="tight")


def _generate_scatter_plots(adata):
    x1 = adata.obs["nCount_RNA"].to_numpy(dtype=float)
    y1 = adata.obs["percent.NegPrb"].to_numpy(dtype=float)
    m1 = np.isfinite(x1) & np.isfinite(y1)
    r1 = _corrcoef(x1, y1)

    plt.figure(figsize=(3.0, 2.2), dpi=300)
    plt.scatter(x1[m1], y1[m1], s=2, c="#F8766D", alpha=1, linewidths=0)
    plt.xlabel("nCount_RNA")
    plt.ylabel("percent.NegPrb")
    plt.title(f"{r1:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.show()

    x2 = adata.obs["nCount_RNA"].to_numpy(dtype=float)
    y2 = adata.obs["nFeature_RNA"].to_numpy(dtype=float)
    m2 = np.isfinite(x2) & np.isfinite(y2)
    r2 = _corrcoef(x2, y2)

    plt.figure(figsize=(3.0, 2.2), dpi=300)
    plt.scatter(x2[m2], y2[m2], s=2, c="#F8766D", alpha=1, linewidths=0)
    plt.xlabel("nCount_RNA")
    plt.ylabel("nFeature_RNA")
    plt.title(f"{r2:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.show()

    x3 = adata.obs["Area"].to_numpy(dtype=float)
    y3 = adata.obs["nCount_RNA"].to_numpy(dtype=float)
    m3 = np.isfinite(x3) & np.isfinite(y3)
    r3 = _corrcoef(x3, y3)

    plt.figure(figsize=(3.0, 2.2), dpi=300)
    plt.scatter(x3[m3], y3[m3], s=2, c="#F8766D", alpha=1, linewidths=0)
    plt.xlabel("Area")
    plt.ylabel("nCount_RNA")
    plt.title(f"{r3:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.show()


def _corrcoef(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]
