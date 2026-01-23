import logging

import anndata as ad
import squidpy as sq
from anndata import AnnData

from resources.L34.L34_fov_slices import one, two, three
from thesis_research.qc_pipeline.arrangement_plots import (
    generate_fov_arrangement_plot,
    generate_cells_with_area_plot,
)
from thesis_research.qc_pipeline.fov_qc_plots import general
from thesis_research.qc_pipeline.qc_plots import (
    _generate_violin_plots,
    _generate_scatter_plots,
    _generate_histograms,
)
from thesis_research.qc_pipeline.utils import subset_adata_by_fovs
from thesis_research.utils.constants import COSMX_RAW_DATA_DIR, ADATA_FILE_PATTERN

LOGGER = logging.getLogger(__name__)


def run_pipeline():
    print("Starting QC pipeline...")
    for sample_dir in sorted(p for p in COSMX_RAW_DATA_DIR.iterdir() if p.is_dir()):
        sample_id = sample_dir
        print(f"Processing sample: {sample_id}")
        adata = _get_adata(sample_id, adata_path=ADATA_FILE_PATTERN.format(sample_id))

        one_subset = subset_adata_by_fovs(adata, one)
        two_subset = subset_adata_by_fovs(adata, two)

        three_subset = subset_adata_by_fovs(adata, three)

        general(one_subset)
        general(two_subset)
        general(three_subset)

        # general(adata)
        generate_cells_with_area_plot(adata)
        generate_fov_arrangement_plot(
            "D:\\thesis-research\\resources\\L34\\L34_fov_positions_file.csv"
        )


def _get_adata(sample_id: str, adata_path: str = None) -> AnnData:
    if adata_path:
        adata = ad.read_h5ad(adata_path)
    else:
        adata = _load_adata(sample_id)

    return adata


def qc(adata: AnnData) -> None:
    adata.obs["Identity"] = "C"
    x = adata.obs["propNegative"].astype(float)
    adata.obs["percent.NegPrb"] = x * 100 if x.max() <= 1.0 else x

    keys = ["nFeature_RNA", "nCount_RNA", "percent.NegPrb", "Area"]

    _generate_histograms(adata, keys)
    _generate_violin_plots(adata, keys)
    _generate_scatter_plots(adata)


def _load_adata(sample_id: str) -> AnnData:
    print(f"Loading adata for section {sample_id}")

    adata = sq.read.nanostring(
        path=DATA_PATH_TEMPLATE.format(section_id=sample_id),
        counts_file=f"{sample_id}_exprMat_file.csv",
        meta_file=f"{sample_id}_metadata_file.csv",
        fov_file=f"{sample_id}_fov_positions_file.csv",
    )
    adata.uns["section_id"] = sample_id

    print(
        f"Adata {sample_id} initial shapes:\n"
        f"{adata}\n"
        f"{adata.var.head()}\n"
        f"Cells: {adata.n_obs}\n"
        f"Genes: {adata.n_vars}"
    )

    return adata
