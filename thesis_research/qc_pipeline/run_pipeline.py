import logging

import anndata as ad
import squidpy as sq
from anndata import AnnData

from thesis_research.qc_pipeline.arrangement_plots import (
    generate_fov_arrangement_plot,
    generate_cells_with_area_plot,
)
from thesis_research.qc_pipeline.fov_qc_plots import run_fov_qc

from thesis_research.qc_pipeline.qc_plots import (
    _generate_violin_plots,
    _generate_scatter_plots,
    _generate_histograms,
)
from thesis_research.qc_pipeline.utils import subset_adata_by_fovs, load_slices_from_csv
from thesis_research.utils.columns import AdataUns
from thesis_research.utils.constants import COSMX_RAW_DATA_DIR

from thesis_research.utils.entity_type import EntityType, get_path

LOGGER = logging.getLogger(__name__)


def run_pipeline(fov_qc: bool = False) -> None:
    print("Starting QC pipeline...")

    for sample_dir in sorted(p for p in COSMX_RAW_DATA_DIR.iterdir() if p.is_dir()):
        sample_id = sample_dir.name
        print(f"Processing sample {sample_id}...")

        adata = _get_adata(sample_id)

        slices = load_slices_from_csv(sample_id)

        slice_adatas = []
        for slice_group in slices:
            subset_adata = subset_adata_by_fovs(adata, slice_group)
            slice_adatas.append(subset_adata)
            if fov_qc:
                subset_adata = run_fov_qc(subset_adata)

        # general(adata)
        generate_cells_with_area_plot(adata)
        generate_fov_arrangement_plot(
            "D:\\thesis-research\\resources\\L34\\L34_fov_positions_file.csv"
        )


def _get_adata(sample_id: str) -> AnnData:
    adata_file_path = get_path(EntityType.ADATA_FILE, sample_id)

    if adata_file_path.exists():
        adata = ad.read_h5ad(adata_file_path)
    else:
        adata = _load_adata(sample_id)
        adata.write(adata_file_path, compression="gzip")

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
    print(f"Loading adata for sample {sample_id}")

    sample_dir = get_path(EntityType.SAMPLE_DIR, sample_id)
    adata = sq.read.nanostring(
        path=sample_dir,
        counts_file=f"{sample_id}_exprMat_file.csv",
        meta_file=f"{sample_id}_metadata_file.csv",
        fov_file=f"{sample_id}_fov_positions_file.csv",
    )
    adata.uns[AdataUns.SAMPLE_ID] = sample_id

    print(
        f"Adata {sample_id} initial shapes:\n"
        f"{adata}\n"
        f"{adata.var.head()}\n"
        f"Cells: {adata.n_obs}\n"
        f"Genes: {adata.n_vars}"
    )

    return adata


run_pipeline()
