import logging
import uuid

import anndata as ad
import squidpy as sq
from anndata import AnnData

from thesis_research.qc_pipeline.analysis import run_umap
from thesis_research.qc_pipeline.cell_qc_plots import run_cell_qc
from thesis_research.qc_pipeline.position_plots import generate_position_plots

from thesis_research.qc_pipeline.fov_qc_plots import run_fov_qc

from thesis_research.qc_pipeline.utils import get_slice_adata, load_slices_from_csv
from thesis_research.utils.columns import SAMPLE_ID, CELL_ID_UNIQUE, CELL_ID
from thesis_research.utils.constants import COSMX_RAW_DATA_DIR

from thesis_research.utils.entity_type import SampleEntityType, get_sample_resource_path

LOGGER = logging.getLogger(__name__)


def run_pipeline(fov_qc: bool = False, position_plots: bool = True) -> None:
    run_id = str(uuid.uuid4())

    print(f"🧬 Starting QC pipeline run = {run_id}...")
    adatas = []
    for sample_dir in sorted(
        p for p in COSMX_RAW_DATA_DIR.iterdir() if p.is_dir() and str(p.name).startswith("L")
    ):
        sample_id = sample_dir.name
        print(f"🕒 Processing sample {sample_id}...")

        adata = _get_adata(sample_id)
        adata = _add_unique_id_per_cell(adata, sample_id)
        slices = load_slices_from_csv(sample_id)

        if position_plots:
            generate_position_plots(adata, slices, run_id)

        slice_adatas = []
        for sample_slice in slices:
            slice_adata = get_slice_adata(adata, sample_slice)

            if fov_qc:
                slice_adata = run_fov_qc(slice_adata, run_id)

            slice_adata = run_cell_qc(slice_adata, sample_id, sample_slice, run_id)
            slice_adatas.append(slice_adata)

        adatas.append(_merge_slice_adatas(slice_adatas, sample_id))
        print(f"✅ Successfully done QC for sample {sample_id}!")

    analysis_adata = _merge_sample_adatas(adatas)
    run_umap(run_id)
    print("☑️ Successfully merged all sample adatas! ☑️")


def _get_adata(sample_id: str) -> AnnData:
    print(f"🕒 Getting adata for sample {sample_id}...")
    adata_file_path = get_sample_resource_path(SampleEntityType.ADATA_FILE, sample_id)

    if adata_file_path.exists():
        print("Loading cached adata...")
        adata = ad.read_h5ad(adata_file_path)
    else:
        print("Loading adata from raw files...")
        adata = _load_adata(sample_id)
        adata.write(adata_file_path, compression="gzip")

    print(
        f"Adata {sample_id} initial shapes:\n"
        f"{adata}\n"
        f"{adata.var.head()}\n"
        f"Cells: {adata.n_obs}\n"
        f"Genes: {adata.n_vars}"
    )

    return adata


def _add_unique_id_per_cell(adata: AnnData, sample_id: str) -> AnnData:
    if CELL_ID_UNIQUE not in adata.obs.columns:
        adata.obs[CELL_ID_UNIQUE] = adata.obs[CELL_ID].astype(str) + "_" + sample_id
    return adata


def _load_adata(sample_id: str) -> AnnData:
    sample_dir = get_sample_resource_path(SampleEntityType.SAMPLE_DIR, sample_id)
    adata = sq.read.nanostring(
        path=sample_dir,
        counts_file=f"{sample_id}_exprMat_file.csv",
        meta_file=f"{sample_id}_metadata_file.csv",
        fov_file=f"{sample_id}_fov_positions_file.csv",
    )
    adata.uns[SAMPLE_ID] = sample_id

    return adata


def _merge_slice_adatas(slice_adatas: list[AnnData], sample_id: str) -> AnnData:
    adata = ad.concat(slice_adatas)
    adata.uns[SAMPLE_ID] = sample_id
    return adata


def _merge_sample_adatas(adatas: list[AnnData]) -> AnnData:
    return ad.concat(
        adatas,
        keys=[a.uns[SAMPLE_ID] for a in adatas],
        label=SAMPLE_ID,
        index_unique="-",  # makes obs_names like "L321-1_1"
    )


run_pipeline()
