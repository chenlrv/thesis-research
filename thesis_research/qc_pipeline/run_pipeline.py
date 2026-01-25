import logging
import uuid

import anndata
import anndata as ad
import squidpy as sq
from anndata import AnnData

from thesis_research.qc_pipeline.analysis import extract_features_for_pca
from thesis_research.qc_pipeline.cell_qc_plots import run_cell_qc
from thesis_research.qc_pipeline.position_plots import generate_position_plots, plot_counts_over_space

from thesis_research.qc_pipeline.fov_qc_plots import run_fov_qc

from thesis_research.qc_pipeline.utils import subset_adata_by_fovs, load_slices_from_csv
from thesis_research.utils.columns import SAMPLE_ID
from thesis_research.utils.constants import COSMX_RAW_DATA_DIR

from thesis_research.utils.entity_type import EntityType, get_path

LOGGER = logging.getLogger(__name__)


def run_pipeline(fov_qc: bool = False, position_plots: bool = False) -> None:
    run_id = str(uuid.uuid4())

    print(f"Starting QC pipeline run = {run_id}...")
    adatas = []
    for sample_dir in sorted(p for p in COSMX_RAW_DATA_DIR.iterdir() if p.is_dir()):
        sample_id = sample_dir.name
        print(f"Processing sample {sample_id}...")

        adata = _get_adata(sample_id)
        slices = load_slices_from_csv(sample_id)

        if position_plots:
            generate_position_plots(adata, slices, run_id)

        slice_adatas = []
        for sample_slice in slices:
            subset_adata = subset_adata_by_fovs(adata, sample_slice)
            if fov_qc:
                subset_adata = run_fov_qc(subset_adata, run_id)

            subset_adata = run_cell_qc(subset_adata, sample_id, sample_slice, run_id)
            slice_adatas.append(subset_adata)

        adatas.append(anndata.concat(slice_adatas))
        print(f"✅ Successfully done QC for sample {sample_id}!")

    analysis_adata = anndata.concat(adatas)
    extract_features_for_pca(analysis_adata)




def _get_adata(sample_id: str) -> AnnData:
    print(f"Getting adata for sample {sample_id}...")
    adata_file_path = get_path(EntityType.ADATA_FILE, sample_id)

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


def _load_adata(sample_id: str) -> AnnData:
    sample_dir = get_path(EntityType.SAMPLE_DIR, sample_id)
    adata = sq.read.nanostring(
        path=sample_dir,
        counts_file=f"{sample_id}_exprMat_file.csv",
        meta_file=f"{sample_id}_metadata_file.csv",
        fov_file=f"{sample_id}_fov_positions_file.csv",
    )
    adata.uns[SAMPLE_ID] = sample_id
    return adata


run_pipeline()
