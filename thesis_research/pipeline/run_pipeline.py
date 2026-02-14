import logging
import uuid
from collections import defaultdict

import anndata as ad
import pandas as pd
import squidpy as sq
from anndata import AnnData

from thesis_research.pipeline.cell_qc_plots import run_cell_qc
from thesis_research.pipeline.filters import (
    get_negative_system_probes,
    remove_negative_system_probes,
)
from thesis_research.pipeline.fov_qc_plots import run_fov_qc
from thesis_research.pipeline.position_plots import generate_position_plots

from thesis_research.pipeline.sample_slice import SampleSlice
from thesis_research.pipeline.utils import get_slice_adata, load_slices_from_csv
from thesis_research.utils.columns import (
    SAMPLE_ID,
    CELL_GLOBAL_ID,
    CELL_ID,
    FOV,
    SLICE_ID,
    SLICE_TYPE,
    MOUSE_ID,
)
from thesis_research.utils.constants import COSMX_RAW_DATA_DIR, CACHE_DIR_PATH
from thesis_research.utils.entity_type import (
    SampleEntityType,
    get_sample_resource_path,
    get_global_resource_path,
    GlobalEntityType,
)

LOGGER = logging.getLogger(__name__)


def run_pipeline(run_exploration: bool = False, run_qc: bool = False) -> None:
    run_id = str(uuid.uuid4())
    print(f"🧬 Starting pipeline with run_id = {run_id}...")

    if run_exploration:
        slice_adatas = run_exploration_pipeline(run_id)
    else:
        slice_adatas = _load_slice_adatas()

    if run_qc:
        run_qc_pipeline(run_id, slice_adatas)

    run_clustering_pipeline()



def run_exploration_pipeline(run_id: str, save: bool = False) -> list[AnnData]:
    print(f"🔵 Starting exploration pipeline...")
    slice_adatas: dict[str, AnnData] = {}

    for sample_dir in sorted(p for p in COSMX_RAW_DATA_DIR.iterdir() if p.is_dir()):
        sample_id = sample_dir.name

        print(f"🕒 Processing sample {sample_id}...")

        adata = _get_adata(sample_id)
        generate_position_plots(adata, run_id)

        sample_slice_adatas = {
            slice_id: get_slice_adata(adata, slice_id)
            for slice_id in adata.obs[SLICE_ID].cat.categories
        }

        slice_adatas.update(sample_slice_adatas)

    if save:
        for slice_id, adata in slice_adatas.items():
            adata.write_h5ad(CACHE_DIR_PATH / f"slice_{slice_id}_adata.h5ad", compression="gzip")
            print(f"💾 Saved adata for slice {slice_id} to cache!")

    print("☑️ Successfully finished exploration! ☑️")
    return list(slice_adatas.values())


def run_qc_pipeline(
    run_id: str, slice_adatas: list[AnnData], fov_qc: bool = False, save: bool = False
) -> None:
    print(f"🔵 Starting QC pipeline...")

    processed_slice_adatas = []
    for adata in slice_adatas:
        slice_id = adata.uns[SLICE_ID]
        print(f"🕒 Processing QC for slice {slice_id}...")

        if fov_qc:
            adata = run_fov_qc(adata, run_id)

        adata = run_cell_qc(adata, run_id)
        probes = get_negative_system_probes(adata)
        adata = remove_negative_system_probes(adata, probes)

        processed_slice_adatas.append(adata)

    sample_to_slices: dict[str, list[AnnData]] = defaultdict(list)
    for adata in processed_slice_adatas:
        sample_id = adata.uns[SAMPLE_ID]
        sample_to_slices[sample_id].append(adata)

    sample_adatas = []

    for sample_id, adatas in sample_to_slices.items():
        sample_adata = _merge_slice_adatas(adatas, sample_id)
        sample_adatas.append(sample_adata)

    if save:
        _save_adatas(processed_slice_adatas, sample_adatas)

    # plot_spatial_gene_positive(s, f"Lyve1 Positive Cells for sample {sample_id}")

    # analysis_adata = _merge_sample_adatas(adatas)
    # run_umap(run_id)
    print("☑️ Successfully finished QC! ☑️")


def _get_adata(sample_id: str) -> AnnData:
    print(f"🕒 Getting adata for sample {sample_id}...")
    adata_file_path = get_sample_resource_path(SampleEntityType.ADATA_FILE, sample_id)

    if adata_file_path.exists():
        print("Loading cached adata...")
        adata = ad.read_h5ad(adata_file_path)
    else:
        print("Loading adata from raw files...")
        adata = _load_adata(sample_id)
        adata = _add_unique_id_per_cell(adata, sample_id)

        slices = load_slices_from_csv(sample_id)
        adata = _add_slice_metadata(adata, slices)

        adata.write(adata_file_path, compression="gzip")

    print(f"Adata {sample_id} initial shapes:\n" f"Cells: {adata.n_obs}\n" f"Genes: {adata.n_vars}")

    return adata


def _add_unique_id_per_cell(adata: AnnData, sample_id: str) -> AnnData:
    if CELL_GLOBAL_ID not in adata.obs.columns:
        adata.obs[CELL_GLOBAL_ID] = adata.obs[CELL_ID].astype(str) + "_" + sample_id

    adata.obs_names = adata.obs[CELL_GLOBAL_ID].astype(str)

    return adata


def _add_slice_metadata(adata: AnnData, slices: list[SampleSlice]) -> AnnData:
    fov_to_slice_id = {fov: s.slice_id for s in slices for fov in s.fov_ids}
    slice_id_to_type = {s.slice_id: s.slice_type for s in slices}
    slice_id_to_mouse_id = {s.slice_id: s.mouse_id for s in slices}

    fovs = pd.to_numeric(adata.obs[FOV])
    adata.obs[SLICE_ID] = fovs.map(fov_to_slice_id).astype("category")
    adata.obs[SLICE_TYPE] = adata.obs[SLICE_ID].map(slice_id_to_type).astype("category")
    adata.obs[MOUSE_ID] = adata.obs[SLICE_ID].map(slice_id_to_mouse_id).astype("category")

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


def _load_slice_adatas() -> list[AnnData]:
    cache_dir = get_global_resource_path(GlobalEntityType.CACHE_DIR)
    paths = sorted(cache_dir.glob("slice_*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No slice_*.h5ad found in {cache_dir}")
    return [ad.read_h5ad(p) for p in paths]


def _merge_slice_adatas(slice_adatas: list[AnnData], sample_id: str) -> AnnData:
    adata = ad.concat(slice_adatas)
    adata.uns[SAMPLE_ID] = sample_id
    return adata


def _save_adatas(slice_adatas: list[AnnData], sample_adatas: list[AnnData]) -> None:
    for slice_adata in slice_adatas:
        slice_id = slice_adata.uns[SLICE_ID]
        slice_adata.write_h5ad(CACHE_DIR_PATH / f"slice_{slice_id}_adata.h5ad", compression="gzip")
        print(f"💾 Saved adata for slice {slice_id} to cache!")

    for sample_adata in sample_adatas:
        sample_id = sample_adata.uns[SAMPLE_ID]
        sample_adata.write_h5ad(
            CACHE_DIR_PATH / f"sample_{sample_id}_adata.h5ad", compression="gzip"
        )
        print(f"💾 Saved adata for sample {sample_id} to cache!")

    adata = _merge_sample_adatas(sample_adatas)
    adata.write_h5ad(CACHE_DIR_PATH / "adata_full.h5ad", compression="gzip")
    print(f"💾 Saved merged adata for all samples to cache!")


def _merge_sample_adatas(adatas: list[AnnData]) -> AnnData:
    return ad.concat(
        adatas,
        keys=[a.uns[SAMPLE_ID] for a in adatas],
        label=SAMPLE_ID,
    )


run_pipeline(run_exploration=True)
