import numpy as np
from anndata import AnnData
import pandas as pd

from thesis_research.qc_pipeline.sample_slice import SampleSlice
from thesis_research.utils.columns import SLICE, START, END, FOV
from thesis_research.utils.constants import FOV_SLICES_FILE_PATH, COSMX_RAW_DATA_DIR


def load_slices_from_csv(sample_id: str) -> list[SampleSlice]:
    fov_slices_file_path = FOV_SLICES_FILE_PATH.format(
        resources=COSMX_RAW_DATA_DIR, sample_id=sample_id
    )

    df = pd.read_csv(fov_slices_file_path)
    slices: list[SampleSlice] = []
    for _, group in df.sort_values([SLICE, START]).groupby(SLICE):
        fov_ids: set[int] = set()
        for low, high in group[[START, END]].to_numpy():
            fov_ids.update(range(low, high + 1))

        slices.append(SampleSlice(fov_ids))

    return slices


def subset_adata_by_fovs(adata: AnnData, sample_slice: SampleSlice) -> AnnData:
    mask = adata.obs[FOV].astype(int).isin(sample_slice.fov_ids).to_numpy()
    return adata[mask].copy()
