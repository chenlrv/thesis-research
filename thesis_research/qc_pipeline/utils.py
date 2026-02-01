from anndata import AnnData
import pandas as pd

from thesis_research.qc_pipeline.sample_slice import SampleSlice
from thesis_research.utils.columns import SLICE, START, END, FOV, SLICE_ID, SLICE_TYPE
from thesis_research.utils.constants import FOV_SLICES_FILE_PATH, COSMX_RAW_DATA_DIR, SLICE_TYPES_FILE_PATH


def load_slices_from_csv(sample_id: str) -> list[SampleSlice]:
    fov_slices_file_path = FOV_SLICES_FILE_PATH.format(
        resources=COSMX_RAW_DATA_DIR, sample_id=sample_id
    )

    df = pd.read_csv(fov_slices_file_path)
    slice_types = load_slice_types_from_csv(sample_id)

    slices: list[SampleSlice] = []
    for i, group in df.sort_values([SLICE, START]).groupby(SLICE):
        fov_ids: set[int] = set()
        for low, high in group[[START, END]].to_numpy():
            fov_ids.update(range(low, high + 1))

        slices.append(SampleSlice(i, fov_ids, slice_types[i]))

    return slices


def load_slice_types_from_csv(sample_id: str) -> dict[int, str]:
    df = pd.read_csv(SLICE_TYPES_FILE_PATH.format(resources=COSMX_RAW_DATA_DIR, sample_id=sample_id))
    return dict(zip(df['slice'], df['type']))


def get_slice_adata(adata: AnnData, sample_slice: SampleSlice) -> AnnData:
    mask = adata.obs[FOV].astype(int).isin(sample_slice.fov_ids).to_numpy()
    adata = adata[mask].copy()
    adata.obs[SLICE_ID] = sample_slice.slice_id
    adata.obs[SLICE_TYPE] = sample_slice.slice_type

    return adata
