import numpy as np
from anndata import AnnData
import pandas as pd

from thesis_research.utils.columns import SLICE, START, END, FOV
from thesis_research.utils.constants import FOV_SLICES_FILE_PATH, COSMX_RAW_DATA_DIR


def load_slices_from_csv(sample_id: str) -> list[list[range]]:
    fov_slices_file_path = FOV_SLICES_FILE_PATH.format(root=COSMX_RAW_DATA_DIR, sample_id=sample_id)

    df = pd.read_csv(fov_slices_file_path)
    tuples_by_slice = (
        df.sort_values([SLICE, START])
        .groupby(SLICE)[[START, END]]
        .apply(lambda group: [tuple(x) for x in group.to_numpy()])
        .tolist()
    )
    return [[range(low, high + 1) for low, high in ranges] for ranges in tuples_by_slice]


def subset_adata_by_fovs(adata: AnnData, fov_ranges: list[range]) -> AnnData:
    fov_vals = adata.obs[FOV].astype(int).to_numpy()
    mask = np.array([any(v in r for r in fov_ranges) for v in fov_vals], dtype=bool)
    return adata[mask].copy()
