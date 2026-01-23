import numpy as np
from anndata import AnnData
import pandas as pd


def q(x, percentile):
    return x.quantile(percentile)


def load_slices_from_csv(path: str) -> list[list[range]]:
    df = pd.read_csv(path)
    tuples_by_slice = (
        df.sort_values(["slice", "start"])
        .groupby("slice")[["start", "end"]]
        .apply(lambda group: [tuple(x) for x in group.to_numpy()])
        .tolist()
    )
    return [[range(low, high + 1) for low, high in ranges] for ranges in tuples_by_slice]


def subset_adata_by_fovs(adata: AnnData, fov_ranges: list[range]) -> AnnData:
    fov_vals = adata.obs["fov"].astype(int).to_numpy()
    mask = np.array([any(v in r for r in fov_ranges) for v in fov_vals], dtype=bool)
    return adata[mask].copy()
