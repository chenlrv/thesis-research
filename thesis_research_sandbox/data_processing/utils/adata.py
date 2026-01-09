import numpy as np
from anndata import AnnData


def get_adata_section_id(adata: AnnData) -> str:
    return adata.uns.get('section_id', 'unknown')


def add_section_sample_ids(adata: AnnData, section_number: int):
    xs = np.sort(adata.obs["CenterY_global_px"].values)
    diffs = np.diff(xs)
    split_idx = np.argmax(diffs)  # biggest gap
    threshold = xs[split_idx]

    print("Detected split threshold:", threshold)

    adata.obs["sample_id"] = np.where(
        adata.obs["CenterY_global_px"] < threshold, section_number, section_number + 1
    )
    adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
