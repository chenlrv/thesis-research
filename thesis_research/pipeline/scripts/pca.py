from anndata import AnnData
from thesis_research.pipeline.filters import (
    get_negative_system_probes,
    remove_negative_system_probes,
)
from scipy.io import mmwrite
from thesis_research.utils.columns import CELL_ID_UNIQUE


def extract_features_for_pca(adata: AnnData) -> None:
    probes = get_negative_system_probes(adata)
    adata = remove_negative_system_probes(adata, probes)

    adata.obs_names = adata.obs[CELL_ID_UNIQUE].astype(str)
    assert adata.obs_names.is_unique
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X

    mmwrite("counts.mtx", X)
    adata.var_names.to_series().to_csv("genes.txt", index=False, header=False)
    adata.obs_names.to_series().to_csv("cells.txt", index=False, header=False)
    adata.obs.to_csv("metadata.csv")
