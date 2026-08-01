from anndata import AnnData

from scipy.io import mmwrite
import anndata as ad


def extract_features_for_pca(adata: AnnData) -> None:
    assert adata.obs_names.is_unique
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X

    mmwrite("counts_slice5.mtx", X)
    adata.var_names.to_series().to_csv("genes_slice5.txt", index=False, header=False)
    adata.obs_names.to_series().to_csv("cells_slice5.txt", index=False, header=False)
    adata.obs.to_csv("metadata_slice5.csv")


adata = ad.read_h5ad("D:\\thesis-research\\resources\\cache\\slice_5_adata.h5ad")
extract_features_for_pca(adata)
