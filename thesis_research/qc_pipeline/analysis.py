import scanpy as sc
import numpy as np
from anndata import AnnData
from thesis_research.qc_pipeline.filters import get_negative_system_probes, remove_negative_system_probes
import scipy.sparse as sp
from scipy.io import mmwrite

def extract_features_for_pca(adata: AnnData):
    probes = get_negative_system_probes(adata)
    adata = remove_negative_system_probes(adata, probes)

    X = adata.layers["counts"] if "counts" in adata.layers else adata.X

    mmwrite("counts.mtx", X)
    adata.obs.to_csv("metadata.csv")
    adata.var_names.to_series().to_csv("genes.txt", index=False)
    adata.obs_names.to_series().to_csv("cells.txt", index=False)



def run_pca(adata: AnnData):
    probes = get_negative_system_probes(adata)
    adata = remove_negative_system_probes(adata, probes)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


