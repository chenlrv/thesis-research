import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scvi

from thesis_research.utils.columns import SAMPLE_ID, SLICE_ID, SLICE_TYPE, MOUSE_ID

adata = ad.read_h5ad("D:\\thesis-research\\resources\\cache\\adata_full.h5ad")
adata.layers["counts"] = adata.X.copy()

BATCH_KEY = SAMPLE_ID
SLICE_KEY = SLICE_ID
MOUSE_KEY = MOUSE_ID
TUMOR_KEY = SLICE_TYPE

adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")

adata.obs[TUMOR_KEY] = adata.obs[TUMOR_KEY].astype("category")
covs = []
covs.append(TUMOR_KEY)

scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key=BATCH_KEY,
    categorical_covariate_keys=covs if covs else None,
)

model = scvi.model.SCVI(adata)
print("starting train")
model.train()

adata.obsm["X_scVI"] = model.get_latent_representation()


print("starting neighbors")
sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=30)
print("starting umap")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0, key_added="leiden_scvi")

sc.pl.umap(adata, color=["leiden_scvi", BATCH_KEY])

adata.write_h5ad("D:\\thesis-research\\resources\\adata_scvi.h5ad", compression="gzip")
model.save("D:\\thesis-research\\resources\\scvi_model\\", overwrite=True)

sc.pl.umap(adata, color=["leiden_scvi", BATCH_KEY], save="_clusters_sample.png")
if TUMOR_KEY in adata.obs.columns:
    sc.pl.umap(adata, color=[TUMOR_KEY], save="_tumor.png")
sc.pl.umap(adata, color=[SLICE_KEY], save="_slice.png")
# sc.pl.umap(adata, color=[MOUSE_KEY], save="_mouse.png")
