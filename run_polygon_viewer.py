import anndata as ad
from thesis_research.viewer_tool import show_cell_polygons

POLYGONS = r"D:\thesis-research\resources\cosmx\L321\L321-polygons.csv"
ADATA = r"D:\thesis-research\resources\clustering\adata_full_with_pca_clustered.h5ad"

adata = ad.read_h5ad(ADATA)

show_cell_polygons(
    polygons_path=POLYGONS,
    adata=adata,
    color_col="cluster",
    fovs=None,
    use_global=True,
)
