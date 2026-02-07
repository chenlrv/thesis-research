from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCES_DIR = PROJECT_ROOT / "resources"
COSMX_RAW_DATA_DIR = RESOURCES_DIR / "cosmx"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


CACHE_DIR_PATH = RESOURCES_DIR / "cache"
SAMPLE_DIR_PATH = "{cosmx_resources}/{sample_id}"
METADATA_FILE_PATH = "{cosmx_resources}/{sample_id}/{sample_id}_metadata_file.csv"
FOV_POSITIONS_FILE_PATH = "{cosmx_resources}/{sample_id}/{sample_id}_fov_positions_file.csv"
EXPRESSION_MATRIX_FILE_PATH = "{cosmx_resources}/{sample_id}/{sample_id}_exprMat_file.csv"
ADATA_FILE_PATH = "{cosmx_resources}/{sample_id}/{sample_id}_adata.h5ad"
FOV_SLICES_FILE_PATH = "{cosmx_resources}/{sample_id}/{sample_id}_fov_slices.csv"
SLICE_TYPES_FILE_PATH = "{cosmx_resources}/{sample_id}/{sample_id}_slice_types.csv"
