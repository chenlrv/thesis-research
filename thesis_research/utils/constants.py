from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COSMX_RAW_DATA_DIR = PROJECT_ROOT / "resources"

METADATA_FILE_PATTERN = "{sample_id}_metadata_file.csv"
FOV_POSITIONS_FILE_PATTERN = "{sample_id}_fov_positions_file.csv"
EXPRESSION_MATRIX_FILE_PATTERN = "{sample_id}_exprMat_file.csv"
ADATA_FILE_PATTERN = "{sample_id}_adata.h5ad"
FOV_SLICES_FILE_PATTERN = "{sample_id}_fov_slices.py"
