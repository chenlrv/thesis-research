from enum import Enum
from pathlib import Path
import uuid

from thesis_research.utils.constants import (
    ADATA_FILE_PATH,
    COSMX_RAW_DATA_DIR,
    FOV_POSITIONS_FILE_PATH,
    FOV_SLICES_FILE_PATH,
    SAMPLE_DIR_PATH,
    OUTPUTS_DIR,
)


class SampleEntityType(str, Enum):
    # files
    ADATA_FILE = "adata"
    FOV_POSITIONS_FILE = "fov_positions"
    FOV_SLICES_FILE = "fov_slices"
    SAMPLE_DIR = "sample_dir"


class GlobalEntityType(str, Enum):
    X_PCA_PEARSON_BATCH_FILE = "x_pca_pearson_batch"
    METADATA_FILE = "metadata"


def get_sample_resource_path(entity_type: SampleEntityType, sample_id: str) -> Path:
    templates = {
        SampleEntityType.ADATA_FILE: ADATA_FILE_PATH,
        SampleEntityType.FOV_POSITIONS_FILE: FOV_POSITIONS_FILE_PATH,
        SampleEntityType.FOV_SLICES_FILE: FOV_SLICES_FILE_PATH,
        SampleEntityType.SAMPLE_DIR: SAMPLE_DIR_PATH,
    }

    try:
        template = templates[entity_type]
        return Path(template.format(resources=COSMX_RAW_DATA_DIR, sample_id=sample_id))
    except KeyError:
        raise ValueError(f"Unsupported type: {entity_type}")


def get_global_resource_path(entity_type: GlobalEntityType) -> Path:
    templates = {
        GlobalEntityType.X_PCA_PEARSON_BATCH_FILE: COSMX_RAW_DATA_DIR / "x_pca_pearson_batch.csv",
        GlobalEntityType.METADATA_FILE: COSMX_RAW_DATA_DIR / "metadata.csv",
    }

    try:
        return templates[entity_type]
    except KeyError:
        raise ValueError(f"Unsupported type: {entity_type}")


def get_output_dir(run_id: str, sample_id: str = None) -> Path:
    run_id = run_id or str(uuid.uuid4())
    output_dir = OUTPUTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_id is None:
        return output_dir

    output_dir = output_dir / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
