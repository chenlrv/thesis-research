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


class EntityType(str, Enum):
    # files
    ADATA_FILE = "adata"
    FOV_POSITIONS_FILE = "fov_positions"
    FOV_SLICES_FILE = "fov_slices"
    SAMPLE_DIR = "sample_dir"


def get_path(entity_type: EntityType, sample_id: str) -> Path:
    templates = {
        EntityType.ADATA_FILE: ADATA_FILE_PATH,
        EntityType.FOV_POSITIONS_FILE: FOV_POSITIONS_FILE_PATH,
        EntityType.FOV_SLICES_FILE: FOV_SLICES_FILE_PATH,
        EntityType.SAMPLE_DIR: SAMPLE_DIR_PATH,
    }

    try:
        template = templates[entity_type]
        return Path(template.format(resources=COSMX_RAW_DATA_DIR, sample_id=sample_id))
    except KeyError:
        raise ValueError(f"Unsupported type: {entity_type}")


def get_output_dir(sample_id: str, run_id: str = None) -> Path:
    run_id = run_id or str(uuid.uuid4())
    sample_output_dir = OUTPUTS_DIR / sample_id / run_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    return sample_output_dir
