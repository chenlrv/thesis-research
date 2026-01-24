from enum import Enum

FOV = "fov"
X_GLOBAL_PX = "x_global_px"
Y_GLOBAL_PX = "y_global_px"


class AdataObs(str, Enum):
    FOV = FOV
    SLIDE_ID = "slide_ID"
    N_COUNT_RNA = "nCount_RNA"
    N_FEATURE_RNA = "nFeature_RNA"
    N_COUNT_NEGPROBES = "nCount_negprobes"
    PROP_NEGATIVE = "propNegative"
    PERCENT_NEGPRB = "percent.NegPrb"
    X_GLOBAL_PX = X_GLOBAL_PX
    Y_GLOBAL_PX = Y_GLOBAL_PX
    CENTER_X_GLOBAL_PX = "CenterX_global_px"
    CENTER_Y_GLOBAL_PX = "CenterY_global_px"
    AREA = "Area"


SLIDE_ID = AdataObs.SLIDE_ID
N_COUNT_RNA = AdataObs.N_COUNT_RNA
N_FEATURE_RNA = AdataObs.N_FEATURE_RNA
N_COUNT_NEGPROBES = AdataObs.N_COUNT_NEGPROBES
PROP_NEGATIVE = AdataObs.PROP_NEGATIVE
PERCENT_NEGPRB = AdataObs.PERCENT_NEGPRB
CENTER_X_GLOBAL_PX = AdataObs.CENTER_X_GLOBAL_PX
CENTER_Y_GLOBAL_PX = AdataObs.CENTER_Y_GLOBAL_PX
AREA = AdataObs.AREA


class AdataUns(str, Enum):
    SAMPLE_ID = "sample_id"


SAMPLE_ID = AdataUns.SAMPLE_ID.value


class FovSlicesCol(str, Enum):
    SLICE = "slice"
    START = "start"
    END = "end"


SLICE = FovSlicesCol.SLICE
START = FovSlicesCol.START
END = FovSlicesCol.END


class FovPositionsCol(str, Enum):
    FOV = FOV
    X_GLOBAL_PX = X_GLOBAL_PX
    Y_GLOBAL_PX = Y_GLOBAL_PX
