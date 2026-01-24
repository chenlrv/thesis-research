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
    PERCENT_NEGPRB = "percent_NegPrb"
    X_GLOBAL_PX = X_GLOBAL_PX
    Y_GLOBAL_PX = Y_GLOBAL_PX
    CENTER_X_GLOBAL_PX = "CenterX_global_px"
    CENTER_Y_GLOBAL_PX = "CenterY_global_px"
    AREA = "Area"


SLIDE_ID = AdataObs.SLIDE_ID.value
N_COUNT_RNA = AdataObs.N_COUNT_RNA.value
N_FEATURE_RNA = AdataObs.N_FEATURE_RNA.value
N_COUNT_NEGPROBES = AdataObs.N_COUNT_NEGPROBES.value
PROP_NEGATIVE = AdataObs.PROP_NEGATIVE.value
PERCENT_NEGPRB = AdataObs.PERCENT_NEGPRB.value
CENTER_X_GLOBAL_PX = AdataObs.CENTER_X_GLOBAL_PX.value
CENTER_Y_GLOBAL_PX = AdataObs.CENTER_Y_GLOBAL_PX.value
AREA = AdataObs.AREA.value


class AdataUns(str, Enum):
    SAMPLE_ID = "sample_id"


SAMPLE_ID = AdataUns.SAMPLE_ID.value


class FovSlicesCol(str, Enum):
    SLICE = "slice"
    START = "start"
    END = "end"


SLICE = FovSlicesCol.SLICE.value
START = FovSlicesCol.START.value
END = FovSlicesCol.END.value


class FovPositionsCol(str, Enum):
    FOV = FOV
    X_GLOBAL_PX = X_GLOBAL_PX
    Y_GLOBAL_PX = Y_GLOBAL_PX
    SLICE_ID = "slice_ID"


SLICE_ID = FovPositionsCol.SLICE_ID.value
