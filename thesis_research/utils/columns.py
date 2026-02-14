FOV = "fov"
X_GLOBAL_PX = "x_global_px"
Y_GLOBAL_PX = "y_global_px"
SLICE_ID = "slice_id"


class AdataObs:
    FOV = FOV
    CELL_ID = "cell_id"
    SLIDE_ID = "slide_id"
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
    AREA_MICROMETERS = "Area.um2"
    LOW_RNA_COUNT = "low_count"
    SAMPLE_ID = "sample_id"
    SLICE_ID = SLICE_ID
    SLICE_TYPE = "slice_type"
    MOUSE_ID = "mouse_id"
    CELL_GLOBAL_ID = "cell_global_id"


CELL_ID = AdataObs.CELL_ID
SLIDE_ID = AdataObs.SLIDE_ID
N_COUNT_RNA = AdataObs.N_COUNT_RNA
N_FEATURE_RNA = AdataObs.N_FEATURE_RNA
N_COUNT_NEGPROBES = AdataObs.N_COUNT_NEGPROBES
PROP_NEGATIVE = AdataObs.PROP_NEGATIVE
PERCENT_NEGPRB = AdataObs.PERCENT_NEGPRB
CENTER_X_GLOBAL_PX = AdataObs.CENTER_X_GLOBAL_PX
CENTER_Y_GLOBAL_PX = AdataObs.CENTER_Y_GLOBAL_PX
AREA = AdataObs.AREA
AREA_MICROMETERS = AdataObs.AREA_MICROMETERS
LOW_RNA_COUNT = AdataObs.LOW_RNA_COUNT
SAMPLE_ID = AdataObs.SAMPLE_ID
SLICE_TYPE = AdataObs.SLICE_TYPE
MOUSE_ID = AdataObs.MOUSE_ID
CELL_GLOBAL_ID = AdataObs.CELL_GLOBAL_ID


class FovSlicesCol:
    SLICE = "slice"
    START = "start"
    END = "end"


SLICE = FovSlicesCol.SLICE
START = FovSlicesCol.START
END = FovSlicesCol.END


class FovPositionsCol:
    FOV = FOV
    X_GLOBAL_PX = X_GLOBAL_PX
    Y_GLOBAL_PX = Y_GLOBAL_PX
    SLICE_ID = SLICE_ID
