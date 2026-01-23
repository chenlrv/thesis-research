from enum import Enum


class AdataCol(str, Enum):
    FOV = "fov"
    SLIDE_ID = "slide_ID"
    N_COUNT_RNA = "nCount_RNA"
    N_FEATURE_RNA = "nFeature_RNA"
    N_COUNT_NEGPROBES = "nCount_negprobes"
    PROP_NEGATIVE = "propNegative"
    PERCENT_NEGPRB = "percent.NegPrb"
    X_GLOBAL_PX = "x_global_px"
    Y_GLOBAL_PX = "y_global_px"
    CENTER_X_GLOBAL_PX = "CenterX_global_px"
    CENTER_Y_GLOBAL_PX = "CenterY_global_px"
    AREA = "Area"
