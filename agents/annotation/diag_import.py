import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch  # import FIRST, before MKL-heavy scanpy/numpy
print("torch", torch.__version__)
import numpy, anndata, scanpy
print("sci stack imported")
sys.path.insert(0, "D:/thesis-research")
from thesis_research.pipeline.cell_type_annotation.myeloid.myeloid_typing import annotate_myeloid_cells
print("module import OK")
