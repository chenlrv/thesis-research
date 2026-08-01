import os
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import _read_var_names, SLICE

with h5py.File(SLICE, "r") as h5:
    v = set(_read_var_names(h5))

candidates = {
    "Oligodendrocyte": ["Mbp", "Plp1", "Mog", "Mag", "Sox10", "Cldn11", "Mobp", "Olig1", "Olig2", "Cnp"],
    "OPC": ["Pdgfra", "Cspg4", "Olig1", "Olig2", "Sox10", "Lhfpl3"],
    "T / NK": ["Cd3d", "Cd3e", "Cd3g", "Cd8a", "Cd4", "Nkg7", "Klrb1c", "Gzmb", "Ncr1", "Cd2"],
    "B cell": ["Cd79a", "Cd79b", "Ms4a1", "Cd19", "Ighm", "Jchain"],
    "Fibroblast/meningeal": ["Col1a1", "Col1a2", "Dcn", "Lum", "Pdgfra", "Slc6a13", "Col3a1"],
}
for t, genes in candidates.items():
    present = [g for g in genes if g in v]
    print(f"{t:22s} present: {present}")
