import os
import sys

import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import _read_var_names, SLICE

with h5py.File(SLICE, "r") as h5:
    v = set(_read_var_names(h5))
cand = ["Csf1r", "C1qa", "C1qb", "C1qc", "Aif1", "Cd68", "Itgam", "Ptprc",
        "Tyrobp", "Fcer1g", "Trem2", "Ctss", "Laptm5", "Fcgr3", "Cx3cr1", "Hexb",
        "P2ry12", "Selplg", "TMEM119", "P2rx5", "Mrc1", "Cd163", "Lyve1", "Ccr2",
        "Plac8", "Cd14", "Pf4", "Marco", "Selenop", "Cd209a", "Csf1", "Fcgr1",
        "Itgax", "Cd84", "Sirpa", "C3ar1", "Lgmn", "Aif1l"]
print("PRESENT:", sorted([g for g in cand if g in v]))
print("MISSING:", sorted([g for g in cand if g not in v]))
