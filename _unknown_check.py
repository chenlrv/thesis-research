"""Are the large `unknown` cells genuinely ambiguous, or microglia the 5-gene
module misses because it leans on TMEM119 (a weak custom probe)? Compare the
microglia vs unknown bins on brain-biased microglia markers NOT in the module
(Cx3cr1, C1qa/b/c) and on pan-myeloid identity."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_genes"))

import anndata as ad
import numpy as np
from myeloid_subtype_gate import (load_myeloid, MODULES, Q, FLOOR, sg, _dense,
                                  SLICES, CONTROL_SLICE, assign)

comb = ad.concat([load_myeloid(n)[0] for n in SLICES], join="inner", index_unique="-")
scores = {m: sg(comb, MODULES[m], f"sc_{m}") for m in MODULES}
sl = comb.obs["slice"].to_numpy()
is_ctrl = sl == CONTROL_SLICE
labels, thr, hits = assign(scores, is_ctrl)

CHK = ["TMEM119", "Adgrg1", "Cx3cr1", "C1qa", "C1qb", "Csf1r", "Aif1", "Ccr2", "Pf4"]
def m(g, mask):
    return round(float(_dense(comb[:, g].X).ravel()[mask].mean()), 2) if g in comb.var_names and mask.sum() else np.nan
def frac_expr(g, mask):
    return round(float((_dense(comb[:, g].X).ravel()[mask] > 0).mean()), 2) if g in comb.var_names and mask.sum() else np.nan

print("\n=== mean expression: Microglia bin vs unknown bin ===")
for n in SLICES:
    for lab in ["Microglia", "unknown"]:
        mask = (labels == lab) & (sl == n)
        print(f"s{n} {lab:9s} n={int(mask.sum()):6d}  "
              + "  ".join(f"{g}={m(g,mask)}" for g in CHK))

print("\n=== % of cells EXPRESSING each gene (raw>0): micro vs unknown ===")
for n in SLICES:
    for lab in ["Microglia", "unknown"]:
        mask = (labels == lab) & (sl == n)
        print(f"s{n} {lab:9s}  "
              + "  ".join(f"{g}={frac_expr(g,mask)}" for g in ["TMEM119", "Cx3cr1", "C1qa", "Ccr2", "Pf4"]))

# how many `unknown` look like microglia on Cx3cr1/C1q (brain-biased) despite
# failing the TMEM119-heavy module?
print("\n=== unknown cells that look microglia-like (Cx3cr1+ & C1qa+ & not MDM/BAM hit) ===")
cx = _dense(comb[:, "Cx3cr1"].X).ravel() > 0
c1 = _dense(comb[:, "C1qa"].X).ravel() > 0
for n in SLICES:
    u = (labels == "unknown") & (sl == n)
    looklike = u & cx & c1 & ~hits["mdm"] & ~hits["bam"]
    print(f"s{n}: unknown={int(u.sum()):6d}  Cx3cr1+&C1qa+ within unknown={int(looklike.sum()):6d} "
          f"({100*looklike.sum()/max(u.sum(),1):.0f}%)")
