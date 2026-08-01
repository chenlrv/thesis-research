"""Is the large Microglia bin real or an artifact of the permissive p10 bar?
Reports (1) broad-myeloid fraction, (2) pan-myeloid + microglia purity of the
current microglia bin, (3) a sweep of the microglia control-percentile bar."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_genes"))

import anndata as ad
import numpy as np
import pandas as pd
from myeloid_subtype_gate import (load_myeloid, MODULES, Q, sg, _dense,
                                  SLICES, CONTROL_SLICE)

myes, ntsize = [], {}
for n in SLICES:
    mye, bg = load_myeloid(n)
    myes.append(mye)
    ntsize[n] = len(bg["cxnt"])          # non-tumor cell count
comb = ad.concat(myes, join="inner", index_unique="-")
scores = {m: sg(comb, MODULES[m], f"sc_{m}") for m in MODULES}
sl = comb.obs["slice"].to_numpy()
is_ctrl = sl == CONTROL_SLICE

print("\n=== broad-myeloid fraction of non-tumor cells ===")
for n in SLICES:
    nmye = int((sl == n).sum())
    print(f"slice {n}: myeloid {nmye:6d} / non-tumor {ntsize[n]:6d} = "
          f"{100*nmye/ntsize[n]:.1f}%")

PAN = ["Csf1r", "C1qa", "Aif1", "Cx3cr1"]        # pan-myeloid (should be high if real)
MIC = ["TMEM119", "Adgrg1"]                       # microglia-specific
def mean(g, m):
    return round(float(_dense(comb[:, g].X).ravel()[m].mean()), 2) if g in comb.var_names and m.sum() else np.nan

# fixed MDM/BAM bars (current gate, p97)
bam_hit = scores["bam"] >= np.quantile(scores["bam"][is_ctrl], Q["bam"])
bstab_hit = scores["bam_stable"] >= np.quantile(scores["bam_stable"][is_ctrl], Q["bam_stable"])
mdm_hit = (scores["mdm"] >= np.quantile(scores["mdm"][is_ctrl], Q["mdm"])) & ~bstab_hit

print("\n=== Microglia bar sweep (MDM/BAM fixed at p97). thr / count / purity ===")
print("micro score>0 means enriched ABOVE expression-matched background.")
rows = []
for qlabel, q in [("p10*(current)", 0.10), ("p25", 0.25), ("p50", 0.50),
                  ("abs>0", None)]:
    if q is None:
        thr = 0.0
    else:
        thr = float(np.quantile(scores["micro"][is_ctrl], q))
    micro_hit = scores["micro"] >= thr
    # same precedence as the gate
    sub = np.full(comb.n_obs, "unknown", dtype=object)
    sub[micro_hit] = "Microglia"
    sub[bam_hit & ~mdm_hit] = "BAM"
    sub[mdm_hit & ~bam_hit] = "MDM"
    sub[mdm_hit & bam_hit] = "unknown"
    for n in SLICES:
        m = (sub == "Microglia") & (sl == n)
        u = (sub == "unknown") & (sl == n)
        rows.append({"micro_bar": qlabel, "thr": round(thr, 2), "slice": n,
                     "Microglia": int(m.sum()), "unknown": int(u.sum()),
                     **{g: mean(g, m) for g in MIC + PAN}})
df = pd.DataFrame(rows)
print(df.to_string(index=False))

print("\n=== current microglia bin vs unknown bin (p10), pan-myeloid contrast ===")
thr = float(np.quantile(scores["micro"][is_ctrl], 0.10))
micro_hit = scores["micro"] >= thr
sub = np.full(comb.n_obs, "unknown", dtype=object)
sub[micro_hit] = "Microglia"
sub[bam_hit & ~mdm_hit] = "BAM"; sub[mdm_hit & ~bam_hit] = "MDM"; sub[mdm_hit & bam_hit] = "unknown"
for n in SLICES:
    for lab in ["Microglia", "unknown"]:
        m = (sub == lab) & (sl == n)
        print(f"slice {n} {lab:9s} n={int(m.sum()):6d}  "
              + "  ".join(f"{g}={mean(g,m)}" for g in MIC + PAN))
