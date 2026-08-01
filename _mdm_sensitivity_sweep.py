"""Diagnose why tumor-slice MDM counts are low: sweep the MDM control-percentile
bar and report per-slice MDM count + purity, plus how many `unknown` cells are
near-miss MDM (mdm score between the p95 and p99 control bars)."""
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

myes = [load_myeloid(n)[0] for n in SLICES]
comb = ad.concat(myes, join="inner", index_unique="-")
scores = {m: sg(comb, MODULES[m], f"sc_{m}") for m in MODULES}
sl = comb.obs["slice"].to_numpy()
is_ctrl = sl == CONTROL_SLICE

ccr2 = _dense(comb[:, "Ccr2"].X).ravel()
s100 = _dense(comb[:, "S100a8"].X).ravel()

# fixed non-MDM bars (same as the gate)
micro_hit = scores["micro"] >= np.quantile(scores["micro"][is_ctrl], Q["micro"])
bam_hit = scores["bam"] >= np.quantile(scores["bam"][is_ctrl], Q["bam"])
bstab_hit = scores["bam_stable"] >= np.quantile(scores["bam_stable"][is_ctrl], Q["bam_stable"])

print("\n=== MDM count vs control-percentile bar (BAM/micro fixed) ===")
rows = []
for q in [0.90, 0.95, 0.97, 0.99]:
    thr = np.quantile(scores["mdm"][is_ctrl], q)
    mdm_hit = (scores["mdm"] >= thr) & ~bstab_hit & ~bam_hit
    for n in SLICES:
        m = mdm_hit & (sl == n)
        rows.append({"mdm_Q": f"p{int(q*100)}", "slice": n,
                     "thr": round(float(thr), 3), "MDM_n": int(m.sum()),
                     "pct_myeloid": round(100 * m.sum() / (sl == n).sum(), 1),
                     "Ccr2": round(float(ccr2[m].mean()), 2) if m.sum() else np.nan,
                     "S100a8": round(float(s100[m].mean()), 2) if m.sum() else np.nan})
print(pd.DataFrame(rows).to_string(index=False))

# how much MDM signal is hiding in the p99 `unknown` bin?
print("\n=== near-miss MDM sitting in `unknown` at p99 (per tumor slice) ===")
thr99 = np.quantile(scores["mdm"][is_ctrl], 0.99)
thr95 = np.quantile(scores["mdm"][is_ctrl], 0.95)
for n in SLICES:
    sm = sl == n
    tot = int(sm.sum())
    called = ((scores["mdm"] >= thr99) & ~bstab_hit & ~bam_hit & sm).sum()
    nearmiss = ((scores["mdm"] >= thr95) & (scores["mdm"] < thr99)
                & ~micro_hit & ~bam_hit & sm).sum()
    print(f"slice {n}: MDM called(p99)={int(called):5d}  "
          f"near-miss(p95-p99, non-micro/bam)={int(nearmiss):5d}")
