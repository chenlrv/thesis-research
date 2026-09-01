"""
METHOD B — spatial multi-class GAT (myeloid_typing.annotate_myeloid_cells, use_scgpt=False).

Run AS-IS (Stage-1 reporter gating on GFP/tdTomato). GFP is a DEAD probe (S/N 1.73),
so this quantifies how much the compromised reporter gating distorts the result.
Writes method_b_asis_labels.csv.

Time budget: GAT runs on a myeloid-focused subset capped at max_gat_cells (20k default),
400 epochs, CPU. Prototype-safe.
"""
import os, sys, time
# --- torch DLL fixes: import torch FIRST, before MKL-heavy numpy/scanpy/anndata ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_libbin = r"C:\Users\chenr\miniconda3\envs\thesis_research\Library\bin"
os.environ["PATH"] = _libbin + os.pathsep + os.environ.get("PATH", "")
try:
    os.add_dll_directory(_libbin)
except Exception:
    pass
import torch  # MUST precede numpy/scanpy to avoid libiomp5md/shm.dll conflict
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, "D:/thesis-research")
from thesis_research.pipeline.cell_type_annotation.myeloid.myeloid_typing import annotate_myeloid_cells

OUT = "D:/thesis-research/agents/outputs/annotation"
os.makedirs(OUT, exist_ok=True)
INP = "D:/thesis-research/resources/cache/annotation_bench/slice1_nontumor.h5ad"

a = ad.read_h5ad(INP)
# GAT scores genes on .X; score_genes(use_raw=False) works on whatever .X holds.
# Pipeline expects normalised/usable .X + X_pca in obsm. Provide log1p in .X for scoring.
a.X = a.layers["log1p"].copy()
print("loaded", a.shape, "| X=log1p, X_pca present:", "X_pca" in a.obsm)

t0 = time.time()
res = annotate_myeloid_cells(a, use_scgpt=False)
print(f"\n[timing] GAT total {time.time()-t0:.1f}s")

df = pd.DataFrame(index=a.obs.index)
df["cell_global_id"] = a.obs["cell_global_id"].values
df["method_b_asis"] = res.obs["myeloid_cell_type"].astype(str).values
df["method_b_asis_prob"] = res.obs["myeloid_cell_type_prob"].values
df["reporter_label"] = res.obs["myeloid_reporter_label"].astype(str).values
df["gfp_hi"] = res.obs["myeloid_gfp_hi"].values
df["tdt_hi"] = res.obs["myeloid_tdt_hi"].values
df["anchor_label"] = res.obs["myeloid_anchor_label"].values
df.to_csv(OUT + "/method_b_asis_labels.csv", index=True)

print("\n=== METHOD B AS-IS myeloid_cell_type ===")
print(df["method_b_asis"].value_counts())
print("\nreporter_label counts:")
print(df["reporter_label"].value_counts())
print("anchors (non -1):", int((df["anchor_label"] != -1).sum()))
print("gfp_hi:", int(df["gfp_hi"].sum()), "tdt_hi:", int(df["tdt_hi"].sum()))
print("\nWROTE", OUT + "/method_b_asis_labels.csv")
