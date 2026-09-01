"""
METHOD B (FIXED) — same spatial multi-class GAT, but MDM/microglia/BAM/astro ANCHORS come
from transcriptome markers (Method-A hierarchical gating) instead of GFP+tdTomato reporter
gating. Stages 2/3/5 (marker scoring, vascular proximity, GAT propagation) are IDENTICAL to
the as-is pipeline; only the anchor source changes. This isolates how much the (dead-GFP)
reporter gating distorts the GAT output.

We reuse the pipeline's own functions for scoring/proximity/GAT, and substitute
assign_anchor_labels with anchors derived from Method-A core labels.
"""
import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
_libbin = r"C:\Users\chenr\miniconda3\envs\thesis_research\Library\bin"
os.environ["PATH"] = _libbin + os.pathsep + os.environ.get("PATH", "")
try:
    os.add_dll_directory(_libbin)
except Exception:
    pass
import torch  # FIRST
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, "D:/thesis-research")
import thesis_research.pipeline.cell_type_annotation.myeloid.myeloid_typing as MT
from thesis_research.pipeline.cell_type_annotation.myeloid.myeloid_typing import (
    gate_by_reporters, score_marker_genes, compute_vascular_proximity, train_myeloid_gat,
    CLASS_MDM, CLASS_BAM, CLASS_MICRO_H, CLASS_MICRO_DAM, CLASS_ASTROCYTE, UNLABELED, LABEL_TO_CLASS,
)

OUT = "D:/thesis-research/agents/outputs/annotation"
INP = "D:/thesis-research/resources/cache/annotation_bench/slice1_nontumor.h5ad"
METHOD_A = OUT + "/method_a_labels.csv"

a = ad.read_h5ad(INP)
a.X = a.layers["log1p"].copy()
print("loaded", a.shape)

# --- run the pipeline stages, but supply Method-A anchors at stage 4 ---
a = gate_by_reporters(a)              # still computes gfp_hi/tdt_hi (used by GAT subset selection)
a = score_marker_genes(a)
a = compute_vascular_proximity(a)

# Method-A core labels -> anchor integer labels (transcriptome anchors)
ma = pd.read_csv(METHOD_A, index_col=0)
assert len(ma) == a.n_obs
core = ma["method_a_core"].values
mafine = ma["method_a"].values
labels = np.full(a.n_obs, UNLABELED, dtype=np.int64)
labels[core == "Astrocyte"] = CLASS_ASTROCYTE
labels[core == "BAM"] = CLASS_BAM
labels[core == "MDM"] = CLASS_MDM
# split microglia into homeostatic/DAM using the DAM score if available, else all homeostatic
labels[core == "Microglia"] = CLASS_MICRO_H
# DAM: microglia cells with high DAM score
if "score_microglia_dam" in a.obs:
    dam = a.obs["score_microglia_dam"].to_numpy(float)
    micro_mask = core == "Microglia"
    thr = np.quantile(dam[micro_mask], 0.8) if micro_mask.sum() else 1.0
    labels[(micro_mask) & (dam >= thr)] = CLASS_MICRO_DAM
# leave Myeloid_unresolved + unassigned as UNLABELED so GAT must resolve / -> Other

a.obs["myeloid_anchor_label"] = labels
a.obs["myeloid_is_anchor"] = labels != UNLABELED
import pandas as _pd
ac = _pd.Series(labels[labels != UNLABELED]).map(LABEL_TO_CLASS).value_counts()
print("[FIXED anchors] transcriptome-derived:", int((labels != UNLABELED).sum()))
print(ac.to_string())

t0 = time.time()
a, _ = train_myeloid_gat(a, scgpt_embeddings=None)
print(f"[timing] GAT fixed {time.time()-t0:.1f}s")

df = pd.DataFrame(index=a.obs.index)
df["cell_global_id"] = a.obs["cell_global_id"].values
df["method_b_fixed"] = a.obs["myeloid_cell_type"].astype(str).values
df["method_b_fixed_prob"] = a.obs["myeloid_cell_type_prob"].values
df.to_csv(OUT + "/method_b_fixed_labels.csv", index=True)
print("\n=== METHOD B FIXED myeloid_cell_type ===")
print(df["method_b_fixed"].value_counts())
print("WROTE", OUT + "/method_b_fixed_labels.csv")
