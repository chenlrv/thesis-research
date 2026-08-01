"""Cross-check the REAL high-confidence myeloid set (5,538 = clean Myeloid &
high_conf in the merged pipeline) against SingleR's brain_myeloid (Brain_Immune)
calls -- with the barcode join proven correct by coordinate alignment.

Reconstruction must match score_genes/myeloid_bam.py exactly:
  SLICE = resources/cache/with_tumor_prediction/slice_1_adata.h5ad
  PROV  = score_genes_slice1_merged/cell_scores.csv      (non-tumor order)
  HC    = score_genes_slice1_merged/high_confidence_labels.csv (labeled subset)
  myeloid = (provisional_label == "Myeloid") & high_conf
"""
import os, sys
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, r"D:\thesis-research\score_genes")
import check_homogeneity as ch  # exact readers / loader used by the pipeline

ch.SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
SR_CSV = (r"D:\thesis-research\outputs\cell_annotation\L321\05\1"
          r"\slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")


def read_obs_index(path):
    with h5py.File(path, "r") as h5:
        obs = h5["obs"]
        key = obs.attrs.get("_index", "_index")
        key = key.decode() if isinstance(key, bytes) else key
        return ch._decode(obs[key][...]).astype(str)


# ---- 1) reconstruct the non-tumor adata exactly as the pipeline does ----
ch.SCORES_CSV = PROV
adata = ch.load_adata_with_calls()           # non-tumor, normalized; obs x/y/celltype
celltype = np.asarray(adata.obs["celltype"])
labeled = celltype != "unknown"

# ---- 2) recover barcodes for the SAME non-tumor rows, prove alignment ----
with h5py.File(ch.SLICE, "r") as h5:
    cx = ch._read_obs_num(h5, "CenterX_global_px")
    cy = ch._read_obs_num(h5, "CenterY_global_px")
    tumor = ch._read_obs_bool(h5, ch.TUMOR_COL)
barcodes_all = read_obs_index(ch.SLICE)
bc_nt = barcodes_all[~tumor]                 # non-tumor barcodes, h5ad order
cx_nt, cy_nt = cx[~tumor], cy[~tumor]

assert len(bc_nt) == adata.n_obs, (len(bc_nt), adata.n_obs)
# PROOF the barcode vector is in the same order as the pipeline's x/y:
assert np.allclose(cx_nt, adata.obs["x"].to_numpy(), atol=1e-3)
assert np.allclose(cy_nt, adata.obs["y"].to_numpy(), atol=1e-3)
print(f"barcode<->coord alignment proven for {len(bc_nt):,} non-tumor cells")
print(f"barcode sample: {list(bc_nt[:3])}")

# ---- 3) high-conf myeloid mask (replicates myeloid_bam.py) ----
hc = pd.read_csv(HC)
assert np.allclose(hc["x"].to_numpy(), adata.obs["x"].to_numpy()[labeled], atol=1e-3), \
    "HC not aligned to labeled subset"
high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
prov = hc["provisional_label"].to_numpy()

myeloid = np.zeros(adata.n_obs, bool)
myeloid[np.where(labeled)[0]] = (prov == "Myeloid") & high
n_my = int(myeloid.sum())
print(f"\nhigh-conf myeloid (clean Myeloid & high_conf): {n_my:,}  "
      f"(expected 5,538)")

myeloid_bc = pd.Index(bc_nt[myeloid])

# ---- 4) join to SingleR by barcode ----
sr = pd.read_csv(SR_CSV, dtype=str).set_index("cell_barcode")
matched = myeloid_bc.intersection(sr.index)
print(f"of {n_my:,} high-conf myeloid, found in SingleR csv: {len(matched):,}")
m = sr.loc[matched]

print("\n=== SingleR predicted_cell_type of the high-conf myeloid ===")
vc = m["predicted_cell_type"].value_counts()
for k, v in vc.items():
    print(f"  {k:35s} {v:6,d}  ({100*v/len(m):5.1f}%)")

print("\n=== SingleR predicted_tissue_origin (1=Brain_Struct 2=Brain_Immune 3=Tumor) ===")
for k, v in m["predicted_tissue_origin"].value_counts(dropna=False).items():
    print(f"  origin {str(k):4s} {v:6,d}  ({100*v/len(m):5.1f}%)")

is_brain_myeloid = m["predicted_cell_type"].isin(["macrophage", "microglial cell"])
print(f"\nhigh-conf myeloid that SingleR also calls brain_myeloid "
      f"(macrophage/microglial cell): {int(is_brain_myeloid.sum()):,} / {len(m):,} "
      f"({100*is_brain_myeloid.mean():.1f}%)")

# also report by Brain_Immune reference winning (origin==2)
o2 = (m["predicted_tissue_origin"] == "2").sum()
print(f"high-conf myeloid where Brain_Immune reference won (origin 2): "
      f"{int(o2):,} ({100*o2/len(m):.1f}%)")
