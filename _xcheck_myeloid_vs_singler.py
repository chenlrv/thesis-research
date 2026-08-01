"""Cross-check: are the score_genes high-confidence myeloid cells (slice 1)
also called myeloid (Brain_Immune / brain_myeloid) by SingleR?

high-conf myeloid = _myeloid_mask_slice1.npy  (>=3/10 pan-myeloid markers,
aligned to slice_1_adata.h5ad obs order).
SingleR = outputs/.../slice_1_..._avinoam.csv (predicted_cell_type,
predicted_tissue_origin).
"""
import numpy as np
import pandas as pd
import anndata as ad

H5 = r"D:\thesis-research\resources\cache\slice_1_adata.h5ad"
MASK = r"D:\thesis-research\_myeloid_mask_slice1.npy"
CSV = (r"D:\thesis-research\outputs\cell_annotation\L321\05\1"
       r"\slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")

adata = ad.read_h5ad(H5, backed="r")
obs_names = np.asarray(adata.obs.index)
adata.file.close()
mask = np.load(MASK)
assert mask.shape[0] == obs_names.shape[0], (mask.shape, obs_names.shape)

myeloid_ids = pd.Index(obs_names[mask])
print(f"slice_1 cells: {len(obs_names):,}")
print(f"high-conf myeloid (score_genes gate): {mask.sum():,}")

sr = pd.read_csv(CSV, dtype=str)
sr["cell_barcode"] = sr["cell_barcode"].astype(str)
sr = sr.set_index("cell_barcode")
print(f"SingleR rows: {len(sr):,}")

# what reference / tissue origin did each label come from?
print("\n=== SingleR predicted_tissue_origin vs predicted_cell_type (whole slice) ===")
print(pd.crosstab(sr["predicted_tissue_origin"], sr["predicted_cell_type"]).T
      .sort_values(sr["predicted_tissue_origin"].unique().tolist()
                   if False else sr["predicted_tissue_origin"].dropna().unique()[0],
                   ascending=False) if False else
      pd.crosstab(sr["predicted_cell_type"], sr["predicted_tissue_origin"]))

# align SingleR onto the high-conf myeloid set
common = myeloid_ids.intersection(sr.index)
print(f"\nhigh-conf myeloid matched in SingleR csv: {len(common):,} / {len(myeloid_ids):,}")
m = sr.loc[common]

print("\n=== SingleR predicted_cell_type of the high-conf myeloid cells ===")
vc = m["predicted_cell_type"].value_counts()
for k, v in vc.items():
    print(f"  {k:35s} {v:7,d}  ({100*v/len(m):5.1f}%)")

print("\n=== SingleR predicted_tissue_origin of the high-conf myeloid cells ===")
vo = m["predicted_tissue_origin"].value_counts(dropna=False)
for k, v in vo.items():
    print(f"  origin {str(k):10s} {v:7,d}  ({100*v/len(m):5.1f}%)")

MYELOID_LABELS = {"macrophage", "microglial cell"}
is_my_label = m["predicted_cell_type"].isin(MYELOID_LABELS)
print(f"\nhigh-conf myeloid called macrophage/microglial by SingleR: "
      f"{is_my_label.sum():,} / {len(m):,} ({100*is_my_label.mean():.1f}%)")

# reverse direction: of all SingleR macrophage/microglia, how many are in our gate?
sr_my = sr.index[sr["predicted_cell_type"].isin(MYELOID_LABELS)]
inter = myeloid_ids.intersection(sr_my)
print(f"SingleR macrophage/microglia total: {len(sr_my):,}; "
      f"of those in score_genes gate: {len(inter):,} ({100*len(inter)/max(len(sr_my),1):.1f}%)")
