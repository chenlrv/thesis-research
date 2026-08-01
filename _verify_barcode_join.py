"""VERIFY the barcode join is correct before trusting any overlap numbers.

Tests:
 1. format / sample of barcodes on both sides
 2. full coverage + duplicates (is the join 1:1 and complete?)
 3. positional check: is the CSV already in the SAME order as adata.obs?
 4. DECISIVE independent-signal test: SingleR was run on these same counts.
    If the join is correct, cells SingleR calls macrophage/microglial must
    have high pan-myeloid marker counts and high score_brain_immune; cells it
    calls neuron must not. A scrambled join destroys this relationship.
    We compare the TRUE (barcode) join against a SCRAMBLED join as a control.
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

H5 = r"D:\thesis-research\resources\cache\slice_1_adata.h5ad"
MASK = r"D:\thesis-research\_myeloid_mask_slice1.npy"
CSV = (r"D:\thesis-research\outputs\cell_annotation\L321\05\1"
       r"\slice_1_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")

PANMY = ["Csf1r", "C1qa", "C1qb", "C1qc", "Aif1", "Tyrobp", "Itgam", "Ptprc", "Cd68", "Trem2"]

adata = ad.read_h5ad(H5)
obs_names = np.asarray(adata.obs.index, dtype=str)
mask = np.load(MASK)
sr = pd.read_csv(CSV, dtype=str)
sr["cell_barcode"] = sr["cell_barcode"].astype(str)

print("=== 1. barcode formats ===")
print("adata.obs index name:", adata.obs.index.name)
print("adata obs_names[:3]:", list(obs_names[:3]))
print("csv  cell_barcode[:3]:", list(sr["cell_barcode"].iloc[:3]))
print("csv columns:", list(sr.columns))

print("\n=== 2. coverage / duplicates ===")
obs_set = pd.Index(obs_names)
csv_idx = pd.Index(sr["cell_barcode"])
print(f"adata cells: {len(obs_set):,} (unique: {obs_set.nunique():,})")
print(f"csv rows:    {len(csv_idx):,} (unique: {csv_idx.nunique():,})")
print(f"intersection: {obs_set.intersection(csv_idx).nunique():,}")
print(f"in adata not in csv: {len(obs_set.difference(csv_idx)):,}")
print(f"in csv not in adata: {len(csv_idx.difference(obs_set)):,}")

print("\n=== 3. is csv already in adata.obs order? ===")
same_order = (len(sr) == len(obs_names)) and bool(np.all(sr["cell_barcode"].to_numpy() == obs_names))
print("csv row-order identical to adata.obs order:", same_order)

# build the marker count per cell (from adata, the ground truth expression)
cols = [adata.var_names.get_loc(g) for g in PANMY if g in adata.var_names]
X = adata[:, cols].X
X = X.toarray() if sp.issparse(X) else np.asarray(X)
panmy_count = (X >= 1).sum(1)            # # pan-myeloid markers detected, per adata cell
panmy_df = pd.Series(panmy_count, index=obs_names, name="panmy")

print("\n=== 4. DECISIVE independent-signal test (true vs scrambled join) ===")
# TRUE join: map each adata cell to its SingleR label by barcode
sr_lbl = sr.set_index("cell_barcode")["predicted_cell_type"]
sr_imm = pd.to_numeric(sr.set_index("cell_barcode")["score_brain_immune"], errors="coerce")
lbl_true = sr_lbl.reindex(obs_names)
imm_true = sr_imm.reindex(obs_names).to_numpy()

def summarize(label_arr, tag):
    s = pd.DataFrame({"panmy": panmy_count, "lbl": np.asarray(label_arr)})
    myeloid_lbls = ["macrophage", "microglial cell"]
    mean_my = s.loc[s["lbl"].isin(myeloid_lbls), "panmy"].mean()
    mean_neu = s.loc[s["lbl"] == "neuron", "panmy"].mean()
    print(f"  [{tag}] mean pan-myeloid markers: "
          f"SingleR-myeloid={mean_my:.2f}  SingleR-neuron={mean_neu:.2f}  "
          f"separation={mean_my - mean_neu:+.2f}")

summarize(lbl_true.to_numpy(), "TRUE barcode join")

# SCRAMBLED control: shuffle the labels (breaks the cell<->label correspondence)
rng = np.random.default_rng(0)
summarize(lbl_true.sample(frac=1.0, random_state=1).to_numpy(), "SCRAMBLED (shuffled)")

# correlation of score_brain_immune with marker count (true join)
ok = ~np.isnan(imm_true)
r_true = np.corrcoef(imm_true[ok], panmy_count[ok])[0, 1]
imm_scr = pd.Series(imm_true).sample(frac=1.0, random_state=2).to_numpy()
r_scr = np.corrcoef(imm_scr[ok], panmy_count[ok])[0, 1]
print(f"\n  corr(score_brain_immune, pan-myeloid count):  TRUE={r_true:+.3f}   "
      f"SCRAMBLED={r_scr:+.3f}")
print("\nIf TRUE shows clear separation/positive corr and SCRAMBLED ~0, the join is correct.")
