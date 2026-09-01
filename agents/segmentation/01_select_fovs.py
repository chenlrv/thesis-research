"""
Step 1: Select ~5 representative slice-1 FOVs from L321 with healthy cell counts
and appreciable myeloid content.

Slice 1 (run L321) = FOVs [303-525] U [587-620]  (tumor, mouse 2).

We rank candidate FOVs by:
  - n cells passing QC (qcCellsPassed) -- want healthy counts
  - myeloid fraction: cells with C1qa/C1qb/Csf1r/Cx3cr1/Ctss positive (proxy)
We avoid the FOV-level qcFlagsFOV == Fail where possible.

Run:
  conda run -n thesis_research python agents/segmentation/01_select_fovs.py
"""
import os
import numpy as np
import pandas as pd

BASE = r"D:\20251214_CosMx_ReuvenStein\20251214_CosMx_ReuvenStein.tar\Analysis\L321__1__31_12_2025_12_32_59_204\flatFiles\L321"
META = os.path.join(BASE, "L321_metadata_file.csv")
EXPR = os.path.join(BASE, "L321_exprMat_file.csv")
OUT = r"D:\thesis-research\agents\outputs\segmentation"
os.makedirs(OUT, exist_ok=True)

SLICE1_FOVS = set(range(303, 526)) | set(range(587, 621))

# Myeloid proxy markers present on panel (microglia + macrophage core)
MYELOID = ["C1qa", "C1qb", "C1qc", "Csf1r", "Cx3cr1", "Ctss", "Tyrobp", "Hexb"]

print("Loading metadata (subset cols)...")
meta_cols = ["fov", "cell_ID", "cell", "nCount_RNA", "nFeature_RNA",
             "qcCellsPassed", "qcFlagsFOV", "Area.um2", "CenterX_global_px",
             "CenterY_global_px", "propNegative"]
meta = pd.read_csv(META, usecols=lambda c: c in meta_cols)
print("meta shape:", meta.shape, "cols:", list(meta.columns))

m1 = meta[meta["fov"].isin(SLICE1_FOVS) & (meta["cell_ID"] != 0)].copy()
print(f"slice-1 cells in metadata: {len(m1):,} across {m1['fov'].nunique()} FOVs")

# Per-FOV cell counts and QC
fov_stats = m1.groupby("fov").agg(
    n_cells=("cell", "size"),
    median_nCount=("nCount_RNA", "median"),
    median_nFeature=("nFeature_RNA", "median"),
    median_area=("Area.um2", "median"),
    fov_qc=("qcFlagsFOV", "first"),
).reset_index()

# Now bring in myeloid content from exprMat for slice-1 FOVs only.
# exprMat has a 'fov' and 'cell_ID' column plus one column per gene.
print("Scanning exprMat header for myeloid genes...")
hdr = pd.read_csv(EXPR, nrows=0)
present = [g for g in MYELOID if g in hdr.columns]
print("myeloid genes present:", present)
usecols = ["fov", "cell_ID"] + present

myeloid_frac = {}
n_myeloid = {}
chunks = pd.read_csv(EXPR, usecols=usecols, chunksize=200_000)
agg = []
for ch in chunks:
    ch = ch[ch["fov"].isin(SLICE1_FOVS) & (ch["cell_ID"] != 0)]
    if len(ch) == 0:
        continue
    ch = ch.copy()
    ch["myeloid_pos"] = (ch[present] >= 1).sum(axis=1) >= 3  # >=3 of myeloid markers
    agg.append(ch.groupby("fov")["myeloid_pos"].agg(["sum", "size"]))
mye = pd.concat(agg).groupby(level=0).sum()
mye["myeloid_frac"] = mye["sum"] / mye["size"]
mye = mye.rename(columns={"sum": "n_myeloid", "size": "n_cells_expr"})

fov_stats = fov_stats.merge(mye[["n_myeloid", "myeloid_frac"]], left_on="fov", right_index=True, how="left")

# Selection: healthy cell count (>=300 cells), reasonable counts, and high myeloid fraction.
cand = fov_stats[(fov_stats["n_cells"] >= 300) & (fov_stats["median_nCount"] >= 50)].copy()
cand = cand.sort_values("myeloid_frac", ascending=False)
print("\nTop candidate FOVs by myeloid fraction (n_cells>=300, median nCount>=50):")
print(cand.head(15).to_string(index=False))

# Pick 5 spread across the candidate pool but prefer high myeloid + passing FOV QC
cand_pass = cand[cand["fov_qc"] != "Fail"]
pool = cand_pass if len(cand_pass) >= 5 else cand
chosen = pool.head(5)["fov"].tolist()
# If fewer than 5 with pass, top up from full candidate list
if len(chosen) < 5:
    extra = [f for f in cand.head(10)["fov"].tolist() if f not in chosen]
    chosen += extra[: 5 - len(chosen)]
chosen = sorted(chosen[:5])
print("\nCHOSEN FOVs:", chosen)

fov_stats.to_csv(os.path.join(OUT, "fov_stats_slice1.csv"), index=False)
with open(os.path.join(OUT, "chosen_fovs.txt"), "w") as f:
    f.write(",".join(map(str, chosen)))
print("Wrote fov_stats_slice1.csv and chosen_fovs.txt")
