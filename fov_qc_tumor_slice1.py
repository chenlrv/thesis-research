"""
Slice 1 (L321): FOV-level QC tiers + tumor content, as a spatial FOV-tile map.

Panel A: each FOV drawn as a tile, colored by QC tier (good/usable/poor).
Panel B: each FOV tile colored by tumor-cell FRACTION; dot size = tumor-cell COUNT.
Also: scatter of QC score vs tumor fraction to show their relationship.
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

H5AD = "resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
FOVPOS = "D:/20251214_CosMx_ReuvenStein/20251214_CosMx_ReuvenStein.tar/Analysis/L321__1__31_12_2025_12_32_59_204/flatFiles/L321/L321_fov_positions_file.csv"
OUTDIR = "fov_qc"
PITCH = 4256.0  # FOV size in global px (grid spacing from fov_positions)
os.makedirs(OUTDIR, exist_ok=True)

a = ad.read_h5ad(H5AD)
o = a.obs.copy()
o["fov"] = o["fov"].astype(int)
o["_tumor"] = o["pred_tumor_XGBoost"].astype(bool)

# ---- per-FOV aggregation ----
g = o.groupby("fov")
fov = pd.DataFrame({
    "n_cells":      g.size(),
    "med_DAPI":     g["Mean.DAPI"].median(),
    "med_NucArea":  g["NucArea"].median(),
    "med_nCount":   g["nCount_RNA"].median(),
    "med_nFeature": g["nFeature_RNA"].median(),
    "tumor_count":  g["_tumor"].sum(),
    "cx":           g["CenterX_global_px"].mean(),
    "cy":           g["CenterY_global_px"].mean(),
})
fov["tumor_frac"] = fov["tumor_count"] / fov["n_cells"]
if "qcFlagsFOV" in o.columns:
    fov["atomx_fail"] = g["qcFlagsFOV"].first().astype(str).eq("Fail").values

# ---- composite QC score (density + nuclei + counts) ----
good = ["n_cells", "med_DAPI", "med_nCount", "med_nFeature", "med_NucArea"]
Z = (fov[good] - fov[good].mean()) / fov[good].std(ddof=0)
fov["qc_score"] = Z.mean(axis=1)

# ---- 3 tiers: poor = AtoMx-fail | bottom-decile; good = top third; else usable ----
q10 = fov["qc_score"].quantile(0.10)
q66 = fov["qc_score"].quantile(0.66)
poor = (fov["qc_score"] <= q10)
if "atomx_fail" in fov.columns:
    poor = poor | fov["atomx_fail"]
tier = np.where(poor, "poor", np.where(fov["qc_score"] >= q66, "good", "usable"))
fov["qc_tier"] = tier

# ---- true FOV centers from fov_positions (robust anchor via median offset) ----
pos = pd.read_csv(FOVPOS).rename(columns=lambda c: c.strip())
pos = pos.set_index("FOV")
fov = fov.join(pos[["x_global_px", "y_global_px"]], how="left")
off_x = float((fov["cx"] - fov["x_global_px"]).median())
off_y = float((fov["cy"] - fov["y_global_px"]).median())
fov["fx"] = fov["x_global_px"] + off_x  # FOV center X
fov["fy"] = fov["y_global_px"] + off_y  # FOV center Y
print(f"anchor offset (px): dx={off_x:.0f} dy={off_y:.0f}  (≈{PITCH/2:.0f} => stored coord is a corner)")

fov = fov.sort_values("qc_score")
fov.to_csv(f"{OUTDIR}/slice1_fov_qc_tumor.csv")

# ---- summary ----
print("\ntier counts:\n", fov["qc_tier"].value_counts())
print("\nmean by tier:\n", fov.groupby("qc_tier")[["n_cells", "med_DAPI", "med_nCount", "tumor_count", "tumor_frac"]].mean().round(2))
print(f"\nFOVs with >=1 tumor cell: {(fov['tumor_count'] > 0).sum()} / {len(fov)}")
print(f"total tumor cells: {int(fov['tumor_count'].sum())} ({100*o['_tumor'].mean():.1f}% of slice)")

def tiles(ax, xc, yc, side, facecolors):
    patches = [Rectangle((x - side / 2, y - side / 2), side, side) for x, y in zip(xc, yc)]
    pc = PatchCollection(patches, facecolors=facecolors, edgecolors="white", linewidths=0.3)
    ax.add_collection(pc)
    ax.set_xlim(xc.min() - side, xc.max() + side)
    ax.set_ylim(yc.min() - side, yc.max() + side)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("global X (px)"); ax.set_ylabel("global Y (px)")

# ============ FIG 1: two panels (QC tier | tumor content) ============
fig, (axA, axB) = plt.subplots(1, 2, figsize=(22, 11))

# Panel A: QC tiers
tcol = {"good": "#2ca25f", "usable": "#fec44f", "poor": "#de2d26"}
tiles(axA, fov["fx"], fov["fy"], PITCH, [tcol[t] for t in fov["qc_tier"]])
axA.set_title("A. FOV QC tier")
from matplotlib.patches import Patch
axA.legend(handles=[Patch(facecolor=tcol[k], label=f"{k} ({(fov['qc_tier']==k).sum()})")
                    for k in ["good", "usable", "poor"]], loc="upper right", framealpha=0.9)

# Panel B: tumor fraction (tile color) + tumor count (dot size)
cmap = plt.cm.magma_r
norm = plt.Normalize(0, max(0.01, fov["tumor_frac"].quantile(0.98)))
tiles(axB, fov["fx"], fov["fy"], PITCH, [cmap(norm(v)) for v in fov["tumor_frac"]])
has_t = fov["tumor_count"] > 0
axB.scatter(fov.loc[has_t, "fx"], fov.loc[has_t, "fy"],
            s=np.sqrt(fov.loc[has_t, "tumor_count"]) * 6, c="#00e5ff",
            edgecolors="k", linewidths=0.3, alpha=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
plt.colorbar(sm, ax=axB, shrink=0.6, label="tumor-cell fraction")
axB.set_title("B. Tumor content per FOV  (color = fraction, cyan dot size = # tumor cells)")
plt.suptitle(f"Slice 1 — FOV QC & tumor content   ({len(fov)} FOVs, "
             f"{(fov['tumor_count']>0).sum()} contain tumor)", fontsize=14)
plt.tight_layout(); plt.savefig(f"{OUTDIR}/slice1_qc_tumor_panels.png", dpi=150); plt.close()

# ============ FIG 2: QC score vs tumor fraction ============
fig, ax = plt.subplots(figsize=(8, 6))
for k in ["poor", "usable", "good"]:
    m = fov["qc_tier"] == k
    ax.scatter(fov.loc[m, "qc_score"], fov.loc[m, "tumor_frac"], s=28,
               c=tcol[k], edgecolors="k", linewidths=0.3, label=k, alpha=0.85)
ax.set_xlabel("FOV QC score (higher = better)"); ax.set_ylabel("tumor-cell fraction")
ax.set_title("Slice 1 — QC score vs tumor content per FOV"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUTDIR}/slice1_qc_vs_tumor.png", dpi=150); plt.close()

print(f"\nwrote: {OUTDIR}/slice1_fov_qc_tumor.csv + 2 PNGs")
