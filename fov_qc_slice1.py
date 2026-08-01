"""
FOV-level QC stratification for slice 1 (L321).

Goal: separate trustworthy FOVs (dense, good nuclei, high counts -> tumor/TME)
from untrustworthy ones (sparse, weak DAPI, low counts -> non-tumor haze),
so fine annotation can be restricted to the good FOVs.

Metrics are aggregated per FOV from the per-cell obs table (no pixel work).
Outputs: per-FOV CSV, spatial QC maps, and a distributions figure.
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

H5AD = "resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUTDIR = "fov_qc"
os.makedirs(OUTDIR, exist_ok=True)

a = ad.read_h5ad(H5AD)
o = a.obs.copy()
print(f"loaded {a.n_obs} cells, {o['fov'].nunique()} FOVs")

# ---- tumor fraction (robust to dtype of the predictor column) ----
def tumor_positive(series):
    if series.dtype == bool:
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0.5).astype(float)
    s = series.astype(str).str.lower()
    return s.isin({"1", "true", "tumor", "yes", "malignant"}).astype(float)

tumor_col = "pred_tumor_XGBoost" if "pred_tumor_XGBoost" in o.columns else None
if tumor_col:
    o["_tumor"] = tumor_positive(o[tumor_col])
    print(f"tumor predictor '{tumor_col}' uniques: {list(pd.Series(o[tumor_col].unique())[:6])}")

# ---- negprobe rate per cell ----
o["_negrate"] = o["nCount_negprobes"] / o["nCount_RNA"].clip(lower=1)

# ---- per-FOV aggregation ----
g = o.groupby("fov")
fov = pd.DataFrame({
    "n_cells":      g.size(),
    "med_DAPI":     g["Mean.DAPI"].median(),
    "med_NucArea":  g["NucArea"].median(),
    "med_nCount":   g["nCount_RNA"].median(),
    "med_nFeature": g["nFeature_RNA"].median(),
    "med_Area":     g["Area"].median(),
    "med_MeanG":    g["Mean.G"].median(),
    "negprobe_rate": g["nCount_negprobes"].sum() / g["nCount_RNA"].sum().clip(lower=1),
    "cx":           g["CenterX_global_px"].mean(),
    "cy":           g["CenterY_global_px"].mean(),
})
if tumor_col:
    fov["tumor_frac"] = g["_tumor"].mean()
# carry AtoMx's own FOV flag if present, for comparison
if "qcFlagsFOV" in o.columns:
    fov["atomx_qcFlagsFOV"] = g["qcFlagsFOV"].first().astype(str)

# ---- composite QC score ----
good = ["n_cells", "med_DAPI", "med_nCount", "med_nFeature", "med_NucArea"]
bad = ["negprobe_rate"]
Z = (fov[good + bad] - fov[good + bad].mean()) / fov[good + bad].std(ddof=0)
fov["qc_score"] = Z[good].mean(axis=1) - Z[bad].mean(axis=1)

# ---- flag method A: bottom-quintile of composite score ----
thr = fov["qc_score"].quantile(0.20)
fov["flag_quintile"] = fov["qc_score"] < thr

# ---- flag method B: 2-cluster GMM on standardized metrics ----
feats = good + bad
Zg = ((fov[feats] - fov[feats].mean()) / fov[feats].std(ddof=0)).values
gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(Zg)
lab = gm.predict(Zg)
# the "good" cluster is the one with the higher mean qc_score
good_cluster = fov.groupby(lab)["qc_score"].mean().idxmax()
fov["qc_group"] = np.where(lab == good_cluster, "pass", "flag")

fov = fov.sort_values("qc_score")
fov.to_csv(f"{OUTDIR}/slice1_fov_qc.csv")

# ---- summary ----
npass = (fov["qc_group"] == "pass").sum()
nflag = (fov["qc_group"] == "flag").sum()
print(f"\nGMM: {npass} pass / {nflag} flag  (of {len(fov)} FOVs)")
print(f"quintile flag: {fov['flag_quintile'].sum()} FOVs")
print("\nmean metric by GMM group:")
cols = good + bad + (["tumor_frac"] if tumor_col else [])
print(fov.groupby("qc_group")[cols].mean().round(2).T)
if "atomx_qcFlagsFOV" in fov.columns:
    print("\nAtoMx qcFlagsFOV vs our GMM group:")
    print(pd.crosstab(fov["atomx_qcFlagsFOV"], fov["qc_group"]))

# ---- map cells -> their FOV group for the spatial plot ----
grp = fov["qc_group"].to_dict()
o["_grp"] = o["fov"].map(grp)

# ============ FIG 1: per-cell spatial map colored by FOV QC ============
fig, ax = plt.subplots(figsize=(11, 11))
for gname, col in [("pass", "#2c7fb8"), ("flag", "#d7301f")]:
    m = o["_grp"] == gname
    ax.scatter(o.loc[m, "CenterX_global_px"], o.loc[m, "CenterY_global_px"],
               s=0.6, c=col, alpha=0.35, linewidths=0, label=f"{gname} ({(fov['qc_group']==gname).sum()} FOVs)")
ax.set_aspect("equal"); ax.invert_yaxis()
ax.set_title("Slice 1 — FOV QC (cells colored by their FOV's quality group)")
ax.set_xlabel("global X (px)"); ax.set_ylabel("global Y (px)")
lg = ax.legend(markerscale=12, loc="upper right");
plt.tight_layout(); plt.savefig(f"{OUTDIR}/slice1_fov_qc_map.png", dpi=150); plt.close()

# ============ FIG 2: per-FOV centroids colored by continuous score ============
fig, ax = plt.subplots(figsize=(11, 11))
sc = ax.scatter(fov["cx"], fov["cy"], c=fov["qc_score"], s=140,
                cmap="RdYlBu", edgecolors="k", linewidths=0.4)
ax.set_aspect("equal"); ax.invert_yaxis()
plt.colorbar(sc, ax=ax, shrink=0.6, label="QC score (higher = better)")
ax.set_title("Slice 1 — per-FOV QC score")
ax.set_xlabel("global X (px)"); ax.set_ylabel("global Y (px)")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/slice1_fov_score_map.png", dpi=150); plt.close()

# ============ FIG 3: metric distributions ============
mets = good + bad
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for axi, mname in zip(axes.ravel(), mets):
    for gname, col in [("pass", "#2c7fb8"), ("flag", "#d7301f")]:
        vals = fov.loc[fov["qc_group"] == gname, mname]
        axi.hist(vals, bins=25, alpha=0.6, color=col, label=gname)
    axi.set_title(mname); axi.legend(fontsize=8)
axes.ravel()[-1].axis("off") if len(mets) < 6 else None
plt.suptitle("Slice 1 — per-FOV metric distributions by QC group")
plt.tight_layout(); plt.savefig(f"{OUTDIR}/slice1_fov_metrics.png", dpi=150); plt.close()

print(f"\nwrote: {OUTDIR}/slice1_fov_qc.csv + 3 PNGs")
