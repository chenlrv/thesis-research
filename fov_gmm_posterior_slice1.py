"""
Slices 1-3 (L321): per-slice spatial FOV-tile maps combining
  - tile color  = FOV quality  (two versions: GMM posterior, and continuous QC score)
  - overlay dot = tumor content per FOV (dot size proportional to tumor-cell ratio)

Orientation matches the rest of the pipeline's spatial plots (CenterX/Y_global_px,
no y-axis inversion).
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
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture

FOVPOS_L321 = "D:/20251214_CosMx_ReuvenStein/20251214_CosMx_ReuvenStein.tar/Analysis/L321__1__31_12_2025_12_32_59_204/flatFiles/L321/L321_fov_positions_file.csv"
FOVPOS_L34 = "D:/20251214_CosMx_ReuvenStein/20251214_CosMx_ReuvenStein.tar/Analysis/Reuven_Stein_L34__1__31_12_2025_12_18_22_859/flatFiles/L34/L34_fov_positions_file.csv"
SLICES = (4, 5, 6)  # 1-3 -> L321, 4-6 -> L34
OUTDIR = "fov_qc"
PITCH = 4256.0
FEATS = ["n_cells", "med_DAPI", "med_nCount", "med_nFeature", "med_NucArea"]
os.makedirs(OUTDIR, exist_ok=True)


def load_pos(n):
    fp = FOVPOS_L321 if n <= 3 else FOVPOS_L34
    return pd.read_csv(fp).rename(columns=lambda c: c.strip()).set_index("FOV")


def per_fov_table(o, pos):
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
    Z = ((fov[FEATS] - fov[FEATS].mean()) / fov[FEATS].std(ddof=0)).values
    fov["qc_score"] = Z.mean(axis=1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(Z)
    lab = gm.predict(Z)
    good_comp = pd.Series(fov["qc_score"].values).groupby(lab).mean().idxmax()
    fov["p_good"] = gm.predict_proba(Z)[:, good_comp]
    fov = fov.join(pos[["x_global_px", "y_global_px"]], how="left")
    fov["fx"] = fov["x_global_px"] + float((fov["cx"] - fov["x_global_px"]).median())
    fov["fy"] = fov["y_global_px"] + float((fov["cy"] - fov["y_global_px"]).median())
    return fov


def draw_map(fov, color_vals, clim, cbar_label, title, outfile):
    has_t = fov["tumor_count"] > 0
    fig, ax = plt.subplots(figsize=(13, 11))
    patches = [Rectangle((x - PITCH / 2, y - PITCH / 2), PITCH, PITCH) for x, y in zip(fov["fx"], fov["fy"])]
    pc = PatchCollection(patches, cmap=plt.cm.RdYlGn, edgecolors="white", linewidths=0.3)
    pc.set_array(np.asarray(color_vals)); pc.set_clim(*clim)
    ax.add_collection(pc)
    ax.scatter(fov.loc[has_t, "fx"], fov.loc[has_t, "fy"],
               s=fov.loc[has_t, "tumor_frac"] * 1600, facecolors="none",
               edgecolors="black", linewidths=1.3, alpha=0.9)
    ax.set_xlim(fov["fx"].min() - PITCH, fov["fx"].max() + PITCH)
    ax.set_ylim(fov["fy"].min() - PITCH, fov["fy"].max() + PITCH)
    ax.set_aspect("equal")  # no invert_yaxis -> matches pipeline spatial plots
    ax.set_xlabel("global X (px)"); ax.set_ylabel("global Y (px)")
    plt.colorbar(pc, ax=ax, shrink=0.6, label=cbar_label)
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                      markeredgecolor="black", markersize=np.sqrt(r * 1600),
                      label=f"{int(r*100)}% tumor") for r in [0.05, 0.15, 0.25]]
    ax.legend(handles=handles, title="tumor-cell ratio (dot size)", loc="upper right",
              framealpha=0.9, labelspacing=1.6, borderpad=1.0)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()
    print("  wrote:", outfile)


for n in SLICES:
    h5 = f"resources/cache/with_tumor_prediction/slice_{n}_adata.h5ad"
    a = ad.read_h5ad(h5)
    o = a.obs.copy()
    o["fov"] = o["fov"].astype(int)
    o["_tumor"] = o["pred_tumor_XGBoost"].astype(bool)
    fov = per_fov_table(o, load_pos(n))
    fov.sort_values("qc_score").to_csv(f"{OUTDIR}/slice{n}_fov_qc_tumor.csv")
    has_t = fov["tumor_count"] > 0
    sub = (f"{len(fov)} FOVs, {int(has_t.sum())} contain tumor, "
           f"{int(fov['tumor_count'].sum())} tumor cells ({100*o['_tumor'].mean():.1f}% of slice)")
    print(f"slice {n}: {sub}")
    draw_map(fov, fov["p_good"].values, (0, 1), "GMM P(good-quality cluster)",
             f"Slice {n} — FOV quality (GMM posterior) & tumor ratio\n{sub}",
             f"{OUTDIR}/slice{n}_gmm_tumor_map.png")
    m = float(np.abs(fov["qc_score"]).quantile(0.98))
    draw_map(fov, fov["qc_score"].values, (-m, m), "QC score (higher = better)",
             f"Slice {n} — FOV quality (QC score) & tumor ratio\n{sub}",
             f"{OUTDIR}/slice{n}_qcscore_tumor_map.png")
