"""
Spatial maps of each method's focal labels + per-subtype confidence summary.
Run after 04_compare.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "D:/thesis-research/agents/outputs/annotation"
lab = pd.read_csv(OUT + "/all_method_labels.csv", index_col=0)
CLASSES = ["Microglia", "MDM", "BAM", "Astrocyte", "Other"]
colors = {"Microglia": "#1f77b4", "MDM": "#d62728", "BAM": "#2ca02c",
          "Astrocyte": "#9467bd", "Other": "#dddddd"}

x = lab["CenterX_global_px"].values
y = lab["CenterY_global_px"].values

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, m, title in zip(axes.ravel(), ["A", "Basis", "Bfixed", "C"],
                        ["A: gating", "B-asis: GAT(reporter)", "B-fixed: GAT(transcriptome)", "C: InSituType-ML"]):
    # plot Other first (background), then focal on top
    oth = lab[m] == "Other"
    ax.scatter(x[oth], y[oth], s=0.5, c=colors["Other"], linewidths=0)
    for cls in ["Astrocyte", "Microglia", "MDM", "BAM"]:
        mm = lab[m] == cls
        ax.scatter(x[mm], y[mm], s=1.2, c=colors[cls], linewidths=0, label=cls)
    ax.set_title(f"{title}  (n focal={(~oth).sum()})", fontsize=11)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=8, fontsize=8, loc="upper right")
fig.suptitle("Slice 1 non-tumor — focal cell types by method", fontsize=14)
fig.tight_layout()
fig.savefig(OUT + "/spatial_maps_by_method.png", dpi=120)
plt.close(fig)
print("WROTE spatial_maps_by_method.png")

# ---- per-subtype confidence / marker-specificity summary ----
# For each focal class & method, report mean key-marker counts (own vs cross) on cells it labels.
rows = []
markmap = {
    "Microglia": ["cnt_Cx3cr1", "cnt_TMEM119", "cnt_P2rx5"],
    "MDM": ["cnt_Ccr2", "cnt_Plac8"],
    "BAM": ["cnt_Mrc1", "cnt_Cd163"],
    "Astrocyte": ["cnt_GFAP"],
}
for m in ["A", "Bfixed", "C", "Basis"]:
    for cls in ["Microglia", "MDM", "BAM", "Astrocyte"]:
        sel = lab[lab[m] == cls]
        if len(sel) == 0:
            continue
        d = {"method": m, "class": cls, "n": len(sel),
             "median_nCount": round(float(sel["nCount"].median()), 1)}
        # own-marker positivity
        own = markmap[cls]
        ownpos = (sel[own] > 0).any(axis=1).mean() if all(c in sel for c in own) else np.nan
        d["own_marker_pos_frac"] = round(float(ownpos), 3)
        # cross contamination: fraction positive for OTHER lineages' specific markers
        cross_cols = []
        for ocls, ocols in markmap.items():
            if ocls != cls:
                cross_cols += ocols
        crosspos = (sel[cross_cols] > 0).any(axis=1).mean()
        d["cross_marker_pos_frac"] = round(float(crosspos), 3)
        rows.append(d)
conf = pd.DataFrame(rows)
conf.to_csv(OUT + "/per_subtype_confidence.csv", index=False)
print("\n=== Per-subtype confidence (own vs cross marker positivity) ===")
print(conf.to_string(index=False))
print("\nWROTE per_subtype_confidence.csv")
