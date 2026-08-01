"""Detection reliability as a two-panel dot-and-range plot (mirrors the table):
probes on a shared y-axis; left = S/N, right = specificity index. Mean across the
six slices with a [min-max] whisker. Reads the CSV produced by
make_detection_reliability_6slice.py.

Run: conda run -n thesis_research python thesis_plots/make_snspec_dotplot.py
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = "D:/thesis-research"
CSV = ROOT + "/thesis_plots/detection_reliability_all6.csv"
OUT = ROOT + "/thesis_plots/detection_snspec_all6.png"

df = pd.read_csv(CSV)
df["spec"] = df["above_bg"] / 100.0

g = df.groupby(["probe", "cls"]).agg(
    SN_m=("SN", "mean"), SN_lo=("SN", "min"), SN_hi=("SN", "max"),
    sp_m=("spec", "mean"), sp_lo=("spec", "min"), sp_hi=("spec", "max"),
).reset_index()
g = g.sort_values("SN_m", ascending=True).reset_index(drop=True)   # lowest at bottom

def color(row):
    if row["probe"] == "GFP":
        return "#c1121f"          # failed probe
    return "#0072B2" if row["cls"] == "custom" else "#9A9A9A"

cols = [color(r) for _, r in g.iterrows()]
y = np.arange(len(g))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6), dpi=150, sharey=True)

# ---- (a) S/N, log scale ----
for i, r in g.iterrows():
    axL.plot([r.SN_lo, r.SN_hi], [i, i], color=cols[i], lw=2, alpha=0.6, zorder=1)
    axL.scatter(r.SN_m, i, s=55, color=cols[i], zorder=2, edgecolor="white", lw=0.8)
axL.axvline(1, ls="--", lw=1.2, color="#444")
axL.text(1, len(g) - 0.3, "background (S/N = 1)", rotation=90, va="top", ha="right",
         fontsize=8, color="#444")
axL.set_xscale("log")
axL.set_xlim(0.7, 160)
axL.set_yticks(y)
axL.set_yticklabels(g["probe"])
axL.set_xlabel("S/N  (log scale)")
axL.set_title("(a) Signal-to-background", fontsize=11)
axL.grid(axis="x", color="#eee")

# ---- (b) specificity index, 0-1 ----
for i, r in g.iterrows():
    axR.plot([r.sp_lo, r.sp_hi], [i, i], color=cols[i], lw=2, alpha=0.6, zorder=1)
    axR.scatter(r.sp_m, i, s=55, color=cols[i], zorder=2, edgecolor="white", lw=0.8)
axR.axvline(0.5, ls="--", lw=1.2, color="#444")
axR.text(0.5, len(g) - 0.3, "50% — half is noise", rotation=90, va="top", ha="right",
         fontsize=8, color="#444")
axR.set_xlim(0, 1.02)
axR.set_xlabel("Specificity index  (fraction of signal above background)")
axR.set_title("(b) Detection specificity", fontsize=11)
axR.grid(axis="x", color="#eee")

# legend
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2", label="custom probe", markersize=8),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#9A9A9A", label="panel reference", markersize=8),
       Line2D([0], [0], marker="o", color="w", markerfacecolor="#c1121f", label="GFP (failed)", markersize=8)]
axR.legend(handles=leg, fontsize=8.5, frameon=False, loc="lower right")

fig.suptitle("Custom-probe detection reliability across all six slices "
             "(dot = mean, bar = min–max across slices; non-tumor cells)",
             fontsize=12, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved:", OUT)
