"""Figure DQ2 -- the tdTomato reporter does not report lineage, and Lyve1 does not
anchor BAM.

(a) Percentage of cells positive for tdTomato against the endogenous pan-myeloid
    transcripts Cx3cr1 and Csf1r, per section.
(b) The same three probes contrasted between the sham-injected control section and
    the tumor-bearing sections of each slide. Monocyte-derived cells are recruited
    to the lesion, so a monocyte-restricted tag must rise from control to tumor.
All panels are computed over every cell of each section: no cell-type or tumor
assignment enters the figure, only which section a cell came from.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_reporter.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
CSV = ROOT + "/thesis_plots/reporter_prevalence_all6.csv"
OUT = ROOT + "/thesis_plots/dq_fig_reporter.png"
SLICES = [1, 2, 3, 4, 5, 6]
SLIDE = {1: "L321", 2: "L321", 3: "L321", 4: "L34", 5: "L34", 6: "L34"}
CONTROL = {3, 4}
GENES = ["tdTomato", "Cx3cr1", "Csf1r", "Lyve1", "Mrc1", "Cd163"]

RED, BLUE, GREY, LIGHT = "#D62728", "#0072B2", "#6A6A6A", "#C9CDD1"
COL = {"tdTomato": RED, "Cx3cr1": GREY, "Csf1r": "#A9AEB3"}


def compute():
    rows = []
    for s in SLICES:
        a = ad.read_h5ad(WN.format(s))
        v = a.var_names.astype(str)

        def pos(name):
            k = [x for x in v if x.lower() == name.lower()]
            if not k:
                return None
            y = a[:, k[0]].X
            y = y.toarray().ravel() if issparse(y) else np.asarray(y).ravel()
            return y > 0

        p = {g: pos(g) for g in GENES}
        ly = p["Lyve1"]
        bam = p["Mrc1"] | p["Cd163"]
        rows.append(dict(
            slice=s, slide=SLIDE[s], control=s in CONTROL, n=a.n_obs,
            tdTomato=100 * p["tdTomato"].mean(), Cx3cr1=100 * p["Cx3cr1"].mean(),
            Csf1r=100 * p["Csf1r"].mean(),
            lyve1_pos=int(ly.sum()), lyve1_with_bam=int((ly & bam).sum())))
        print(f"slice {s} done")
    d = pd.DataFrame(rows)
    d.to_csv(CSV, index=False)
    return d


d = pd.read_csv(CSV) if os.path.exists(CSV) else compute()
d["label"] = ["Slice {} ({})".format(r.slice, "C" if r.control else "T")
              for r in d.itertuples()]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.5), dpi=200,
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ---- (a) prevalence per section ------------------------------------------------
probes = ["tdTomato", "Cx3cr1", "Csf1r"]
x = np.arange(len(d))
w = 0.26
for i, g in enumerate(probes):
    axA.bar(x + (i - 1) * w, d[g], width=w, color=COL[g], label=g)
    for xi, val in zip(x + (i - 1) * w, d[g]):
        axA.text(xi, val + 0.4, "{:.1f}".format(val), ha="center", va="bottom", fontsize=6.4)
axA.set_xticks(x)
axA.set_xticklabels(d["label"], rotation=45, ha="right", fontsize=8.5)
axA.set_ylabel("Cells positive (%)")
axA.set_ylim(0, 30)          # headroom so the legend clears the tallest value label
axA.legend(fontsize=8.5, frameon=False, loc="upper left")
axA.spines[["top", "right"]].set_visible(False)

# ---- (b) control vs tumor, within slide ----------------------------------------
offsets = {"L321": 0, "L34": 1.6}
for slide, off in offsets.items():
    sub = d[d.slide == slide]
    ctrl = sub[sub.control]
    tum = sub[~sub.control]
    for g in probes:
        y0 = float(ctrl[g].iloc[0])
        ys = tum[g].to_numpy()
        axB.plot([off, off + 0.8], [y0, ys.mean()], "-", color=COL[g], lw=2, zorder=2)
        axB.plot([off], [y0], "o", color=COL[g], ms=7, zorder=3)
        axB.plot([off + 0.8] * len(ys), ys, "o", color=COL[g], ms=5, mfc="white",
                 mew=1.6, zorder=3)
    axB.text(off + 0.4, 24.2, slide, ha="center", fontsize=9.5, fontweight="bold")
axB.set_xticks([0, 0.8, 1.6, 2.4])
axB.set_xticklabels(["control", "tumor", "control", "tumor"], fontsize=8.5)
axB.set_ylabel("Cells positive (%)")
axB.set_ylim(0, 25.5)
axB.set_xlim(-0.35, 2.75)
axB.spines[["top", "right"]].set_visible(False)
axB.legend(handles=[Line2D([], [], color=COL[g], lw=2, marker="o", label=g) for g in probes]
           + [Line2D([], [], color="#444", ls="none", marker="o", ms=5, mfc="white",
                     mew=1.6, label="individual tumor sections")],
           fontsize=8, frameon=False, loc="lower right")

for ax, lab in ((axA, "a"), (axB, "b")):
    ax.text(-0.02, 1.06, "(" + lab + ")", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
