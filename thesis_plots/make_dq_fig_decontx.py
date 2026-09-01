"""Figure DQ3 -- ambient correction does not remove the probe anomalies.

Percentage of cells positive for each probe before and after decontX ambient
correction, in every section. The three anomalous probes are unchanged; Meg3,
the most abundant transcript on the panel and hence the largest contributor to
the ambient pool, drops substantially -- showing that the correction acts where
ambient signal genuinely exists.

Computed over every cell of each section. decontX returns an integer corrected
matrix, so positivity is defined identically before and after: at least one count.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_decontx.py
"""
import os

import anndata as ad
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
RAW = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
DX = ROOT + "/resources/cache/decontx/slice_{}_decontx.h5ad"
CSV = ROOT + "/thesis_plots/decontx_before_after.csv"
OUT = ROOT + "/thesis_plots/dq_fig_decontx.png"
SLICES = [1, 2, 3, 4, 5, 6]
CONTROL = {3, 4}
PROBES = ["tdTomato", "GFP", "Lyve1", "Meg3"]
RAW_C, DX_C = "#9A9A9A", "#E67300"


def compute():
    rows = []
    for s in SLICES:
        raw = ad.read_h5ad(RAW.format(s))
        dx = ad.read_h5ad(DX.format(s))
        cont = float(np.median(dx.obs["decontx_contamination"].to_numpy()))

        def vec(a, name):
            k = [x for x in a.var_names.astype(str) if x.lower() == name.lower()]
            if not k:
                return None
            y = a[:, k[0]].X
            return (y.toarray().ravel() if issparse(y) else np.asarray(y).ravel()).astype(float)

        for g in PROBES:
            r, d = vec(raw, g), vec(dx, g)
            rows.append(dict(slice=s, probe=g, contamination=cont,
                             raw=100 * (r > 0).mean(), corrected=100 * (d > 0).mean()))
        print(f"slice {s} done (median contamination {cont:.3f})")
    out = pd.DataFrame(rows)
    out.to_csv(CSV, index=False)
    return out


d = pd.read_csv(CSV) if os.path.exists(CSV) else compute()
labels = ["Slice {} ({})".format(s, "C" if s in CONTROL else "T") for s in SLICES]

fig, axes2 = plt.subplots(2, 2, figsize=(9.6, 8.0), dpi=200, sharey=True)
axes = axes2.ravel()
x = np.arange(len(SLICES))
w = 0.36
for ax, g in zip(axes, PROBES):
    sub = d[d.probe == g].set_index("slice").loc[SLICES]
    ax.bar(x - w / 2, sub["raw"], width=w, color=RAW_C, label="raw counts")
    ax.bar(x + w / 2, sub["corrected"], width=w, color=DX_C, label="after decontX")
    for xi, (a, b) in enumerate(zip(sub["raw"], sub["corrected"])):
        drop = a - b
        if drop >= 1.0:
            ax.text(xi, max(a, b) + 1.2, "-{:.1f}".format(drop), ha="center",
                    va="bottom", fontsize=7.5, color="#B00", fontweight="bold")
    ax.set_title(g, fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes2[:, 0]:
    ax.set_ylabel("Cells positive (%)")
axes[0].set_ylim(0, 50)
axes[0].legend(fontsize=9, frameon=False, loc="upper left")

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
