"""Figure DQ3 -- Lyve1 does not identify border-associated macrophages.

(a) Lyve1-positive cells per section, split by whether the same cell also detects
    a canonical BAM marker (Mrc1 or Cd163).
(b) The same split in space, two sections per row, at full page height.

Computed over every cell of each section -- positivity is raw detection (>= 1
count), so no cell-type or tumor assignment enters the figure. Coordinates are
CenterX/Y_global_px, the tissue-level frame (obsm["spatial"] is FOV-local here).

Colours follow the rest of the data-quality figures: red marks the failing
category (Lyve1 positive with no BAM marker), blue a genuine BAM candidate.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_lyve1.py
"""
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
OUT = ROOT + "/thesis_plots/dq_fig_lyve1.png"
SLICES = [1, 2, 3, 4, 5, 6]
CONTROL = {3, 4}
PX_UM = 0.12028
GREY, RED, BLUE = "#E2E4E6", "#D62728", "#0072B2"


def load(s):
    a = ad.read_h5ad(WN.format(s))
    v = a.var_names.astype(str)

    def cnt(n):
        k = [x for x in v if x.lower() == n.lower()]
        if not k:
            return np.zeros(a.n_obs)
        y = a[:, k[0]].X
        return (y.toarray().ravel() if issparse(y) else np.asarray(y).ravel()).astype(float)

    x = a.obs["CenterX_global_px"].to_numpy() * PX_UM / 1000.0
    y = a.obs["CenterY_global_px"].to_numpy() * PX_UM / 1000.0
    lyve = cnt("Lyve1") > 0
    bam = (cnt("Mrc1") + cnt("Cd163")) > 0
    print(f"slice {s}: Lyve1+ {lyve.sum()}, of which no BAM marker "
          f"{int((lyve & ~bam).sum())} ({100*(lyve & ~bam).sum()/lyve.sum():.1f}%)")
    return x, y, lyve, bam


data = {s: load(s) for s in SLICES}
labels = ["Slice {} ({})".format(s, "C" if s in CONTROL else "T") for s in SLICES]

fig = plt.figure(figsize=(12.8, 15.2), dpi=200)
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.85, 1, 1, 1],
              hspace=0.30, wspace=0.04)

# ---- (a) composition of the Lyve1-positive population --------------------------
axA = fig.add_subplot(gs[0, :])
x = np.arange(len(SLICES))
no_bam = np.array([100 * (d[2] & ~d[3]).sum() / d[2].sum() for d in data.values()])
axA.bar(x, no_bam, color=RED, width=0.55, label="no canonical BAM marker")
axA.bar(x, 100 - no_bam, bottom=no_bam, color=BLUE, width=0.55,
        label="Mrc1 or Cd163 also detected")
for xi, val in zip(x, no_bam):
    axA.text(xi, val / 2, "{:.0f}%".format(val), ha="center", va="center",
             fontsize=11, color="white", fontweight="bold")
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=10)
axA.set_ylabel("Lyve1-positive cells (%)", fontsize=10.5)
axA.set_ylim(0, 118)
axA.set_yticks([0, 25, 50, 75, 100])
axA.legend(fontsize=10, frameon=False, loc="upper center", ncol=2)
axA.spines[["top", "right"]].set_visible(False)
axA.text(-0.045, 1.12, "(a)", transform=axA.transAxes, fontsize=15,
         fontweight="bold", va="top", ha="right")

# ---- (b) the same split in space -----------------------------------------------
for i, s in enumerate(SLICES):
    ax = fig.add_subplot(gs[1 + i // 2, i % 2])
    xx, yy, lyve, bam = data[s]
    ax.scatter(xx, yy, s=0.2, c=GREY, linewidths=0, rasterized=True)
    ax.scatter(xx[lyve & ~bam], yy[lyve & ~bam], s=0.5, c=RED,
               linewidths=0, rasterized=True)
    ax.scatter(xx[lyve & bam], yy[lyve & bam], s=0.9, c=BLUE,
               linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(labels[i], fontsize=11, pad=4)
    if i == 0:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.plot([x1 - 1.4, x1 - 0.4], [y0 + 0.25, y0 + 0.25], "-", color="#333", lw=2.4)
        ax.text(x1 - 0.9, y0 + 0.38, "1 mm", ha="center", va="bottom", fontsize=9)
        first_map = ax

# (b) is placed from the axes position so it cannot land on panel (a)'s labels
pos = first_map.get_position()
fig.text(0.055, pos.y1 + 0.012, "(b)", fontsize=15, fontweight="bold",
         va="bottom", ha="right")

fig.legend(handles=[Patch(color=RED, label="Lyve1+, no canonical BAM marker"),
                    Patch(color=BLUE, label="Lyve1+ and Mrc1/Cd163+"),
                    Patch(color=GREY, label="all other cells")],
           fontsize=10.5, frameon=False, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.055))

fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
