"""Figure DQ3 -- Lyve1 positivity is not anatomically organised.

(a) Lyve1-positive cells, and (b) cells positive for a canonical BAM marker
(Mrc1 or Cd163), in each of the six sections. Border-associated macrophages
reside at the meningeal surface and along the vasculature, so a marker of that
population should trace the tissue border and vessel tracks rather than fill the
parenchyma uniformly.

All cells of each section are shown in grey; positive cells are drawn on top.
Positivity is raw detection (>= 1 count), so no cell-type or tumor assignment
enters the figure. Coordinates are CenterX/Y_global_px, the tissue-level frame
(obsm["spatial"] in these files is FOV-local and must not be used here).

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_lyve1_spatial.py
"""
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
OUT = ROOT + "/thesis_plots/dq_fig_lyve1_spatial.png"
SLICES = [1, 2, 3, 4, 5, 6]
CONTROL = {3, 4}
PX_UM = 0.12028
GREY, RED, BLUE = "#DCDEE0", "#D62728", "#0072B2"


def load(s):
    a = ad.read_h5ad(WN.format(s))
    v = a.var_names.astype(str)

    def cnt(n):
        k = [x for x in v if x.lower() == n.lower()]
        if not k:
            return np.zeros(a.n_obs)
        y = a[:, k[0]].X
        return (y.toarray().ravel() if issparse(y) else np.asarray(y).ravel()).astype(float)

    x = a.obs["CenterX_global_px"].to_numpy() * PX_UM / 1000.0   # mm
    y = a.obs["CenterY_global_px"].to_numpy() * PX_UM / 1000.0
    return x, y, cnt("Lyve1") > 0, (cnt("Mrc1") + cnt("Cd163")) > 0


data = {s: load(s) for s in SLICES}
print("loaded")

fig, axes = plt.subplots(4, 3, figsize=(13.4, 10.4), dpi=200)
for block, (row0, colour, label) in enumerate(
        ((0, RED, "Lyve1"), (2, BLUE, "Mrc1 or Cd163"))):
    for i, s in enumerate(SLICES):
        ax = axes[row0 + i // 3, i % 3]
        x, y, lyve, bam = data[s]
        m = lyve if block == 0 else bam
        ax.scatter(x, y, s=0.12, c=GREY, linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=0.28, c=colour, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title("Slice {} ({}) — {} {}+  ({:.1f}%)".format(
            s, "control" if s in CONTROL else "tumor", label,
            "" if block else "", 100 * m.mean()).replace("+ ", "+ "),
            fontsize=8.5, pad=3)

# scale bar on the first panel of each block, 1 mm
for row0 in (0, 2):
    ax = axes[row0, 0]
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.plot([x1 - 1.35, x1 - 0.35], [y0 - 0.25, y0 - 0.25], "-", color="#333", lw=2.2)
    ax.text(x1 - 0.85, y0 - 0.42, "1 mm", ha="center", va="bottom", fontsize=7.5)

for row0, lab in ((0, "a"), (2, "b")):
    axes[row0, 0].text(-0.03, 1.22, "(" + lab + ")", transform=axes[row0, 0].transAxes,
                       fontsize=13, fontweight="bold", va="top", ha="right")

fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
