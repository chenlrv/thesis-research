"""Spatial distribution of the tumor-cell calls in each section.

Shows that the calls form contiguous lesions rather than scattered individual
cells -- a coherence check that does not depend on the expression measurement
being correct, only on where the called cells lie.

Note on the control sections: the Stage-2 classifier's negative class was defined
as the SingleR-called tumor cells of the sham-injected sections, so the near-
absence of calls there is partly by construction and is not independent evidence.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_tumor_spatial.py
"""
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "D:/thesis-research"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT = ROOT + "/thesis_plots/dq_fig_tumor_spatial.png"
SLICES = [1, 2, 3, 4, 5, 6]
CONTROL = {3, 4}
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}
PX_UM = 0.12028
GREY, TUM = "#DCDEE0", "#111111"

data = {}
for s in SLICES:
    o = ad.read_h5ad(WTP.format(s), backed="r").obs
    t = o["pred_tumor_XGBoost"].astype(object).isin(TRUE).to_numpy()
    x = o["CenterX_global_px"].to_numpy() * PX_UM / 1000.0
    y = o["CenterY_global_px"].to_numpy() * PX_UM / 1000.0
    data[s] = (x, y, t)
    print(f"slice {s}: {t.sum():,} tumor of {len(t):,} cells ({100*t.mean():.1f}%)")

# common data span for every panel, so all sections share one scale and one
# axes shape (the scale bar below is then valid for the whole figure)
PAD = 0.35  # mm
span_x = max(x.max() - x.min() for x, y, t in data.values()) + 2 * PAD
span_y = max(y.max() - y.min() for x, y, t in data.values()) + 2 * PAD

FIG_W = 9.6
panel_w = FIG_W / 2 - 0.25                       # minus inter-panel padding
fig_h = 3 * (panel_w * span_y / span_x + 0.34) + 0.5   # + title strip, + legend
fig, axes = plt.subplots(3, 2, figsize=(FIG_W, fig_h), dpi=200)
for ax, s in zip(axes.ravel(), SLICES):
    x, y, t = data[s]
    ax.scatter(x, y, s=0.14, c=GREY, linewidths=0, rasterized=True)
    ax.scatter(x[t], y[t], s=0.5, c=TUM, linewidths=0, rasterized=True)
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
    ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Slice {} ({}) — {:,} tumor cells ({:.1f}%)".format(
        s, "control" if s in CONTROL else "tumor-bearing", int(t.sum()), 100 * t.mean()),
        fontsize=9.5, pad=4)

ax0 = axes[0, 0]
x0, x1 = ax0.get_xlim(); y0, y1 = ax0.get_ylim()
ax0.plot([x1 - 1.5, x1 - 0.5], [y0 + 0.35, y0 + 0.35], "-", color="#333", lw=2.2)
ax0.text(x1 - 1.0, y0 + 0.48, "1 mm", ha="center", va="bottom", fontsize=8)

fig.legend(handles=[Patch(color=TUM, label="cell called tumor"),
                    Patch(color=GREY, label="all other cells")],
           fontsize=9.5, frameon=False, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(h_pad=0.6, w_pad=0.2, rect=(0, 0.03, 1, 1))
fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
