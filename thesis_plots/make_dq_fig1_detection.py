"""Figure DQ1 -- probe detection against the acceptance bar, all six slices.

A probe is accepted where its mean per-cell count exceeds that of the most
background-prone negative-control probe in the same slice. That threshold differs
between slices (3.0-4.0 times the slice's mean noise floor), so the shaded band
shows its range while each slice-point is tested against its own threshold:
points failing their own slice's threshold are drawn as red crosses.

Bars are the mean of the six per-slice ratios, dots the individual slices;
non-tumor cells only. Probe order follows Table 2 (descending S/N).

Per-slice acceptance bars are computed once from the negative-control probes and
cached to acceptance_bar_all6.csv; the per-slice probe means come from
detection_reliability_all6.csv, the source of Table 2.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig1_detection.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "D:/thesis-research"
SRC = ROOT + "/thesis_plots/detection_reliability_all6.csv"
BAR = ROOT + "/thesis_plots/acceptance_bar_all6.csv"
OUT = ROOT + "/thesis_plots/dq_fig1_detection.png"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
SLICES = [1, 2, 3, 4, 5, 6]
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}
CUSTOM, PANEL, FAILED = "#0072B2", "#9A9A9A", "#D62728"


def compute_bars():
    """Most background-prone negative probe per slice, in units of the mean floor."""
    import anndata as ad
    from scipy.sparse import issparse
    rows = []
    for s in SLICES:
        at = ad.read_h5ad(WTP.format(s), backed="r")
        nt = ~at.obs["pred_tumor_XGBoost"].astype(object).isin(TRUE).to_numpy()
        at.file.close()
        awn = ad.read_h5ad(WN.format(s), backed="r")
        neg = [v for v in awn.var_names if str(v).lower().startswith("negative")]
        X = awn[:, neg].to_memory().X
        X = (X.toarray() if issparse(X) else np.asarray(X))[nt]
        pm = X.mean(0)
        rows.append(dict(slice=s, mean_floor=float(pm.mean()),
                         worst_neg=float(pm.max()), bar_x_floor=float(pm.max() / pm.mean())))
        awn.file.close()
        print(f"slice {s}: bar = {pm.max() / pm.mean():.2f} x mean floor")
    out = pd.DataFrame(rows)
    out.to_csv(BAR, index=False)
    return out


d = pd.read_csv(SRC)
bars = pd.read_csv(BAR) if os.path.exists(BAR) else compute_bars()
bar = dict(zip(bars["slice"], bars["bar_x_floor"]))

# S/N is the count in units of the slice's mean floor; dividing by the slice's bar
# (also in those units) gives the count in units of that slice's acceptance bar.
d["rel"] = d.apply(lambda r: r["SN"] / bar[r["slice"]], axis=1)

order = (d.groupby(["probe", "cls"])["SN"].mean().reset_index()
           .sort_values(["cls", "SN"], ascending=[True, False]))
probes = list(order.probe)
cols = [FAILED if p == "GFP" else (CUSTOM if c == "custom" else PANEL)
        for p, c in zip(order.probe, order.cls)]
x = np.arange(len(probes))

fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=200)
m = np.array([d.loc[d.probe == p, "SN"].mean() for p in probes])
ax.bar(x, m, color=cols, width=0.72, zorder=2)
for xi, p in zip(x, probes):
    sub = d.loc[d.probe == p]
    passed = sub["SN"].to_numpy() > sub["slice"].map(bar).to_numpy()
    top = sub["SN"].max()
    ax.plot(np.full(passed.sum(), xi), sub["SN"].to_numpy()[passed], "o", ms=3.4,
            mfc="none", mec="#222", mew=0.8, zorder=3)
    ax.plot(np.full((~passed).sum(), xi), sub["SN"].to_numpy()[~passed], "x", ms=5.5,
            color="#B00", mew=1.6, zorder=4)
    ax.text(xi, top * 1.14, "{:.1f}".format(m[list(probes).index(p)]),
            ha="center", va="bottom", fontsize=7.5)

ax.set_yscale("log")
ax.set_ylim(0.7, 400)
ax.axhline(1, ls="--", lw=1.1, color="#000", zorder=1)
ax.axhspan(3.02, 4.03, color="#B00", alpha=0.13, lw=0, zorder=0)
ax.set_ylabel("Signal-to-background (S/N)")
ax.set_xticks(x)
ax.set_xticklabels(probes, rotation=45, ha="right", fontsize=8.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Patch(color=CUSTOM, label="custom add-on probe"),
                   Patch(color=PANEL, label="panel reference gene"),
                   Patch(color=FAILED, label="GFP (failed)"),
                   Line2D([], [], ls="none", marker="o", ms=3.4, mfc="none",
                          mec="#222", label="individual slices"),
                   Line2D([], [], ls="none", marker="x", ms=5.5, color="#B00",
                          mew=1.6, label="slice not exceeding its threshold"),
                   Line2D([], [], ls="--", lw=1.1, color="#000",
                          label="noise floor (S/N = 1)"),
                   Patch(color="#B00", alpha=0.13,
                         label="range of slice-specific acceptance thresholds")],
          fontsize=8, frameon=False, loc="upper left", handlelength=1.9,
          labelspacing=0.32)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
