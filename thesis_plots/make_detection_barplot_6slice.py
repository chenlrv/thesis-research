"""Detection reliability in the slice-1 bar-chart style, but summarising ALL six
slices: each bar = mean across slices, whisker = min-max range.
(a) S/N (log), (b) specificity, (c) GFP positivity vs cell-depth decile per slice.

Run: conda run -n thesis_research python thesis_plots/make_detection_barplot_6slice.py
"""
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT = ROOT + "/thesis_plots/detection_barplot_all6.png"
SLICES = [1, 2, 3, 4, 5, 6]
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}
ORDER = ["Ccl2", "Cxcl13", "GFAP", "Lyve1", "TMEM119", "Trem2", "GFP", "tdTomato",
         "Cx3cr1", "Pecam1", "Meg3", "Csf1r"]


def role_color(p):
    if p == "GFAP":
        return "#009E73"      # positive control
    if p == "GFP":
        return "#D55E00"      # failed
    if p == "tdTomato":
        return "#E69F00"      # reporter
    if p in ("Cx3cr1", "Pecam1", "Meg3", "Csf1r"):
        return "#9A9A9A"      # panel reference
    return "#0072B2"          # other custom


def counts(a, gene):
    panel = {g.lower(): g for g in a.var_names}
    k = panel.get(gene.lower())
    if k is None:
        return None
    x = a[:, k].X
    return (x.toarray().ravel() if issparse(x) else np.asarray(x).ravel()).astype(float)


def nt_mask(awn, at):
    tv = at.obs["pred_tumor_XGBoost"]
    if set(map(str, awn.obs_names)) == set(map(str, at.obs_names)):
        tv = tv.reindex(awn.obs_names)
    return ~tv.astype(object).isin(TRUE).to_numpy()


sn = {p: [] for p in ORDER}
sp = {p: [] for p in ORDER}
gfp_depth = {}
for s in SLICES:
    awn = ad.read_h5ad(WN.format(s))
    at = ad.read_h5ad(WTP.format(s))
    nt = nt_mask(awn, at)
    neg = [v for v in awn.var_names if v.lower().startswith("negative")]
    negX = awn[:, neg].X
    negX = negX.toarray() if issparse(negX) else np.asarray(negX)
    bg = float(negX[nt].mean())
    real = [v for v in awn.var_names if not v.lower().startswith(("negative", "systemcontrol"))]
    totals = np.asarray(awn[:, real].X.sum(axis=1)).ravel()[nt]
    for p in ORDER:
        x = counts(awn, p)
        if x is None:
            continue
        xn = x[nt]
        m = float(xn.mean())
        sn[p].append(m / bg)
        sp[p].append(np.clip((m - bg) / m, 0, 1) if m > 0 else 0.0)
    gfp = counts(awn, "GFP")[nt] > 0
    edges = np.quantile(totals, np.linspace(0, 1, 11))
    gfp_depth[s] = [100 * gfp[(totals >= edges[i]) & (totals <= edges[i + 1] if i == 9
                    else totals < edges[i + 1])].mean() for i in range(10)]
    print(f"slice {s} done (bg={bg:.4f})")
    del awn, at


def agg(d, p):
    v = np.array(d[p])
    return v.mean(), v.mean() - v.min(), v.max() - v.mean()


cols = [role_color(p) for p in ORDER]
x = np.arange(len(ORDER))
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(17, 5.4), dpi=150,
                                    gridspec_kw={"width_ratios": [1.15, 1.15, 1]})

# (a) S/N, log
means = [agg(sn, p)[0] for p in ORDER]
err = np.array([[agg(sn, p)[1] for p in ORDER], [agg(sn, p)[2] for p in ORDER]])
axA.bar(x, means, color=cols, yerr=err, capsize=3, ecolor="#444", error_kw={"lw": 1})
axA.set_yscale("log")
axA.set_ylim(0.7, 170)
axA.axhline(1, ls="--", lw=1, color="#000")
axA.text(len(ORDER) - 0.5, 1.03, "background (S/N = 1)", ha="right", va="bottom", fontsize=8)
axA.set_ylabel("Signal-to-background (S/N)")
axA.set_title("(a) Detection strength", fontsize=11, fontweight="bold")
for xi, m in zip(x, means):
    axA.text(xi, m * (1 + agg(sn, ORDER[xi])[2] / m) * 1.06, f"{m:.1f}",
             ha="center", va="bottom", fontsize=7.5)
axA.set_xticks(x); axA.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8.5)

# (b) specificity
means = [agg(sp, p)[0] for p in ORDER]
err = np.array([[agg(sp, p)[1] for p in ORDER], [agg(sp, p)[2] for p in ORDER]])
axB.bar(x, means, color=cols, yerr=err, capsize=3, ecolor="#444", error_kw={"lw": 1})
axB.axhline(0.5, ls=":", lw=1.2, color="#666")
axB.set_ylim(0, 1.08)
axB.set_ylabel("Fraction of signal above background")
axB.set_title("(b) Detection specificity", fontsize=11, fontweight="bold")
for xi, p in zip(x, ORDER):
    m, _, hi = agg(sp, p)
    axB.text(xi, m + hi + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5)
axB.set_xticks(x); axB.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8.5)

# (c) GFP-depth per slice
cmap = plt.get_cmap("viridis")
for k, s in enumerate(SLICES):
    axC.plot(range(1, 11), gfp_depth[s], "-o", ms=4, lw=1.6,
             color=cmap(k / (len(SLICES) - 1)), label=f"Slice {s}")
axC.set_xlabel("Total-count decile (low → high)")
axC.set_ylabel("% cells GFP-positive")
axC.set_title("(c) GFP tracks cell depth (ambient signature)", fontsize=11, fontweight="bold")
axC.set_xticks(range(1, 11)); axC.grid(axis="y", color="#eee")
axC.legend(fontsize=8, frameon=False, ncol=2)

# role legend on (a)
from matplotlib.patches import Patch
axA.legend(handles=[Patch(color="#009E73", label="GFAP (pos. control)"),
                    Patch(color="#E69F00", label="tdTomato"),
                    Patch(color="#D55E00", label="GFP (failed)"),
                    Patch(color="#0072B2", label="other custom"),
                    Patch(color="#9A9A9A", label="panel reference")],
           fontsize=7.5, frameon=False, loc="upper left")

fig.suptitle("Custom-probe detection reliability across all six slices "
             "(bar = mean, whisker = min–max across slices; non-tumor cells)",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=150)
plt.close(fig)
print("Saved:", OUT)
