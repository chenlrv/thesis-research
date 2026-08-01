"""Two figures for the Data-Quality section, all six slices, non-tumor cells:

  detection_barplot_all6.png  -- (a) S/N (log), (b) specificity. bar=mean, whisker=min-max.
                                 Colors: custom=blue, panel reference=grey, GFP=red.
  tdtomato_biospecificity_all6.png -- tdTomato's Pearson correlation with lineage
                                 markers (unconditioned, non-tumor). Its top correlate is
                                 neuronal Meg3, not any myeloid gene -> biological non-specificity.

Run: conda run -n thesis_research python thesis_plots/make_reporter_detection_figs.py
"""
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
SLICES = [1, 2, 3, 4, 5, 6]
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}

ORDER = ["Ccl2", "Cxcl13", "GFAP", "Lyve1", "TMEM119", "Trem2", "GFP", "tdTomato",
         "Cx3cr1", "Pecam1", "Meg3"]
PANEL = {"Cx3cr1", "Pecam1", "Meg3"}
BLUE, GREY, RED = "#0072B2", "#9A9A9A", "#c1121f"

# tdTomato correlation markers, grouped by expected lineage
MARKERS = [("Meg3", "neuronal"), ("Cx3cr1", "myeloid"), ("Csf1r", "myeloid"),
           ("Aif1", "myeloid"), ("C1qa", "myeloid"), ("Ccr2", "MDM"), ("Plac8", "MDM"),
           ("Mrc1", "BAM"), ("Cd163", "BAM"), ("Pecam1", "vascular"), ("S100b", "astrocyte")]


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


def pearson(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    sx, sy = np.sqrt((xm * xm).sum()), np.sqrt((ym * ym).sum())
    return float((xm * ym).sum() / (sx * sy)) if sx > 0 and sy > 0 else np.nan


sn = {p: [] for p in ORDER}
sp = {p: [] for p in ORDER}
cors = {m: [] for m, _ in MARKERS}
for s in SLICES:
    awn = ad.read_h5ad(WN.format(s))
    at = ad.read_h5ad(WTP.format(s))
    nt = nt_mask(awn, at)
    neg = [v for v in awn.var_names if v.lower().startswith("negative")]
    negX = awn[:, neg].X
    negX = negX.toarray() if issparse(negX) else np.asarray(negX)
    bg = float(negX[nt].mean())
    for p in ORDER:
        x = counts(awn, p)
        if x is None:
            continue
        xn = x[nt]
        m = float(xn.mean())
        sn[p].append(m / bg)
        sp[p].append(np.clip((m - bg) / m, 0, 1) if m > 0 else 0.0)
    tdt = counts(awn, "tdTomato")[nt]
    for mk, _ in MARKERS:
        mc = counts(awn, mk)
        cors[mk].append(pearson(tdt, mc[nt]) if mc is not None else np.nan)
    print(f"slice {s} done")
    del awn, at


def agg(d, k):
    v = np.array(d[k], float)
    return np.nanmean(v), np.nanmean(v) - np.nanmin(v), np.nanmax(v) - np.nanmean(v)


# ============================ FIGURE 1: detection barplot (2 panels) ============================
def bar_color(p):
    return RED if p == "GFP" else (GREY if p in PANEL else BLUE)


cols = [bar_color(p) for p in ORDER]
x = np.arange(len(ORDER))
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)

m = [agg(sn, p)[0] for p in ORDER]
e = np.array([[agg(sn, p)[1] for p in ORDER], [agg(sn, p)[2] for p in ORDER]])
axA.bar(x, m, color=cols, yerr=e, capsize=3, ecolor="#444", error_kw={"lw": 1})
axA.set_yscale("log"); axA.set_ylim(0.7, 170)
axA.axhline(1, ls="--", lw=1, color="#000")
axA.text(len(ORDER) - 0.5, 1.03, "background (S/N = 1)", ha="right", va="bottom", fontsize=8)
axA.set_ylabel("Signal-to-background (S/N)")
axA.set_title("(a) Detection strength", fontsize=11, fontweight="bold")
for xi, p in zip(x, ORDER):
    mm, _, hi = agg(sn, p)
    axA.text(xi, (mm + hi) * 1.05, f"{mm:.1f}", ha="center", va="bottom", fontsize=7.5)
axA.set_xticks(x); axA.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8.5)
axA.legend(handles=[Patch(color=BLUE, label="custom probe"),
                    Patch(color=GREY, label="panel reference"),
                    Patch(color=RED, label="GFP (failed)")],
           fontsize=8, frameon=False, loc="upper left")

m = [agg(sp, p)[0] for p in ORDER]
e = np.array([[agg(sp, p)[1] for p in ORDER], [agg(sp, p)[2] for p in ORDER]])
axB.bar(x, m, color=cols, yerr=e, capsize=3, ecolor="#444", error_kw={"lw": 1})
axB.axhline(0.5, ls=":", lw=1.2, color="#666")
axB.set_ylim(0, 1.08)
axB.set_ylabel("Fraction of signal above background")
axB.set_title("(b) Detection specificity", fontsize=11, fontweight="bold")
for xi, p in zip(x, ORDER):
    mm, _, hi = agg(sp, p)
    axB.text(xi, mm + hi + 0.02, f"{mm:.2f}", ha="center", va="bottom", fontsize=7.5)
axB.set_xticks(x); axB.set_xticklabels(ORDER, rotation=45, ha="right", fontsize=8.5)

fig.suptitle("Custom-probe detection reliability across all six slices "
             "(bar = mean, whisker = min–max; non-tumor cells)",
             fontsize=12.5, fontweight="bold", y=1.0)
fig.tight_layout()
fig.savefig(ROOT + "/thesis_plots/detection_barplot_all6.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ============================ FIGURE 2: tdTomato biological specificity ============================
grp_color = {"neuronal": RED, "myeloid": BLUE, "MDM": BLUE, "BAM": BLUE,
             "vascular": GREY, "astrocyte": GREY}
order2 = sorted(MARKERS, key=lambda mk: agg(cors, mk[0])[0])   # ascending -> top strongest
labels = [f"{mk}" + ("  (neuronal)" if grp == "neuronal" else
                      "  (myeloid)" if grp in ("myeloid", "MDM", "BAM") else "") for mk, grp in order2]
means = [agg(cors, mk)[0] for mk, _ in order2]
errs = np.array([[agg(cors, mk)[1] for mk, _ in order2], [agg(cors, mk)[2] for mk, _ in order2]])
colc = [grp_color[grp] for _, grp in order2]
y = np.arange(len(order2))

fig2, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
ax.barh(y, means, xerr=errs, color=colc, capsize=3, ecolor="#555", error_kw={"lw": 1})
ax.axvline(0, color="#000", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Pearson correlation of tdTomato with marker (non-tumor cells)")
ax.set_title("tdTomato's strongest correlate is neuronal (Meg3), not myeloid\n"
             "— real signal, wrong cells (biological non-specificity)", fontsize=11)
ax.legend(handles=[Patch(color=RED, label="neuronal marker"),
                   Patch(color=BLUE, label="myeloid markers (expected)"),
                   Patch(color=GREY, label="other")],
          fontsize=8.5, frameon=False, loc="lower right")
ax.grid(axis="x", color="#eee")
fig2.tight_layout()
fig2.savefig(ROOT + "/thesis_plots/tdtomato_biospecificity_all6.png", bbox_inches="tight", dpi=150)
plt.close(fig2)

# ---- print numbers for the corrected description ----
print("\n=== tdTomato Pearson r with markers (mean [min-max] across slices) ===")
for mk, grp in sorted(MARKERS, key=lambda mk: -agg(cors, mk[0])[0]):
    v = np.array(cors[mk], float)
    print(f"  {mk:8s} ({grp:9s}) r = {np.nanmean(v):+.3f} [{np.nanmin(v):+.3f}, {np.nanmax(v):+.3f}]")
print("\nSaved both figures.")
