"""Figure DQ2a -- tdTomato has no lineage-restricted association profile.

For each probe (rows) the correlation with a panel of lineage markers (columns,
grouped by cell class) is computed over non-tumor cells of all six slices and
averaged. Correlations are partial on log total counts, because detection of any
two sparse targets co-varies with how many molecules were imaged in a cell; the
depth-driven component is shared by all probes and would otherwise dominate.

(a) heatmap of the profiles; (b) myeloid preference, the mean correlation over
myeloid and BAM markers minus the mean over all other markers.

A lineage-restricted probe shows a block of positive values over its own class.
tdTomato, though the best-detected custom probe, is flat.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_tdtomato_profile.py
"""
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT = ROOT + "/thesis_plots/dq_fig_tdtomato_profile.png"
SLICES = [1, 2, 3, 4, 5, 6]
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}

PROBES = ["tdTomato", "GFP", "TMEM119", "Trem2", "Lyve1", "GFAP"]
GROUPS = [("Myeloid", ["Cx3cr1", "Csf1r", "Aif1", "Itgam", "Ptprc"]),
          ("BAM", ["Mrc1", "Cd163", "Pf4"]),
          ("Vascular", ["Pecam1", "Flt1", "Vtn", "Rgs5"]),
          ("Astro / glia", ["S100b", "Pdgfra"]),
          ("Neuronal", ["Meg3"])]
MYELOID = set(GROUPS[0][1]) | set(GROUPS[1][1])
MARKERS = [g for _, gs in GROUPS for g in gs]


def vec(a, name, nt):
    k = [v for v in a.var_names if v.lower() == name.lower()]
    if not k:
        return None
    x = a[:, k[0]].X
    return ((x.toarray().ravel() if issparse(x) else np.asarray(x).ravel())[nt]).astype(float)


def partial(x, y, z):
    """Pearson r of x and y with z partialled out."""
    def r(u, v):
        u = u - u.mean(); v = v - v.mean()
        d = np.sqrt((u * u).sum() * (v * v).sum())
        return float((u * v).sum() / d) if d > 0 else np.nan
    rxy, rxz, ryz = r(x, y), r(x, z), r(y, z)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 0 else np.nan


acc = {p: {m: [] for m in MARKERS} for p in PROBES}
for s in SLICES:
    awn = ad.read_h5ad(WN.format(s))
    at = ad.read_h5ad(WTP.format(s), backed="r").obs
    tv = at["pred_tumor_XGBoost"]
    if set(map(str, awn.obs_names)) == set(map(str, at.index)):
        tv = tv.reindex(awn.obs_names)
    nt = ~tv.astype(object).isin(TRUE).to_numpy()
    real = [v for v in awn.var_names
            if not v.lower().startswith(("negative", "systemcontrol"))]
    z = np.log1p(np.asarray(awn[:, real].X.sum(axis=1)).ravel()[nt])
    mv = {m: vec(awn, m, nt) for m in MARKERS}
    for p in PROBES:
        x = vec(awn, p, nt)
        if x is None:
            continue
        for m in MARKERS:
            if mv[m] is not None:
                acc[p][m].append(partial(x, mv[m], z))
    print(f"slice {s} done")

M = np.array([[np.nanmean(acc[p][m]) if acc[p][m] else np.nan for m in MARKERS]
              for p in PROBES])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 4.3), dpi=200,
                               gridspec_kw={"width_ratios": [2.5, 1]})

lim = np.nanmax(np.abs(M))
im = axA.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
axA.set_xticks(range(len(MARKERS)))
axA.set_xticklabels(MARKERS, rotation=45, ha="right", fontsize=8.5)
axA.set_yticks(range(len(PROBES)))
axA.set_yticklabels(PROBES, fontsize=9)
for i in range(len(PROBES)):
    for j in range(len(MARKERS)):
        if not np.isnan(M[i, j]):
            axA.text(j, i, "{:.2f}".format(M[i, j]), ha="center", va="center",
                     fontsize=6.6, color="#111" if abs(M[i, j]) < lim * 0.6 else "#fff")
# group separators and labels
edge = 0
for name, gs in GROUPS[:-1]:
    edge += len(gs)
    axA.axvline(edge - 0.5, color="#000", lw=1.1)
edge = 0
for name, gs in GROUPS:
    axA.text(edge + (len(gs) - 1) / 2, -0.85, name, ha="center", va="bottom",
             fontsize=8.5, fontweight="bold")
    edge += len(gs)
axA.set_title("(a) Association with lineage markers", fontsize=11,
              fontweight="bold", pad=24)
fig.colorbar(im, ax=axA, shrink=0.8, label="partial r (depth-controlled)")

mye = [j for j, m in enumerate(MARKERS) if m in MYELOID]
oth = [j for j, m in enumerate(MARKERS) if m not in MYELOID]
pref = np.nanmean(M[:, mye], axis=1) - np.nanmean(M[:, oth], axis=1)
order = np.argsort(pref)
cols = ["#D62728" if PROBES[i] in ("tdTomato", "GFP") else "#9A9A9A" for i in order]
axB.barh(range(len(PROBES)), pref[order], color=cols)
axB.set_yticks(range(len(PROBES)))
axB.set_yticklabels([PROBES[i] for i in order], fontsize=9)
axB.axvline(0, color="#000", lw=1)
axB.set_xlabel("myeloid preference (mean r, myeloid - other)")
axB.set_title("(b) Lineage preference", fontsize=11, fontweight="bold", pad=24)
axB.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(pref[order]):
    axB.text(v + (0.002 if v >= 0 else -0.002), i, "{:+.3f}".format(v),
             va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
axB.set_xlim(min(0, pref.min()) - 0.02, max(pref.max() * 1.35, 0.02))

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
