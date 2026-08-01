"""Probe detection-reliability across ALL six slices (non-tumor cells).

For each slice and probe:
  S/N               = mean per-cell probe count / mean per-cell negative-probe count
  % cells positive  = fraction of cells with >= 1 count
  % signal above bg = (mean_signal - background) / mean_signal   (clipped to [0,1])
Also: GFP positivity by total-count decile per slice (the ambient signature).

Counts + negative probes come from *_with_neg.h5ad; the non-tumor mask from the
tumor-prediction cache (pred_tumor_XGBoost), aligned by cell id / row order.

Run: conda run -n thesis_research python thesis_plots/make_detection_reliability_6slice.py
"""
import os
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import issparse

ROOT = "D:/thesis-research"
WN = ROOT + "/resources/cache/slice_{}_adata_with_neg.h5ad"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT_FIG = ROOT + "/thesis_plots/detection_reliability_all6.png"
OUT_CSV = ROOT + "/thesis_plots/detection_reliability_all6.csv"
SLICES = [1, 2, 3, 4, 5, 6]
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}
PROBES = [("GFAP", "custom"), ("tdTomato", "custom"), ("Ccl2", "custom"),
          ("TMEM119", "custom"), ("Lyve1", "custom"), ("Trem2", "custom"),
          ("Cxcl13", "custom"), ("GFP", "custom"),
          ("Meg3", "panel"), ("Pecam1", "panel"), ("Cx3cr1", "panel"), ("Csf1r", "panel")]


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
    elif awn.n_obs != at.n_obs:
        raise RuntimeError("with_neg and WTP cell counts differ and ids don't match")
    tumor = tv.astype(object).isin(TRUE).to_numpy()
    return ~tumor


def main():
    rows, gfp_depth = [], {}
    for s in SLICES:
        awn = ad.read_h5ad(WN.format(s))
        at = ad.read_h5ad(WTP.format(s))
        nt = nt_mask(awn, at)

        neg = [v for v in awn.var_names if v.lower().startswith("negative")]
        negX = awn[:, neg].X
        negX = negX.toarray() if issparse(negX) else np.asarray(negX)
        bg = float(negX[nt].mean())

        real = [v for v in awn.var_names
                if not v.lower().startswith(("negative", "systemcontrol"))]
        totals = np.asarray(awn[:, real].X.sum(axis=1)).ravel()[nt]

        for g, cls in PROBES:
            x = counts(awn, g)
            if x is None:
                continue
            xn = x[nt]
            mean = float(xn.mean())
            sn = mean / bg
            pct_pos = float((xn > 0).mean() * 100)
            above = float(np.clip((mean - bg) / mean, 0, 1) * 100) if mean > 0 else 0.0
            rows.append(dict(slice=s, probe=g, cls=cls, bg=bg, mean=mean,
                             SN=sn, pct_pos=pct_pos, above_bg=above))

        # GFP positivity by total-count decile
        gfp = counts(awn, "GFP")[nt] > 0
        edges = np.quantile(totals, np.linspace(0, 1, 11))
        ys = []
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            md = (totals >= lo) & (totals <= hi if i == 9 else totals < hi)
            ys.append(100 * gfp[md].mean() if md.sum() else np.nan)
        gfp_depth[s] = ys
        print(f"slice {s}: bg={bg:.4f}  GFP S/N={rows[-5]['SN']:.2f} "
              f"(pos {[r for r in rows if r['slice']==s and r['probe']=='GFP'][0]['pct_pos']:.1f}%)")
        del awn, at

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ---- print per-probe S/N across slices (mean & range) ----
    print("\n=== S/N by probe across slices (mean [min-max]) ===")
    for g, _ in PROBES:
        sub = df[df.probe == g]
        print(f"  {g:9s} S/N {sub.SN.mean():6.2f} [{sub.SN.min():.2f}-{sub.SN.max():.2f}]  "
              f"above-bg {sub.above_bg.mean():4.0f}% [{sub.above_bg.min():.0f}-{sub.above_bg.max():.0f}]")

    # ============ FIGURE (3 panels: S/N heatmap | specificity heatmap | GFP-depth) ============
    probe_order = [g for g, _ in PROBES]
    SN = df.pivot(index="probe", columns="slice", values="SN").reindex(probe_order)
    SP = (df.pivot(index="probe", columns="slice", values="above_bg") / 100).reindex(probe_order)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(20, 6.5), dpi=150,
                                        gridspec_kw={"width_ratios": [1.05, 1.05, 1.1]})

    def heat(ax, data, norm, title, cbar_label, fmt, show_y):
        im = ax.imshow(data.values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(SLICES)))
        ax.set_xticklabels([f"S{s}" for s in SLICES])
        ax.set_yticks(range(len(probe_order)))
        ax.set_yticklabels(probe_order if show_y else [""] * len(probe_order))
        for i in range(len(probe_order)):
            for j in range(len(SLICES)):
                ax.text(j, i, fmt.format(data.values[i, j]), ha="center", va="center",
                        fontsize=7.2, color="black")
        ax.set_title(title, fontsize=11)
        gi = probe_order.index("GFP")
        ax.add_patch(plt.Rectangle((-0.5, gi - 0.5), len(SLICES), 1, fill=False,
                                   edgecolor="#c1121f", lw=2.2))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=8)

    heat(axA, SN, LogNorm(vmin=1, vmax=100),
         "(a) Signal-to-background (S/N)", "S/N (log) — 1 = background floor", "{:.1f}", True)
    heat(axB, SP, mpl.colors.Normalize(vmin=0, vmax=1),
         "(b) Detection specificity", "fraction above bg — 0.5 = half noise", "{:.2f}", False)

    # (c) GFP positivity vs total-count decile, one line per slice
    cmap = plt.get_cmap("viridis")
    for k, s in enumerate(SLICES):
        axC.plot(range(1, 11), gfp_depth[s], "-o", ms=4, lw=1.6,
                 color=cmap(k / (len(SLICES) - 1)), label=f"Slice {s}")
    axC.set_xlabel("Total-count decile (low → high)")
    axC.set_ylabel("% cells GFP-positive")
    axC.set_title("(c) GFP positivity rises with cell depth", fontsize=11)
    axC.set_xticks(range(1, 11))
    axC.grid(axis="y", color="#eaeaea")
    axC.legend(fontsize=8, frameon=False, ncol=2)

    fig.suptitle("Custom-probe detection reliability across all six slices (non-tumor cells)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_FIG, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("\nSaved:", OUT_FIG, "\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
