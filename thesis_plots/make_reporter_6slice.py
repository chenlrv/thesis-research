"""Regenerate the two reporter-QC figures across ALL six slices (non-tumor cells).

Figure A: GFP vs tdTomato correlation (bubble scatter + Pearson r), cells with
          GFP>=1 or tdTomato>=1.
Figure B: tdTomato distribution for reporter-defined microglia (GFP+TMEM119+) vs
          MDM (GFP+TMEM119-); % tdTomato>0 per group.

Counts are read from the raw cache; the non-tumor mask is taken from the
tumor-prediction cache (pred_tumor_XGBoost), aligned by cell id.

Run:  conda run -n thesis_research python thesis_plots/make_reporter_6slice.py
"""
import os
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy.stats import pearsonr

ROOT = "D:/thesis-research"
RAW = ROOT + "/resources/cache/slice_{}_adata.h5ad"
WTP = ROOT + "/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT_CORR = ROOT + "/thesis_plots/gfp_tdtomato_correlation_all6.png"
OUT_TMEM = ROOT + "/thesis_plots/gfp_tmem_tdtomato_all6.png"
SLICES = [1, 2, 3, 4, 5, 6]
X_MAX = 16
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}


def get_expr(adata, gene):
    panel = {g.lower(): g for g in adata.var_names}
    key = panel[gene.lower()]
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)


def load_slice(sid):
    a = ad.read_h5ad(RAW.format(sid))
    at = ad.read_h5ad(WTP.format(sid))
    col = "pred_tumor_XGBoost"
    if col not in at.obs.columns:
        col = next(c for c in at.obs.columns if c.startswith("pred_tumor_"))
    tv = at.obs[col].reindex(a.obs_names)
    tumor = tv.astype(object).isin(TRUE).to_numpy()
    nt = ~tumor
    d = {g: get_expr(a, g)[nt] for g in ("GFP", "TMEM119", "tdTomato")}
    del a, at
    return d


def main():
    data = {s: load_slice(s) for s in SLICES}

    # ---------- Figure A: GFP vs tdTomato correlation ------------------------
    figA, axesA = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    figA.suptitle("GFP vs tdTomato (non-tumor cells expressing ≥ 1 count)", fontsize=14)
    print("\n=== GFP vs tdTomato correlation (non-tumor) ===")
    for ax, sid in zip(axesA.ravel(), SLICES):
        gfp, tdt = data[sid]["GFP"], data[sid]["tdTomato"]
        keep = (gfp >= 1) | (tdt >= 1)
        gfp, tdt = gfp[keep], tdt[keep]
        try:
            r = pearsonr(gfp, tdt)[0]
        except Exception:
            r = np.nan
        coords, counts = np.unique(np.column_stack([gfp, tdt]), axis=0, return_counts=True)
        gx, ty = coords[:, 0], coords[:, 1]
        ax.scatter(gx, ty, s=counts / counts.max() * 1200, alpha=0.6,
                   color="tomato", linewidths=0.4, edgecolors="white")
        for xi, yi, cnt in zip(gx, ty, counts):
            if yi <= 14.5:
                ax.text(xi, yi, f"{cnt:,}", ha="center", va="center",
                        fontsize=6, fontweight="bold", color="black")
        ax.text(0.97, 0.97, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        for ref in [10, 100, 1000]:
            ax.scatter([], [], s=ref / counts.max() * 1200, color="tomato",
                       alpha=0.6, label=f"n={ref:,}")
        ax.legend(title="Cells per dot", fontsize=7, title_fontsize=7,
                  loc="lower right", framealpha=0.8)
        ax.set_title(f"Slice {sid}  (n = {gfp.size:,})", fontsize=10)
        ax.set_xlabel("GFP (raw counts)")
        ax.set_ylabel("tdTomato (raw counts)")
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlim(-0.5, max(gfp.max(), 1) + 0.5)
        ax.set_ylim(-0.5, 15)
        print(f"  Slice {sid}: n={gfp.size:,}  Pearson r={r:.3f}")
    figA.tight_layout()
    figA.savefig(OUT_CORR, bbox_inches="tight", dpi=150)
    plt.close(figA)
    print("Saved:", OUT_CORR)

    # ---------- Figure B: TMEM119-based micro/MDM + tdTomato -----------------
    figB, axesB = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    figB.suptitle("tdTomato validation of TMEM119-based separation (non-tumor cells)\n"
                  "Microglia = GFP+ TMEM119+   |   MDMs = GFP+ TMEM119−", fontsize=12)
    bins = np.arange(0.5, X_MAX + 1.5, 1)
    print("\n=== TMEM119-based microglia vs MDM (non-tumor) ===")
    for ax, sid in zip(axesB.ravel(), SLICES):
        gfp, tmem, tdt = data[sid]["GFP"], data[sid]["TMEM119"], data[sid]["tdTomato"]
        micro = (gfp >= 1) & (tmem >= 1)
        mdm = (gfp >= 1) & (tmem == 0)
        p_mic = 100 * (tdt[micro] > 0).mean() if micro.sum() else np.nan
        p_mdm = 100 * (tdt[mdm] > 0).mean() if mdm.sum() else np.nan
        ax.hist(tdt[micro], bins=bins, color="#4393c3", alpha=0.7, histtype="stepfilled",
                edgecolor="none", label=f"Microglia — GFP+ TMEM119+  (n={int(micro.sum()):,})")
        ax.hist(tdt[mdm], bins=bins, color="#d6604d", alpha=0.7, histtype="stepfilled",
                edgecolor="none", label=f"MDMs — GFP+ TMEM119−  (n={int(mdm.sum()):,})")
        ax.set_title(f"Slice {sid}\ntdTomato>0: microglia {p_mic:.1f}%  |  MDMs {p_mdm:.1f}%",
                     fontsize=9)
        ax.set_xlabel("tdTomato (raw counts)")
        ax.set_ylabel("Number of cells")
        ax.set_xlim(0.5, X_MAX + 0.5)
        ax.set_yscale("log")
        ax.legend(fontsize=7.5, framealpha=0.9)
        print(f"  Slice {sid}: micro n={int(micro.sum()):,} ({p_mic:.1f}%)  "
              f"MDM n={int(mdm.sum()):,} ({p_mdm:.1f}%)")
    figB.tight_layout()
    figB.savefig(OUT_TMEM, bbox_inches="tight", dpi=150)
    plt.close(figB)
    print("Saved:", OUT_TMEM)


if __name__ == "__main__":
    main()
