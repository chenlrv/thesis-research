"""Figure: Lyve1 is unreliable as a sole BAM anchor - precision + spatial.

(a) Of Lyve1+ non-tumor cells, the large majority carry NO canonical BAM marker
    (Mrc1/Cd163), across all 6 slices  [read from lyve1_unreliability_stats.csv].
(b) Spatial map (slice 1): Lyve1+ cells are spread across the whole section, not
    confined to a BAM-like niche; the small canonical-confirmed subset (Lyve1+ AND
    Mrc1/Cd163+) is a sparse minority. Visualises the precision failure in (a).

Output -> thesis_plots/lyve1_unreliability_spatial.png
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.sparse import issparse

RAW = "D:/thesis-research/resources/cache/slice_1_adata.h5ad"
TUM = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
TUMOR_COL = "pred_tumor_XGBoost"
STATS = "D:/thesis-research/thesis_plots/lyve1_unreliability_stats.csv"
OUT = "D:/thesis-research/thesis_plots/lyve1_unreliability_spatial.png"

CANON = ["Mrc1", "Cd163"]
BLUE, RED, GREY, TUMGREY = "#2a78d6", "#e34948", "#c9c8c3", "#ececea"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def counts(adata, gene):
    panel = {g.lower(): g for g in adata.var_names}
    x = adata[:, panel[gene.lower()]].X
    return (x.toarray().ravel() if issparse(x) else np.asarray(x).ravel())


def main():
    # ---------- panel (b) data: slice 1 ----------
    print("loading slice 1 ...")
    a = ad.read_h5ad(RAW)
    t = ad.read_h5ad(TUM)
    col = TUMOR_COL if TUMOR_COL in t.obs else next(
        c for c in t.obs.columns if c.startswith("pred_tumor_"))
    tum = t.obs[col].reindex(a.obs_names).fillna(0).to_numpy().astype(int) == 1
    x = a.obs["CenterX_global_px"].to_numpy(float)
    y = a.obs["CenterY_global_px"].to_numpy(float)
    lyve = counts(a, "Lyve1") >= 1
    canon = np.zeros(a.n_obs, bool)
    for g in CANON:
        canon |= counts(a, g) >= 1

    nt = ~tum
    bg = nt & ~lyve                        # non-tumor, Lyve1-
    lyve_only = nt & lyve & ~canon         # Lyve1+ but no canonical BAM marker
    lyve_canon = nt & lyve & canon         # Lyve1+ AND canonical BAM marker
    print(f"non-tumor Lyve1+ = {int((nt & lyve).sum()):,} | "
          f"Lyve1+ no-marker = {int(lyve_only.sum()):,} | "
          f"Lyve1+ & Mrc1/Cd163 = {int(lyve_canon.sum()):,}")

    # ---------- panel (a) data: precision across 6 slices ----------
    df = pd.read_csv(STATS).sort_values("slice")
    slabs = [f"S{int(s)}" for s in df["slice"]]
    xb = np.arange(len(slabs))
    prec = df["precision_pct"].to_numpy()

    # ================= figure =================
    fig = plt.figure(figsize=(15, 6.4), dpi=170)
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.35], wspace=0.18, figure=fig)

    # ---- (a) precision stacked bars ----
    axb = fig.add_subplot(gs[0, 0])
    axb.bar(xb, prec, 0.62, color=BLUE, zorder=3,
            label="carry a canonical BAM marker (Mrc1/Cd163)")
    axb.bar(xb, 100 - prec, 0.62, bottom=prec, color=GREY, zorder=3,
            label="NO canonical BAM marker")
    for xi, p in zip(xb, 100 - prec):
        axb.text(xi, 100 - p / 2, f"{p:.0f}%", ha="center", va="center",
                 fontsize=9.5, color=INK, fontweight="bold")
    axb.set_title("(a) Most Lyve1+ cells are not BAM", fontweight="bold",
                  color=INK, fontsize=12.5, loc="left")
    axb.set_ylabel("% of Lyve1+ non-tumor cells", color=INK)
    axb.set_ylim(0, 100)
    axb.set_xticks(xb); axb.set_xticklabels(slabs)
    axb.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axb.set_axisbelow(True)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    axb.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22),
               fontsize=9, ncol=1)

    # ---- (b) spatial map, slice 1 ----
    axs = fig.add_subplot(gs[0, 1])
    axs.scatter(x[tum], y[tum], s=0.5, c=TUMGREY, linewidths=0, rasterized=True)
    axs.scatter(x[bg], y[bg], s=0.5, c=GREY, linewidths=0, rasterized=True)
    axs.scatter(x[lyve_only], y[lyve_only], s=2.6, c=BLUE, linewidths=0,
                rasterized=True,
                label=f"Lyve1+, no BAM marker  ({int(lyve_only.sum()):,})")
    axs.scatter(x[lyve_canon], y[lyve_canon], s=11, c=RED, linewidths=0,
                rasterized=True,
                label=f"Lyve1+ & Mrc1/Cd163+  ({int(lyve_canon.sum()):,})")
    axs.set_aspect("equal"); axs.set_xticks([]); axs.set_yticks([])
    for sp in axs.spines.values():
        sp.set_visible(False)
    axs.set_title("(b) Lyve1+ cells span the whole section, few are BAM "
                  "(slice 1)", fontweight="bold", color=INK, fontsize=12.5,
                  loc="left")
    leg = axs.legend(loc="lower right", markerscale=2.6, fontsize=9.5,
                     frameon=True, framealpha=0.9)
    leg.get_frame().set_edgecolor(GRID)

    fig.suptitle("Lyve1 is unreliable as a sole BAM anchor",
                 fontsize=15, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
