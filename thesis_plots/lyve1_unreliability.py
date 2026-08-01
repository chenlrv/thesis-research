"""Figure: Lyve1 is unreliable as a sole BAM anchor (all 6 slices, raw counts).

Three orthogonal failure modes, non-tumor cells only:
  (a) Over-detected  - Lyve1 is positive in far more cells than the canonical
      BAM markers Mrc1/Cd163/Pf4, so Lyve1+ cannot be BAM-specific.
  (b) Low precision  - of Lyve1+ cells, the large majority carry NO canonical
      BAM marker (Mrc1/Cd163), so Lyve1 positivity alone over-calls BAM.
  (c) Low recall     - of cells that DO carry a canonical BAM marker, only a
      minority are Lyve1+, so Lyve1 also misses most candidate BAMs.

Output -> thesis_plots/lyve1_unreliability.png  (+ lyve1_unreliability_stats.csv)
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.sparse import issparse

RAW = "D:/thesis-research/resources/cache/slice_{i}_adata.h5ad"
TUM = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{i}_adata.h5ad"
TUMOR_COL = "pred_tumor_XGBoost"
SLICES = range(1, 7)

LYVE1 = "Lyve1"
CANON = ["Mrc1", "Cd163"]          # canonical BAM markers used to test Lyve1
DETECT = ["Lyve1", "Mrc1", "Cd163"]

# validated categorical palette (dataviz skill, light surface)
COL = {"Lyve1": "#2a78d6", "Mrc1": "#008300", "Cd163": "#e87ba4", "Pf4": "#eda100"}
GREY = "#c9c8c3"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def counts(adata, gene):
    panel = {g.lower(): g for g in adata.var_names}
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"{gene} not in panel")
    x = adata[:, key].X
    return (x.toarray().ravel() if issparse(x) else np.asarray(x).ravel())


def load_slice(i):
    a = ad.read_h5ad(RAW.format(i=i))
    t = ad.read_h5ad(TUM.format(i=i))
    col = TUMOR_COL if TUMOR_COL in t.obs else next(
        c for c in t.obs.columns if c.startswith("pred_tumor_"))
    tum = t.obs[col].reindex(a.obs_names).fillna(0).to_numpy().astype(int) == 1
    nt = ~tum
    lyve = counts(a, LYVE1)[nt] >= 1
    canon = np.zeros(nt.sum(), bool)
    for g in CANON:
        canon |= counts(a, g)[nt] >= 1
    detect = {g: (counts(a, g)[nt] >= 1) for g in DETECT}
    return {"n": int(nt.sum()), "lyve": lyve, "canon": canon, "detect": detect}


def main():
    rows = []
    D = {}
    for i in SLICES:
        print(f"loading slice {i} ...")
        s = load_slice(i)
        D[i] = s
        n = s["n"]
        lyve, canon = s["lyve"], s["canon"]
        n_lyve = int(lyve.sum())
        n_canon = int(canon.sum())
        both = int((lyve & canon).sum())
        rows.append({
            "slice": i, "n_nontumor": n,
            "pct_lyve1": 100 * n_lyve / n,
            "pct_mrc1": 100 * s["detect"]["Mrc1"].mean(),
            "pct_cd163": 100 * s["detect"]["Cd163"].mean(),
            "lyve1_pos": n_lyve, "canon_pos": n_canon, "both": both,
            "precision_pct": 100 * both / n_lyve if n_lyve else np.nan,   # of Lyve1+, %canonical+
            "recall_pct": 100 * both / n_canon if n_canon else np.nan,    # of canonical+, %Lyve1+
        })
    df = pd.DataFrame(rows)
    out_csv = "D:/thesis-research/thesis_plots/lyve1_unreliability_stats.csv"
    df.round(2).to_csv(out_csv, index=False)
    print("\n" + df.round(1).to_string(index=False))

    slabs = [f"S{i}" for i in SLICES]
    x = np.arange(len(slabs))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=170)

    # ---- (a) detection frequency: Lyve1 vs canonical BAM markers ----
    ax = axes[0]
    w = 0.26
    for j, g in enumerate(DETECT):
        vals = df[f"pct_{g.lower()}"].to_numpy()
        ax.bar(x + (j - 1) * w, vals, w * 0.9, color=COL[g], label=g, zorder=3)
    ax.set_title("(a) Lyve1 is over-detected", fontweight="bold", color=INK,
                 fontsize=12.5, loc="left")
    ax.set_ylabel("% of non-tumor cells positive (≥1 count)", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(slabs)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.0), fontsize=9.5, columnspacing=1.2,
              handlelength=1.1)
    ax.margins(y=0.18)

    # ---- (b) precision: of Lyve1+ cells, canonical+ vs no canonical marker ----
    ax = axes[1]
    prec = df["precision_pct"].to_numpy()
    ax.bar(x, prec, 0.62, color=COL["Lyve1"], zorder=3,
           label="carry a canonical BAM marker (Mrc1/Cd163)")
    ax.bar(x, 100 - prec, 0.62, bottom=prec, color=GREY, zorder=3,
           label="NO canonical BAM marker")
    for xi, p in zip(x, 100 - prec):
        ax.text(xi, 100 - p / 2, f"{p:.0f}%", ha="center", va="center",
                fontsize=9.5, color=INK, fontweight="bold")
    ax.set_title("(b) Most Lyve1+ cells are not BAM", fontweight="bold",
                 color=INK, fontsize=12.5, loc="left")
    ax.set_ylabel("% of Lyve1+ non-tumor cells", color=INK)
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(slabs)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              fontsize=9, ncol=1)

    # ---- (c) recall: of canonical BAM-marker+ cells, Lyve1+ vs Lyve1- ----
    ax = axes[2]
    rec = df["recall_pct"].to_numpy()
    ax.bar(x, rec, 0.62, color=COL["Lyve1"], zorder=3, label="Lyve1+")
    ax.bar(x, 100 - rec, 0.62, bottom=rec, color=GREY, zorder=3, label="Lyve1−")
    for xi, r in zip(x, rec):
        ax.text(xi, r / 2, f"{r:.0f}%", ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold")
    ax.set_title("(c) Lyve1 misses most marker+ cells", fontweight="bold",
                 color=INK, fontsize=12.5, loc="left")
    ax.set_ylabel("% of Mrc1/Cd163+ non-tumor cells", color=INK)
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(slabs)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28),
              fontsize=9, ncol=2)

    for ax in axes:
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle("Lyve1 is unreliable as a sole BAM anchor",
                 fontsize=15, fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "D:/thesis-research/thesis_plots/lyve1_unreliability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}\nsaved {out_csv}")


if __name__ == "__main__":
    main()
