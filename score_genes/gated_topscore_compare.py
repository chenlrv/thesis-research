"""Two ways to take the top 20% of the GATED cells (score_genes -> FDR 0.05 +
scaled-margin >=1.5 calls), compared on the same gap-separation test.

  Method 1 - global score bar:   keep cells whose scaled top-score is in the top
             20% of the pooled gated distribution (one threshold for everyone).
             Groups contribute UNEQUAL numbers of cells.
  Method 2 - per-group top 20%:  keep the top 20% by scaled top-score WITHIN each
             group. Every group contributes 20% of its cells.

Separation is measured in ONE shared scaled-PCA fit on the full gated pool (the
same space the original centroid-similarity test used): per-group gap =
intra-centroid cosine - nearest-other-centroid cosine, plus the centroid cosine
matrix. Baseline = all gated cells (100%).

Output -> score_genes_slice1_merged/gated_topscore/
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/gated_topscore"
TOPFRAC = 0.20
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2"}

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gap_metrics(emb, lab, groups):
    centroids = np.vstack([emb[lab == g].mean(0) for g in groups])
    cc = cosine_similarity(centroids)
    rows = []
    for i, g in enumerate(groups):
        m = lab == g
        sims = cosine_similarity(emb[m], centroids)
        intra = float(sims[:, i].mean())
        others = np.delete(sims, i, axis=1)
        nearest = float(others.max(1).mean())
        j = int(np.argmax(others.mean(0)))
        nearest_grp = [x for k, x in enumerate(groups) if k != i][j]
        rows.append({"group": g, "n": int(m.sum()),
                     "intra_cos": round(intra, 3),
                     "nearest_other_cos": round(nearest, 3),
                     "gap": round(intra - nearest, 3),
                     "nearest_grp": nearest_grp})
    return cc, pd.DataFrame(rows)


def draw_heatmap(ax, cc, groups, title):
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    for p in range(len(groups)):
        for q in range(len(groups)):
            ax.text(q, p, f"{cc[p, q]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return im


def draw_gapbar(ax, base_tab, tab, title):
    g = tab["group"].tolist()
    x = np.arange(len(g))
    ax.bar(x - 0.2, base_tab.set_index("group").loc[g, "gap"], width=0.4,
           color="#cccccc", label="all gated")
    ax.bar(x + 0.2, tab["gap"], width=0.4,
           color=[COL[k] for k in g], label="selected")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(g, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("separation gap")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = SCORES_CSV
    adata = ch.load_adata_with_calls()
    df = pd.read_csv(SCORES_CSV)
    assert np.allclose(df["x"].to_numpy(), adata.obs["x"].to_numpy(), atol=1e-3)

    gated = (df["celltype"] != "unknown").to_numpy()
    sub = adata[gated].copy()
    lab = df["celltype"].to_numpy()[gated]
    score = df["top_scaled"].to_numpy()[gated]
    print(f"gated cells: {sub.n_obs:,}")

    # shared scaled-PCA on the full gated pool (the original test's space)
    emb = ch.scaled_pca(sub)

    # ---- Method 1: global score bar (top 20% of pooled scaled score) ----
    thr = np.quantile(score, 1 - TOPFRAC)
    sel1 = score >= thr
    print(f"\nMethod 1 - global bar: top_scaled >= {thr:.3f}  "
          f"(keeps {int(sel1.sum()):,} / {sub.n_obs:,} = {100*sel1.mean():.1f}%)")

    # ---- Method 2: per-group top 20% ----
    sel2 = np.zeros(sub.n_obs, bool)
    for g in GROUPS:
        gm = lab == g
        t = np.quantile(score[gm], 1 - TOPFRAC)
        sel2 |= gm & (score >= t)
    print(f"Method 2 - per-group top {TOPFRAC:.0%}: keeps {int(sel2.sum()):,} cells")

    # ---- counts table ----
    counts = pd.DataFrame({
        "group": GROUPS,
        "gated": [int((lab == g).sum()) for g in GROUPS],
        "m1_global_bar": [int((sel1 & (lab == g)).sum()) for g in GROUPS],
        "m2_pergroup20": [int((sel2 & (lab == g)).sum()) for g in GROUPS],
    })
    counts.loc[len(counts)] = ["TOTAL", counts["gated"].sum(),
                               counts["m1_global_bar"].sum(),
                               counts["m2_pergroup20"].sum()]
    print("\nper-group counts:")
    print(counts.to_string(index=False))
    counts.to_csv(f"{OUT}/gated_topscore_counts.csv", index=False)

    # ---- gap metrics for baseline + both methods ----
    cc0, tab0 = gap_metrics(emb, lab, GROUPS)
    cc1, tab1 = gap_metrics(emb[sel1], lab[sel1], GROUPS)
    cc2, tab2 = gap_metrics(emb[sel2], lab[sel2], GROUPS)
    for name, tab, cc in [("all gated", tab0, cc0),
                          ("Method 1 (global bar)", tab1, cc1),
                          ("Method 2 (per-group 20%)", tab2, cc2)]:
        off = (cc.sum() - len(GROUPS)) / (len(GROUPS) ** 2 - len(GROUPS))
        print(f"\n=== {name} ===")
        print(tab.to_string(index=False))
        print(f"mean gap = {tab['gap'].mean():+.3f}   "
              f"mean off-diag centroid cosine = {off:+.3f}")

    summ = pd.concat([tab0.assign(selection="all_gated"),
                      tab1.assign(selection="m1_global_bar"),
                      tab2.assign(selection="m2_pergroup20")])
    summ.to_csv(f"{OUT}/gated_topscore_gap_summary.csv", index=False)

    # ---- figure: heatmaps (row 0) + gap bars (row 1) for each method ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 11), dpi=150)
    im = draw_heatmap(ax[0, 0], cc1, GROUPS,
                      f"Method 1 - global bar (n={int(sel1.sum()):,})")
    draw_heatmap(ax[0, 1], cc2, GROUPS,
                 f"Method 2 - per-group 20% (n={int(sel2.sum()):,})")
    fig.colorbar(im, ax=ax[0, :].tolist(), shrink=0.7, label="centroid cosine")
    draw_gapbar(ax[1, 0], tab0, tab1, "Method 1 - global bar")
    draw_gapbar(ax[1, 1], tab0, tab2, "Method 2 - per-group 20%")
    fig.suptitle("Top-20% gated cells: global score bar vs per-group quantile",
                 fontsize=15, fontweight="bold")
    fig.savefig(f"{OUT}/gated_topscore_compare.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved gated_topscore_compare.png, gated_topscore_counts.csv, "
          f"gated_topscore_gap_summary.csv -> {OUT}")


if __name__ == "__main__":
    main()
