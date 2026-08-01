"""Stage 3 (merged): does high-confidence filtering restore separation?
Recomputes the separation gap on the high-conf subset and compares to provisional,
per type, in the same scaled-PCA space as Stage 1. Plots before/after gap + the
high-conf cells in PCA space.
Output -> score_genes_slice1_merged/storyline/stage3_highconf_gap.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/storyline"

COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2"}
ORDER = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def gap_to(points, point_labels, centroids):
    groups = list(centroids)
    C = np.vstack([centroids[g] for g in groups])
    S = cosine_similarity(points, C)
    idx = np.array([groups.index(l) for l in point_labels])
    own = S[np.arange(len(S)), idx]
    o = S.copy(); o[np.arange(len(S)), idx] = -np.inf
    return own - o.max(1)


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()
    labeled = np.asarray(adata.obs["celltype"]) != "unknown"
    sub = adata[labeled].copy()
    ct_l = np.asarray(adata.obs["celltype"])[labeled]

    hc = pd.read_csv(HC)
    assert np.allclose(hc["x"].to_numpy(), sub.obs["x"].to_numpy(), atol=1e-3)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()

    emb = ch.scaled_pca(sub)                      # labeled-cell PCA (same as Stage 1)
    groups = [g for g in ORDER if g in set(ct_l)]

    cent_prov = {g: emb[ct_l == g].mean(0) for g in groups}
    cent_hc = {g: emb[(ct_l == g) & high].mean(0) for g in groups}

    gprov = gap_to(emb, ct_l, cent_prov)
    ghc = gap_to(emb[high], ct_l[high], cent_hc)

    prov_mean = {g: float(gprov[ct_l == g].mean()) for g in groups}
    hc_mean = {g: float(ghc[ct_l[high] == g].mean()) for g in groups}

    print(f"{'type':12s} {'prov_gap':>9s} {'highconf_gap':>13s} {'n_high':>7s}")
    for g in groups:
        print(f"{g:12s} {prov_mean[g]:+9.3f} {hc_mean[g]:+13.3f} "
              f"{int(((ct_l==g)&high).sum()):7d}")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    xp = np.arange(len(groups)); w = 0.36
    ax[0].bar(xp - w / 2, [prov_mean[g] for g in groups], w, label="provisional",
              color="#bbbbbb")
    ax[0].bar(xp + w / 2, [hc_mean[g] for g in groups], w, label="high-confidence",
              color=[COL[g] for g in groups])
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xticks(xp); ax[0].set_xticklabels(groups, rotation=30, ha="right")
    ax[0].set_ylabel("separation gap (own − nearest-other)")
    ax[0].legend()
    for i, g in enumerate(groups):
        ax[0].text(i - w / 2, prov_mean[g], f"{prov_mean[g]:+.2f}", ha="center",
                   va="bottom" if prov_mean[g] >= 0 else "top", fontsize=9)
        ax[0].text(i + w / 2, hc_mean[g], f"{hc_mean[g]:+.2f}", ha="center",
                   va="bottom" if hc_mean[g] >= 0 else "top", fontsize=9, weight="bold")
    ax[0].set_title("Provisional vs high-confidence separation gap")

    for g in groups:
        m = (ct_l == g) & high
        ax[1].scatter(emb[m, 0], emb[m, 1], s=5, c=COL[g], linewidths=0,
                      alpha=0.6, rasterized=True, label=f"{g} ({int(m.sum())})")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[1].legend(markerscale=2, fontsize=9)
    ax[1].set_title("High-confidence cells in PCA space")

    fig.suptitle("Stage 3 (merged) — does high-confidence filtering restore separation?",
                 fontsize=15, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/stage3_highconf_gap.png", bbox_inches="tight")
    plt.close()
    print("\nsaved", f"{OUT}/stage3_highconf_gap.png")


if __name__ == "__main__":
    main()
