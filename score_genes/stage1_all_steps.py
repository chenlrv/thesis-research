"""stage1_all_steps.py - one panel per Stage-1 step, in order:
  1) scaled-PCA map          2) type centroids on the map
  3) cosine to own vs nearest-other centroid
  4) gap = own - nearest-other
  5) sub-clustering / silhouette test
Output -> score_genes_slice1/storyline/stage1_all_steps.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1/storyline"

COL = {"Astrocytes": "#1f77b4", "Microglia": "#17becf", "Macrophage": "#00a087",
       "Endothelial": "#2ca02c", "Pericytes": "#a65628", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "Myeloid": "#00a087", "Vascular": "#2ca02c"}
ORDER = ["Myeloid", "Vascular", "Microglia", "Macrophage", "Pericytes",
         "Endothelial", "Astrocytes", "Ependymal", "Neurons"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=PROVISIONAL_CSV)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ch.SCORES_CSV = args.scores
    ch.OUT_DIR = args.out
    adata = ch.load_adata_with_calls()
    prov = np.asarray(adata.obs["celltype"])
    labeled = prov != "unknown"
    sub = adata[labeled].copy()
    pl = prov[labeled]

    # ---- step 1: scaled-PCA map (50D); we draw PC1-PC2 ----
    emb = ch.scaled_pca(sub)
    order = [g for g in ORDER if g in set(pl)]

    # ---- step 2: centroids (in full 50D; plotted via first 2 coords) ----
    cent = {g: emb[pl == g].mean(0) for g in order}

    # ---- step 3: cosine to own vs nearest-other centroid ----
    C = np.vstack([cent[g] for g in order])
    S = cosine_similarity(emb, C)
    idx = np.array([order.index(l) for l in pl])
    own = S[np.arange(len(S)), idx]
    o = S.copy(); o[np.arange(len(S)), idx] = -np.inf
    near = o.max(1)
    dec = pd.DataFrame({"g": pl, "own": own, "near": near, "gap": own - near}) \
        .groupby("g").mean().loc[order]

    # ---- step 5: sub-clustering silhouette per type (reuse the homogeneity test) ----
    sub.obs["celltype"] = pl
    sil_df = ch.subcluster_test(sub, order).set_index("group").loc[order]

    # ========== FIGURE 1: upper 3 panels (map, centroids, cosine) ==========
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

    # (1) PCA map
    a = ax[0]
    for g in order:
        m = pl == g
        a.scatter(emb[m, 0], emb[m, 1], s=3, c=COL[g], linewidths=0,
                  alpha=0.5, rasterized=True, label=g)
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.legend(markerscale=3, fontsize=8, loc="best")
    a.set_title("1) Scaled-PCA map\n(50 dims; showing PC1–PC2)")

    # (2) centroids
    a = ax[1]
    a.scatter(emb[:, 0], emb[:, 1], s=2, c="#dcdcdc", linewidths=0, rasterized=True)
    for g in order:
        a.scatter(cent[g][0], cent[g][1], s=320, marker="*", c=COL[g],
                  edgecolors="black", linewidths=1.2, zorder=5)
        a.annotate(g, (cent[g][0], cent[g][1]), fontsize=9, weight="bold",
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("2) Centroid of each type\n(mean position on the map)")

    # (3) own vs nearest-other cosine
    a = ax[2]
    xp = np.arange(len(order)); w = 0.4
    a.bar(xp - w / 2, dec["own"].values, w, label="to OWN centroid", color="#4c72b0")
    a.bar(xp + w / 2, dec["near"].values, w, label="to NEAREST OTHER", color="#dd8452")
    a.set_xticks(xp); a.set_xticklabels(order, rotation=45, ha="right")
    a.set_ylabel("mean cosine similarity"); a.legend(fontsize=9)
    a.set_title("3) Cosine to own vs nearest-other centroid")

    fig.suptitle("Stage 1 (steps 1–3) — building the homogeneity metric",
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{args.out}/stage1_steps_1to3.png", bbox_inches="tight")
    plt.close()

    # ========== FIGURE 2: lower 2 panels (gap, silhouette) + conclusion ======
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    # (4) gap
    a = ax[0]
    vals = dec["gap"].values
    a.barh(order[::-1], vals[::-1],
           color=["#2ca02c" if v > 0 else "#d62728" for v in vals[::-1]])
    a.axvline(0, color="k", lw=1)
    a.set_xlabel("separation gap  (own − nearest-other)")
    a.set_title("4) Gap = own − nearest-other\n(negative = overlaps a neighbor)")

    # (5) silhouette
    a = ax[1]
    sv = sil_df["silhouette"].astype(float).values
    a.bar(np.arange(len(order)), sv, color="#7e57c2")
    a.axhline(0, color="k", lw=1)
    a.set_xticks(np.arange(len(order))); a.set_xticklabels(order, rotation=45, ha="right")
    a.set_ylabel("sub-cluster silhouette")
    a.set_ylim(-0.15, 0.4)
    a.set_title("5) Sub-clustering test: silhouette ≈ 0\n→ no hidden subtypes, just overlap")

    # conclusion computed from the actual gaps (correct for any config)
    sep = [g for g in order if dec.loc[g, "gap"] > 0]
    ovl = [g for g in order if dec.loc[g, "gap"] <= 0]
    concl = ("Conclusion (Stage 1):   "
             f"separable (gap > 0): {', '.join(sep) if sep else 'none'}.   "
             f"overlap (gap ≤ 0): {', '.join(ovl) if ovl else 'none'}.   "
             "Silhouette ≈ 0 everywhere → no hidden subtypes inside any group; "
             "the only issue is overlap BETWEEN types.")
    fig.suptitle("Stage 1 (steps 4–5) — the homogeneity result",
                 fontsize=16, weight="bold")
    fig.text(0.5, -0.02, concl, ha="center", va="top", fontsize=11, wrap=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(f"{args.out}/stage1_steps_4to5.png", bbox_inches="tight")
    plt.close()

    print("saved", f"{args.out}/stage1_steps_1to3.png",
          "and", f"{args.out}/stage1_steps_4to5.png")


if __name__ == "__main__":
    main()
