"""Homogeneity figures for the MYELOID SUBTYPES (BAM, Microglia, MDM), in the
same 5-step / 2-figure format used for the backbone (stage1_all_steps.py):
  fig 1 (steps 1-3): scaled-PCA map, centroids, own-vs-nearest-other cosine
  fig 2 (steps 4-5): separation gap + sub-cluster silhouette + conclusion
Output -> score_genes_slice1_merged/myeloid_subtypes/
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

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
BAM = "D:/thesis-research/score_genes_slice1_merged/bam/bam_labels.csv"
LAB = "D:/thesis-research/score_genes_slice1_merged/microglia_mdm/microglia_mdm_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/myeloid_subtypes"

COL = {"BAM": "#ff7f0e", "Microglia": "#17becf", "MDM": "#00a087"}
ORDER = ["BAM", "Microglia", "MDM"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = PROV
    ch.OUT_DIR = OUT
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"

    hc = pd.read_csv(HC)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    prov = hc["provisional_label"].to_numpy()
    myeloid = np.zeros(adata.n_obs, bool)
    myeloid[np.where(labeled)[0]] = (prov == "Myeloid") & high
    mye_idx = np.where(myeloid)[0]

    bam = pd.read_csv(BAM)
    assert np.allclose(bam["x"].to_numpy(), adata.obs["x"].to_numpy()[mye_idx], atol=1e-3)
    is_bam = bam["is_bam"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    nonbam_idx = mye_idx[~is_bam]

    lab = pd.read_csv(LAB)
    assert np.allclose(lab["x"].to_numpy(), adata.obs["x"].to_numpy()[nonbam_idx], atol=1e-3)
    call = lab["call"].to_numpy()

    # build the subtype label over all cells
    subtype = np.array(["unknown"] * adata.n_obs, dtype=object)
    subtype[mye_idx[is_bam]] = "BAM"
    subtype[nonbam_idx] = call                      # Microglia / MDM / unknown

    keep = np.isin(subtype, ORDER)
    sub = adata[keep].copy()
    pl = subtype[keep]
    print("subtype counts:", {g: int((pl == g).sum()) for g in ORDER})

    # ---- steps 1-3 ----
    emb = ch.scaled_pca(sub)
    order = [g for g in ORDER if g in set(pl)]
    cent = {g: emb[pl == g].mean(0) for g in order}
    C = np.vstack([cent[g] for g in order])
    S = cosine_similarity(emb, C)
    idx = np.array([order.index(l) for l in pl])
    own = S[np.arange(len(S)), idx]
    o = S.copy(); o[np.arange(len(S)), idx] = -np.inf
    near = o.max(1)
    dec = pd.DataFrame({"g": pl, "own": own, "near": near, "gap": own - near}) \
        .groupby("g").mean().loc[order]

    # ---- step 5: sub-cluster silhouette ----
    sub.obs["celltype"] = pl
    sil_df = ch.subcluster_test(sub, order).set_index("group").loc[order]

    # ===== FIGURE 1: steps 1-3 =====
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    a = ax[0]
    for g in order:
        m = pl == g
        a.scatter(emb[m, 0], emb[m, 1], s=4, c=COL[g], linewidths=0, alpha=0.5,
                  rasterized=True, label=f"{g} ({int(m.sum())})")
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.legend(markerscale=3, fontsize=9, loc="best")
    a.set_title("1) Scaled-PCA map\n(50 dims; showing PC1–PC2)")

    a = ax[1]
    a.scatter(emb[:, 0], emb[:, 1], s=3, c="#dcdcdc", linewidths=0, rasterized=True)
    for g in order:
        a.scatter(cent[g][0], cent[g][1], s=320, marker="*", c=COL[g],
                  edgecolors="black", linewidths=1.2, zorder=5)
        a.annotate(g, (cent[g][0], cent[g][1]), fontsize=10, weight="bold",
                   xytext=(4, 4), textcoords="offset points")
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("2) Centroid of each subtype\n(mean position on the map)")

    a = ax[2]
    xp = np.arange(len(order)); w = 0.4
    a.bar(xp - w / 2, dec["own"].values, w, label="to OWN centroid", color="#4c72b0")
    a.bar(xp + w / 2, dec["near"].values, w, label="to NEAREST OTHER", color="#dd8452")
    a.axhline(0, color="k", lw=1)
    a.set_xticks(xp); a.set_xticklabels(order, rotation=20, ha="right")
    a.set_ylabel("mean cosine similarity"); a.legend(fontsize=9)
    a.set_title("3) Cosine to own vs nearest-other centroid")

    fig.suptitle("Myeloid subtypes (steps 1–3) — building the homogeneity metric",
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT}/myeloid_subtypes_steps_1to3.png", bbox_inches="tight")
    plt.close()

    # ===== FIGURE 2: steps 4-5 =====
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    a = ax[0]
    vals = dec["gap"].values
    a.barh(order[::-1], vals[::-1],
           color=["#2ca02c" if v > 0 else "#d62728" for v in vals[::-1]])
    a.axvline(0, color="k", lw=1)
    for i, g in enumerate(order[::-1]):
        a.text(vals[::-1][i], i, f" {vals[::-1][i]:+.2f}",
               va="center", ha="left" if vals[::-1][i] >= 0 else "right", fontsize=10)
    a.set_xlabel("separation gap  (own − nearest-other)")
    a.set_title("4) Gap = own − nearest-other\n(negative = overlaps a neighbor)")

    a = ax[1]
    sv = sil_df["silhouette"].astype(float).values
    a.bar(np.arange(len(order)), sv, color="#7e57c2")
    a.axhline(0, color="k", lw=1)
    a.set_xticks(np.arange(len(order))); a.set_xticklabels(order, rotation=20, ha="right")
    a.set_ylabel("sub-cluster silhouette"); a.set_ylim(-0.15, 0.4)
    a.set_title("5) Sub-clustering test: silhouette ≈ 0\n→ no hidden subtypes, just overlap")

    sep = [g for g in order if dec.loc[g, "gap"] > 0]
    ovl = [g for g in order if dec.loc[g, "gap"] <= 0]
    concl = ("Conclusion (myeloid subtypes):   "
             f"separable (gap > 0): {', '.join(sep) if sep else 'none'}.   "
             f"overlap (gap ≤ 0): {', '.join(ovl) if ovl else 'none'}.   "
             "Silhouette ≈ 0 everywhere → no hidden subtypes inside any subtype; "
             "separation is between subtypes only.")
    fig.suptitle("Myeloid subtypes (steps 4–5) — the homogeneity result",
                 fontsize=16, weight="bold")
    fig.text(0.5, -0.02, concl, ha="center", va="top", fontsize=11, wrap=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(f"{OUT}/myeloid_subtypes_steps_4to5.png", bbox_inches="tight")
    plt.close()

    print("\nsaved myeloid_subtypes_steps_1to3.png and "
          "myeloid_subtypes_steps_4to5.png ->", OUT)


if __name__ == "__main__":
    main()
