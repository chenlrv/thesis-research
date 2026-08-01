"""make_storyline_figures.py - slide figures for the 3-stage storyline.

Stage 1  initial provisional groups are NOT homogeneous (overlap their neighbors).
Stage 2  high-confidence extraction = two independent signals must agree; works
         for Astrocytes + Ependymal -> method diagram + spatial result.
Stage 3  the two cleaned datasets are now homogeneous (positive separation gap,
         no fragmentation, two clean clusters).

Outputs -> score_genes_slice1/storyline/
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
HC_CSV = "D:/thesis-research/score_genes_slice1/high_confidence_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1/storyline"

COL = {"Astrocytes": "#1f77b4", "Microglia": "#17becf", "Macrophage": "#00a087",
       "Endothelial": "#2ca02c", "Pericytes": "#a65628", "Ependymal": "#984ea3",
       "Neurons": "#e377c2"}
ORDER = ["Microglia", "Macrophage", "Pericytes", "Endothelial",
         "Astrocytes", "Ependymal", "Neurons"]

# held-out non-marker AUCs (from validate_anchors.py / validate_other_types.py)
AUC = {"Astrocytes\nvs Ependymal": 1.00,
       "Microglia\nvs Macrophage": 0.587,
       "Pericytes\nvs Endothelial": 0.613}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def gap_per_cell(points, point_labels, centroids):
    """signed separation gap (own − nearest-other centroid cosine) per cell."""
    groups = list(centroids)
    C = np.vstack([centroids[g] for g in groups])
    S = cosine_similarity(points, C)
    idx = np.array([groups.index(l) for l in point_labels])
    own = S[np.arange(len(S)), idx]
    o = S.copy(); o[np.arange(len(S)), idx] = -np.inf
    return own - o.max(1)


def centroids_of(emb, labels):
    return {g: emb[labels == g].mean(0) for g in sorted(set(labels)) if g != "unknown"}


def main():
    os.makedirs(OUT, exist_ok=True)
    adata = ch.load_adata_with_calls()
    prov = np.asarray(adata.obs["celltype"])
    labeled = prov != "unknown"
    sub = adata[labeled].copy()
    x_l, y_l = sub.obs["x"].to_numpy(), sub.obs["y"].to_numpy()

    hc = pd.read_csv(HC_CSV)
    hc_label = np.where(hc["high_conf"].to_numpy(),
                        hc["provisional_label"].to_numpy(), "unknown")

    # tumor + all coords for the spatial backdrop
    with h5py.File(ch.SLICE, "r") as h5:
        xa = ch._read_obs_num(h5, "CenterX_global_px")
        ya = ch._read_obs_num(h5, "CenterY_global_px")
        tum = ch._read_obs_bool(h5, ch.TUMOR_COL)

    # one consistent embedding for all gap numbers: all-non-tumor scaled PCA
    # (the same space check_homogeneity used, so values match what was reported)
    emb_all = ch.scaled_pca(adata)
    emb_l = emb_all[labeled]
    pl = prov[labeled]

    # ---------- STAGE 1 : initial groups not homogeneous ----------
    cent_prov = centroids_of(emb_l, pl)
    cgroups = list(cent_prov)
    C = np.vstack([cent_prov[g] for g in cgroups])
    S = cosine_similarity(emb_l, C)
    pidx = np.array([cgroups.index(l) for l in pl])
    own = S[np.arange(len(S)), pidx]
    o = S.copy(); o[np.arange(len(S)), pidx] = -np.inf
    near = o.max(1)
    dec = pd.DataFrame({"g": pl, "own": own, "near": near, "gap": own - near}) \
        .groupby("g").mean()
    gprov = dec
    order = [g for g in ORDER if g in dec.index]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    vals = dec.loc[order, "gap"].values
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    ax[0].barh(order[::-1], vals[::-1], color=colors[::-1])
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set_xlabel("separation gap  (own − nearest other type)")
    ax[0].set_title("Initial groups overlap their neighbors\n(negative = not separated)")

    xpos = np.arange(len(order)); w = 0.4
    ax[1].bar(xpos - w / 2, dec.loc[order, "own"].values, w,
              label="similar to OWN type", color="#4c72b0")
    ax[1].bar(xpos + w / 2, dec.loc[order, "near"].values, w,
              label="similar to NEAREST OTHER type", color="#dd8452")
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(order, rotation=45, ha="right")
    ax[1].set_ylabel("mean similarity (cosine)")
    ax[1].legend(fontsize=11)
    ax[1].set_title("Where it breaks: 'other' ≥ 'own'\n(myeloid & vascular indistinguishable)")
    fig.suptitle("Stage 1 — initial annotation is not homogeneous", fontsize=17, weight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/stage1_initial_homogeneity.png", bbox_inches="tight"); plt.close()

    # ---------- STAGE 2a : method diagram ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    a = ax[0]; a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")

    def box(xc, yc, w, h, text, fc):
        a.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                    boxstyle="round,pad=0.15", fc=fc, ec="black", lw=1.5))
        a.text(xc, yc, text, ha="center", va="center", fontsize=12, weight="bold")

    box(2.2, 7.3, 3.6, 1.7, "Marker score\n(panel marker sets)", "#cfe8ff")
    box(2.2, 2.7, 3.6, 1.7, "Independent classifier\n(902 non-marker genes)", "#ffe0cc")
    box(6.4, 5.0, 2.6, 1.7, "AGREE?\nsame label\n& prob ≥ 0.8", "#fff2b3")
    box(9.0, 5.0, 1.9, 1.7, "High-\nconfidence\ncell", "#c9f0c9")
    for (xs, ys) in [(4.0, 7.3), (4.0, 2.7)]:
        a.add_patch(FancyArrowPatch((xs, ys), (5.1, 5.0),
                    arrowstyle="-|>", mutation_scale=18, lw=1.5, color="k"))
    a.add_patch(FancyArrowPatch((7.7, 5.0), (8.05, 5.0),
                arrowstyle="-|>", mutation_scale=18, lw=1.5, color="k"))
    a.text(5, 0.6, "Astrocytes 2378 → 87   •   Ependymal 878 → 243",
           ha="center", fontsize=12, style="italic")
    a.set_title("Method: keep a cell only when two independent signals agree")

    bars = list(AUC.keys()); av = [AUC[b] for b in bars]
    bc = ["#2ca02c" if v > 0.8 else "#d62728" for v in av]
    ax[1].bar(range(len(bars)), av, color=bc)
    ax[1].axhline(0.5, color="gray", ls="--", lw=1)
    ax[1].text(len(bars) - 1, 0.52, "chance", color="gray", fontsize=10, ha="right")
    ax[1].set_xticks(range(len(bars))); ax[1].set_xticklabels(bars)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("held-out AUC (non-marker genes)")
    for i, v in enumerate(av):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=12, weight="bold")
    ax[1].set_title("Why it works only here:\nastro/ependymal separate, myeloid/vascular don't")
    fig.suptitle("Stage 2 — extracting high-confidence cells", fontsize=17, weight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/stage2_method.png", bbox_inches="tight"); plt.close()

    # ---------- STAGE 2b : spatial result ----------
    fig, ax = plt.subplots(figsize=(9, 8.5), dpi=170)
    ax.scatter(xa[~tum], ya[~tum], s=0.7, c="#e2e2e2", linewidths=0, rasterized=True, label="other cells")
    ax.scatter(xa[tum], ya[tum], s=1.2, c="#f3b0b0", linewidths=0, rasterized=True, label="tumor")
    for g in ["Astrocytes", "Ependymal"]:
        m = hc_label == g
        ax.scatter(x_l[m], y_l[m], s=6, c=COL[g], linewidths=0, rasterized=True,
                   label=f"{g} (n={int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", markerscale=3, frameon=True, fontsize=10)
    ax.set_title("Stage 2 — high-confidence Astrocytes & Ependymal in tissue", fontsize=16, weight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/stage2_spatial_result.png", bbox_inches="tight"); plt.close()

    # ---------- STAGE 3 : final homogeneity of the 2 cleaned datasets ----------
    # "after" gap: high-conf cells measured against high-conf centroids, in the
    # SAME all-non-tumor PCA space as stage 1 (so before/after is comparable).
    cent_hc = centroids_of(emb_l, hc_label)
    two = np.isin(hc_label, ["Astrocytes", "Ependymal"])
    gap_hc = gap_per_cell(emb_l[two], hc_label[two], cent_hc)
    ghc = pd.DataFrame({"group": hc_label[two], "gap": gap_hc}).groupby("group").mean()
    # standalone 2-cluster scatter (own PCA on just the two high-conf groups)
    hcsub = sub[two].copy()
    hl = hc_label[two]
    emb2 = ch.scaled_pca(hcsub)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    types = ["Astrocytes", "Ependymal"]
    before = [gprov.loc[t, "gap"] for t in types]
    after = [ghc.loc[t, "gap"] for t in types]
    xpos = np.arange(len(types)); w = 0.36
    ax[0].bar(xpos - w / 2, before, w, label="initial", color="#bbbbbb")
    ax[0].bar(xpos + w / 2, after, w, label="high-confidence",
              color=[COL[t] for t in types])
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(types)
    ax[0].set_ylabel("separation gap")
    ax[0].legend()
    for i, (b, a2) in enumerate(zip(before, after)):
        ax[0].text(i - w / 2, b + 0.01, f"{b:+.2f}", ha="center", fontsize=10)
        ax[0].text(i + w / 2, a2 + 0.01, f"{a2:+.2f}", ha="center", fontsize=10, weight="bold")
    ax[0].set_title("Now well-separated\n(gap turns strongly positive)")

    for g in types:
        m = hl == g
        ax[1].scatter(emb2[m, 0], emb2[m, 1], s=10, c=COL[g], linewidths=0,
                      alpha=0.7, label=f"{g} (n={int(m.sum())})", rasterized=True)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
    ax[1].legend(markerscale=2)
    ax[1].set_title("Two clean, coherent clusters")
    fig.suptitle("Stage 3 — the two high-confidence datasets are homogeneous", fontsize=17, weight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/stage3_final_homogeneity.png", bbox_inches="tight"); plt.close()

    print("saved to", OUT)
    for f in ["stage1_initial_homogeneity.png", "stage2_method.png",
              "stage2_spatial_result.png", "stage3_final_homogeneity.png"]:
        print("  -", f)


if __name__ == "__main__":
    main()
