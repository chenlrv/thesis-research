"""Simpler high-conf approach on the BROAD backbone groups (Myeloid, Vascular,
Astrocytes, Ependymal, Neurons): assign each non-tumor cell to its argmax score,
take the top-q% per group by that own score, then check group separation with the
same centroid-cosine / gap test used before (check_homogeneity.similarity_test).

Reports per-group counts (initial argmax vs top-q%) and the separation gap, in:
  * shared : ONE scaled-PCA fit on ALL non-tumor cells (comparable across q).
  * self   : scaled-PCA re-fit on the selected cells only.

Output -> score_genes_slice1_merged/backbone_topscore/
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
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

SCORES_CSV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/backbone_topscore"
QS = [0.10, 0.20, 0.30]
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2"}
UMAP_MAX = 50000  # subsample only for the UMAP picture (gaps use all cells)

sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gap_metrics(emb, group_arr, groups):
    """Centroid cosine matrix + per-group intra vs nearest-other gap."""
    centroids = np.vstack([emb[group_arr == g].mean(0) for g in groups])
    cc = cosine_similarity(centroids)
    rows = []
    for i, g in enumerate(groups):
        m = group_arr == g
        sims = cosine_similarity(emb[m], centroids)
        intra = float(sims[:, i].mean())
        others = np.delete(sims, i, axis=1)
        nearest_other = float(others.max(1).mean())
        j = int(np.argmax(np.delete(sims, i, axis=1).mean(0)))
        nearest_grp = [x for k, x in enumerate(groups) if k != i][j]
        rows.append({"group": g, "n": int(m.sum()),
                     "intra_cos": round(intra, 3),
                     "nearest_other_cos": round(nearest_other, 3),
                     "gap": round(intra - nearest_other, 3),
                     "nearest_grp": nearest_grp})
    return cc, pd.DataFrame(rows)


def select_topq(assign, own_score, q):
    sel = np.zeros(len(assign), bool)
    for g in GROUPS:
        gm = assign == g
        if gm.sum() == 0:
            continue
        thr = np.quantile(own_score[gm], 1 - q)
        sel |= gm & (own_score >= thr)
    return sel


def heatmap(ax, cc, groups, title):
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    for p in range(len(groups)):
        for qcol in range(len(groups)):
            ax.text(qcol, p, f"{cc[p, qcol]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = SCORES_CSV
    adata = ch.load_adata_with_calls()

    df = pd.read_csv(SCORES_CSV)
    assert np.allclose(df["x"].to_numpy(), adata.obs["x"].to_numpy(), atol=1e-3)
    S = df[["score_" + g for g in GROUPS]].to_numpy()
    assign = np.array(GROUPS)[S.argmax(1)]
    own_score = S[np.arange(len(S)), S.argmax(1)]

    print(f"non-tumor cells: {adata.n_obs:,}")
    print("\nper-group counts (argmax assignment):")
    init = {g: int((assign == g).sum()) for g in GROUPS}
    sel10 = select_topq(assign, own_score, 0.10)
    top10 = {g: int((sel10 & (assign == g)).sum()) for g in GROUPS}
    cnt = pd.DataFrame({"group": GROUPS,
                        "initial": [init[g] for g in GROUPS],
                        "top10%": [top10[g] for g in GROUPS]})
    cnt.loc[len(cnt)] = ["TOTAL", sum(init.values()), sum(top10.values())]
    print(cnt.to_string(index=False))
    # for reference: the gated celltype calls
    print("\n(reference) gated celltype calls:")
    print(adata.obs["celltype"].value_counts().to_string())

    # shared scaled-PCA on all non-tumor cells
    print("\ncomputing shared scaled-PCA ...")
    emb_shared = ch.scaled_pca(adata)

    summary = []
    fig_h, ax_h = plt.subplots(1, len(QS) + 1, figsize=(5 * (len(QS) + 1), 4.5), dpi=150)

    def report(name, sel, ax):
        groups_here = [g for g in GROUPS if (assign[sel] == g).sum() >= 2]
        cc_s, tab_s = gap_metrics(emb_shared[sel], assign[sel], groups_here)
        emb_self = ch.scaled_pca(adata[sel].copy())
        cc_f, tab_f = gap_metrics(emb_self, assign[sel], groups_here)
        mean_off_s = (cc_s.sum() - len(groups_here)) / (len(groups_here) ** 2 - len(groups_here))
        print(f"\n=== {name}  (n={int(sel.sum())}) ===")
        merged = tab_s.merge(tab_f[["group", "gap"]], on="group", suffixes=("_shared", "_self"))
        print(merged.to_string(index=False))
        print(f"mean off-diagonal centroid cosine (shared) = {mean_off_s:+.3f}")
        for _, r in merged.iterrows():
            summary.append({"selection": name, **r.to_dict(),
                            "mean_offdiag_cos_shared": round(float(mean_off_s), 3)})
        return heatmap(ax, cc_s, groups_here, f"{name}\n(n={int(sel.sum())})")

    for k, q in enumerate(QS):
        im = report(f"top {q:.0%}", select_topq(assign, own_score, q), ax_h[k])
    im = report("all (argmax)", np.ones(len(assign), bool), ax_h[len(QS)])

    fig_h.suptitle("Backbone centroid similarity (shared scaled-PCA) by selection",
                   fontweight="bold")
    fig_h.colorbar(im, ax=ax_h.tolist(), shrink=0.6, label="centroid cosine")
    fig_h.savefig(f"{OUT}/backbone_centroid_similarity.png", bbox_inches="tight")
    plt.close(fig_h)
    pd.DataFrame(summary).to_csv(f"{OUT}/backbone_gap_summary.csv", index=False)

    # UMAP (subsample for speed), shared space, coloured by argmax group + top10%
    rng = np.random.default_rng(0)
    idx = (np.arange(adata.n_obs) if adata.n_obs <= UMAP_MAX
           else rng.choice(adata.n_obs, UMAP_MAX, replace=False))
    au = adata[idx].copy()
    au.obsm["X_pca"] = emb_shared[idx]
    print(f"\ncomputing UMAP on {au.n_obs:,} cells ...")
    sc.pp.neighbors(au, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(au, random_state=0)
    um = au.obsm["X_umap"]
    a_sub, sel_sub = assign[idx], sel10[idx]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)
    for g in GROUPS:
        m = a_sub == g
        ax[0].scatter(um[m, 0], um[m, 1], s=3, c=COL[g], linewidths=0,
                      rasterized=True, label=f"{g} ({init[g]})")
    ax[0].set_title("all non-tumor by argmax score")
    ax[0].legend(markerscale=3, fontsize=8)
    ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].scatter(um[~sel_sub, 0], um[~sel_sub, 1], s=2, c="#eeeeee", linewidths=0,
                  rasterized=True, label="not selected")
    for g in GROUPS:
        m = sel_sub & (a_sub == g)
        ax[1].scatter(um[m, 0], um[m, 1], s=6, c=COL[g], linewidths=0,
                      rasterized=True, label=f"{g} top10% ({top10[g]})")
    ax[1].set_title("top 10% per group highlighted")
    ax[1].legend(markerscale=3, fontsize=8)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle(f"Backbone UMAP (shared scaled-PCA, n={au.n_obs:,})", fontweight="bold")
    fig.savefig(f"{OUT}/backbone_umap.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved backbone_centroid_similarity.png, backbone_umap.png, "
          f"backbone_gap_summary.csv -> {OUT}")


if __name__ == "__main__":
    main()
