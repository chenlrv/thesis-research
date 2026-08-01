"""Simpler high-conf approach: take the top-q% cells per group by marker score,
then check group separation with the same centroid-cosine / gap test used before
(check_homogeneity.similarity_test). Sweeps q and compares to the full called set.

Selection per group (disjoint): assign each non-BAM myeloid cell to whichever of
{Microglia, MDM} has the higher score (margin > 0), then keep the top-q% by that
own score.

Separation is reported in two spaces:
  * shared : ONE scaled-PCA fit on ALL non-BAM myeloid -> q values are directly
             comparable to each other and to the original -0.30 heatmap.
  * self   : scaled-PCA re-fit on the selected cells only -> the honest "do these
             extremes separate in their own space" view.

Output -> score_genes_slice1_merged/microglia_mdm/topscore/
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
import check_homogeneity as ch  # noqa: E402  (loader + scaled_pca reuse)
import myeloid_microglia_mdm as mm  # noqa: E402  (constants reuse)

OUT = mm.OUT + "/topscore"
QS = [0.10, 0.20, 0.30]
GROUPS = ["Microglia", "MDM"]
COL = {"Microglia": "#17becf", "MDM": "#00a087", "unknown": "#dddddd"}

sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def build_sub():
    """Non-BAM clean myeloid + Microglia/MDM scores (same as myeloid_microglia_mdm)."""
    ch.SCORES_CSV = mm.PROV
    adata = ch.load_adata_with_calls()
    labeled = np.asarray(adata.obs["celltype"]) != "unknown"

    hc = pd.read_csv(mm.HC)
    assert np.allclose(hc["x"].to_numpy(), adata.obs["x"].to_numpy()[labeled], atol=1e-3)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    prov = hc["provisional_label"].to_numpy()
    myeloid = np.zeros(adata.n_obs, bool)
    myeloid[np.where(labeled)[0]] = (prov == "Myeloid") & high
    mye_idx = np.where(myeloid)[0]

    bam = pd.read_csv(mm.BAM)
    assert len(bam) == len(mye_idx), "bam_labels not aligned to myeloid"
    is_bam = bam["is_bam"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    nonbam_idx = mye_idx[~is_bam]
    mask = np.zeros(adata.n_obs, bool)
    mask[nonbam_idx] = True
    sub = adata[mask].copy()

    for label, genes in mm.MARKERS.items():
        g = [x for x in genes if x in sub.var_names]
        sc.tl.score_genes(sub, gene_list=g, score_name="score_" + label,
                          ctrl_size=50, n_bins=25)
    S = sub.obs[["score_" + l for l in mm.MARKERS]].copy()
    S.columns = list(mm.MARKERS)
    return sub, S


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
        rows.append({"group": g, "n": int(m.sum()),
                     "intra_cos": round(intra, 3),
                     "nearest_other_cos": round(nearest_other, 3),
                     "gap": round(intra - nearest_other, 3)})
    return cc, pd.DataFrame(rows)


def select_topq(assign, own_score, q):
    """Boolean mask: top-q% by own_score within each assigned group."""
    sel = np.zeros(len(assign), bool)
    for g in GROUPS:
        gm = assign == g
        if gm.sum() == 0:
            continue
        thr = np.quantile(own_score[gm], 1 - q)
        sel |= gm & (own_score >= thr)
    return sel


def main():
    os.makedirs(OUT, exist_ok=True)
    sub, S = build_sub()
    print(f"non-BAM clean myeloid: {sub.n_obs:,} cells")

    micro = S["Microglia"].to_numpy()
    mdm = S["MDM"].to_numpy()
    margin = micro - mdm
    assign = np.full(sub.n_obs, "unknown", dtype=object)
    assign[margin > 0] = "Microglia"
    assign[margin < 0] = "MDM"
    own_score = np.where(assign == "Microglia", micro,
                         np.where(assign == "MDM", mdm, -np.inf))
    print(f"argmax assignment (margin>0): "
          f"Microglia={int((assign=='Microglia').sum())}, "
          f"MDM={int((assign=='MDM').sum())}")

    # shared scaled-PCA space (fit on all non-BAM myeloid)
    emb_shared = ch.scaled_pca(sub)

    # UMAP for the visual (shared space)
    sub.obsm["X_pca"] = emb_shared
    sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(sub, random_state=0)
    um = sub.obsm["X_umap"]

    summary = []
    fig_h, ax_h = plt.subplots(1, len(QS) + 1, figsize=(5 * (len(QS) + 1), 4.5), dpi=150)

    def report(name, sel, ax):
        groups_here = [g for g in GROUPS if (assign[sel] == g).sum() >= 2]
        cc_s, tab_s = gap_metrics(emb_shared[sel], assign[sel], groups_here)
        emb_self = ch.scaled_pca(sub[sel].copy())
        cc_f, tab_f = gap_metrics(emb_self, assign[sel], groups_here)
        off_s = float(cc_s[0, 1]) if cc_s.shape[0] == 2 else np.nan
        off_f = float(cc_f[0, 1]) if cc_f.shape[0] == 2 else np.nan
        print(f"\n=== {name}  (n={int(sel.sum())}) ===")
        merged = tab_s.merge(tab_f, on=["group", "n"], suffixes=("_shared", "_self"))
        print(merged.to_string(index=False))
        print(f"centroid cosine  shared={off_s:+.3f}  self={off_f:+.3f}")
        for _, r in merged.iterrows():
            summary.append({"selection": name, **r.to_dict(),
                            "centroid_cos_shared": round(off_s, 3),
                            "centroid_cos_self": round(off_f, 3)})
        # heatmap = shared-space centroid cosine
        im = ax.imshow(cc_s, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(groups_here)))
        ax.set_xticklabels(groups_here, rotation=45, ha="right")
        ax.set_yticks(range(len(groups_here)))
        ax.set_yticklabels(groups_here)
        for p in range(len(groups_here)):
            for qcol in range(len(groups_here)):
                ax.text(qcol, p, f"{cc_s[p, qcol]:.2f}", ha="center", va="center", fontsize=10)
        ax.set_title(f"{name}\n(n={int(sel.sum())})")
        return im

    for k, q in enumerate(QS):
        sel = select_topq(assign, own_score, q)
        im = report(f"top {q:.0%}", sel, ax_h[k])
    # baseline: all assigned (q=100%)
    sel_all = assign != "unknown"
    im = report("all assigned", sel_all, ax_h[len(QS)])

    fig_h.suptitle("Microglia vs MDM centroid similarity (shared scaled-PCA) by selection",
                   fontweight="bold")
    fig_h.colorbar(im, ax=ax_h.tolist(), shrink=0.6, label="centroid cosine")
    fig_h.savefig(f"{OUT}/topscore_centroid_similarity.png", bbox_inches="tight")
    plt.close(fig_h)

    pd.DataFrame(summary).to_csv(f"{OUT}/topscore_gap_summary.csv", index=False)

    # UMAP: left = all by argmax, right = top-10% highlighted
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    for g in ["unknown", "Microglia", "MDM"]:
        m = assign == g
        ax[0].scatter(um[m, 0], um[m, 1], s=5, c=COL[g], linewidths=0,
                      rasterized=True, label=f"{g} ({int(m.sum())})")
    ax[0].set_title("all non-BAM myeloid by argmax score")
    ax[0].legend(markerscale=2, fontsize=8)
    ax[0].set_xticks([]); ax[0].set_yticks([])

    sel = select_topq(assign, own_score, QS[0])
    ax[1].scatter(um[~sel, 0], um[~sel, 1], s=4, c="#eeeeee", linewidths=0,
                  rasterized=True, label="not selected")
    for g in GROUPS:
        m = sel & (assign == g)
        ax[1].scatter(um[m, 0], um[m, 1], s=10, c=COL[g], linewidths=0,
                      rasterized=True, label=f"{g} top{QS[0]:.0%} ({int(m.sum())})")
    ax[1].set_title(f"top {QS[0]:.0%} per group highlighted")
    ax[1].legend(markerscale=2, fontsize=8)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle("non-BAM myeloid UMAP (shared scaled-PCA)", fontweight="bold")
    fig.savefig(f"{OUT}/topscore_umap.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved topscore_centroid_similarity.png, topscore_umap.png, "
          f"topscore_gap_summary.csv -> {OUT}")


if __name__ == "__main__":
    main()
