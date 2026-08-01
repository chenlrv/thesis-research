"""select_anchors.py - high-confidence cell-type anchors via a 3-condition AND.

A provisional-called cell becomes an anchor only if ALL hold:
  (1) marker gate     - it passed the original score/FDR/margin rule (i.e. its
                        provisional celltype != 'unknown').
  (2) centroid gap    - in scaled-PCA space it is closer to its own group
                        centroid than to every other, by a meaningful margin:
                        top TOP_PCT within its group AND gap > GAP_FLOOR.
                        (top-% alone isn't enough: Microglia/Macrophage/Pericytes
                        have negative average gap, so a floor is required or we
                        re-import the contamination.)
  (3) kNN purity      - >= PURITY_MIN of its K nearest labeled neighbors in
                        PCA space share its provisional label.

Reuses the loader + scaled-PCA from check_homogeneity. Baseline calls
(FDR 0.05 / margin 1.5) are the provisional pool. Outputs an anchors table and a
spatial map; reports how many anchors survive per group and the purity gain.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402  (loader + scaled_pca reuse)

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"  # baseline
OUT_DIR = "D:/thesis-research/score_genes_slice1"

TOP_PCT = 0.30        # keep the top 30% most-separated cells within each group
GAP_FLOOR = 0.0       # quality floor: own_sim - nearest_other_sim must exceed this
KNN_K = 30            # neighbours for the purity check (labeled cells only)
PURITY_MIN = 0.70     # >= this fraction of neighbours must share the label

CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4", "Microglia": "#17becf", "Macrophage": "#00a087",
    "Endothelial": "#2ca02c", "Pericytes": "#a65628", "Ependymal": "#984ea3",
    "Neurons": "#e377c2",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ch.SCORES_CSV = PROVISIONAL_CSV
    adata = ch.load_adata_with_calls()

    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    emb = ch.scaled_pca(adata)                 # all non-tumor (same space as check 3)
    emb_l = emb[labeled]
    ct_l = ct[labeled]
    x_l = adata.obs["x"].to_numpy()[labeled]
    y_l = adata.obs["y"].to_numpy()[labeled]

    groups = sorted(set(ct_l))
    gidx = {g: i for i, g in enumerate(groups)}

    # ---- (2) centroid gap (own vs nearest-other cosine), labeled cells ----
    centroids = np.vstack([emb_l[ct_l == g].mean(0) for g in groups])
    sims = cosine_similarity(emb_l, centroids)          # cells x groups
    own_i = np.array([gidx[g] for g in ct_l])
    own_sim = sims[np.arange(len(sims)), own_i]
    others = sims.copy()
    others[np.arange(len(sims)), own_i] = -np.inf
    nearest_other = others.max(1)
    gap = own_sim - nearest_other

    # ---- (3) kNN purity among labeled cells in PCA space ----
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb_l)
    _, nbr = nn.kneighbors(emb_l)
    nbr = nbr[:, 1:]                                     # drop self
    purity = (ct_l[nbr] == ct_l[:, None]).mean(1)

    # ---- per-group anchor selection (all three conditions ANDed) ----
    is_anchor = np.zeros(len(ct_l), dtype=bool)
    rows = []
    for g in groups:
        m = ct_l == g
        gap_g = gap[m]
        cut = np.quantile(gap_g, 1 - TOP_PCT)           # top-% gap threshold
        top_pct = gap >= cut
        floor = gap > GAP_FLOOR
        pure = purity >= PURITY_MIN
        anchor_g = m & top_pct & floor & pure
        is_anchor |= anchor_g
        rows.append({
            "group": g,
            "n_provisional": int(m.sum()),
            "pass_top%": int((m & top_pct).sum()),
            "pass_gap>0": int((m & floor).sum()),
            "pass_purity": int((m & pure).sum()),
            "anchors": int(anchor_g.sum()),
            "%_of_group": round(100 * anchor_g.sum() / max(m.sum(), 1), 1),
            "gap_all": round(float(gap_g.mean()), 3),
            "gap_anchor": round(float(gap[anchor_g].mean()), 3) if anchor_g.any() else np.nan,
            "purity_all": round(float(purity[m].mean()), 3),
            "purity_anchor": round(float(purity[anchor_g].mean()), 3) if anchor_g.any() else np.nan,
        })

    summ = pd.DataFrame(rows).sort_values("anchors", ascending=False)
    print(f"anchor rule: marker-gate AND top {TOP_PCT:.0%} gap AND gap>{GAP_FLOOR} "
          f"AND kNN(k={KNN_K}) purity>={PURITY_MIN:.0%}\n")
    print(summ.to_string(index=False))
    n_lab, n_anc = int(labeled.sum()), int(is_anchor.sum())
    print(f"\ntotal: {n_anc:,} anchors / {n_lab:,} provisional "
          f"({100*n_anc/n_lab:.1f}%) / {adata.n_obs:,} non-tumor "
          f"({100*n_anc/adata.n_obs:.1f}%)")

    # ---- save anchors table (labeled cells with all metrics + flag) ----
    out = pd.DataFrame({
        "x": x_l, "y": y_l, "provisional_label": ct_l,
        "own_sim": own_sim.round(4), "nearest_other_sim": nearest_other.round(4),
        "gap": gap.round(4), "knn_purity": purity.round(4),
        "is_anchor": is_anchor,
    })
    out.to_csv(f"{OUT_DIR}/anchors.csv", index=False)
    summ.to_csv(f"{OUT_DIR}/anchors_summary.csv", index=False)

    # ---- spatial map: anchors coloured by group, non-anchor labeled grey ----
    fig, ax = plt.subplots(figsize=(9, 8), dpi=170)
    ax.scatter(x_l[~is_anchor], y_l[~is_anchor], s=1, c="#d8d8d8",
               linewidths=0, rasterized=True, label="non-anchor")
    for g in groups:
        m = (ct_l == g) & is_anchor
        if m.any():
            ax.scatter(x_l[m], y_l[m], s=3, c=CELLTYPE_COLORS.get(g, "#d62728"),
                       linewidths=0, rasterized=True, label=f"{g} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_1 high-confidence anchors (n={n_anc:,})")
    ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/anchors_spatial.png", bbox_inches="tight")
    plt.close()

    print(f"\nsaved anchors.csv, anchors_summary.csv, anchors_spatial.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
