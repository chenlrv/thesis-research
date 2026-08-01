"""select_anchors_pairwise.py - anchors with a pair-aware criterion for the
confused lineages.

The global-centroid version (select_anchors.py) gives Macrophage ZERO anchors,
because in global PCA no macrophage is closer to its own centroid than to the
Microglia centroid, and none has a macrophage-pure neighborhood. The fix: for the
two confused pairs, judge separation on the pair's OWN differential axis instead
of the global centroid.

Confused pairs:  Microglia <-> Macrophage,  Pericytes <-> Endothelial.

For a cell of type A whose competitor is B, condition (2) and (3) become:
  (2) pairwise gap  = score_A - score_B  (from cell_scores.csv) > GAP_FLOOR
                      AND in the top TOP_PCT within group A.
  (3) pair purity   = >= PURITY_MIN of its K nearest neighbours - taken among
                      provisional A+B cells in a PCA built ONLY on the A&B marker
                      genes - carry label A. (Global-PCA purity is unusable here:
                      macrophages have no macrophage-pure neighbourhood there.)

Clean types (Astrocytes, Ependymal, Neurons) keep the global-centroid gap +
global-PCA purity from select_anchors.py. Condition (1) marker gate = provisional
label != 'unknown' for all.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch          # noqa: E402  (loader + scaled_pca reuse)
from annotation_cell_marker_genes import MARKER_GENES  # noqa: E402

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"  # baseline
OUT_DIR = "D:/thesis-research/score_genes_slice1"

TOP_PCT = 0.30
GAP_FLOOR = 0.0
KNN_K = 30
PURITY_MIN = 0.70
PAIR_PCS = 15

# confused pairs (each label -> its competitor)
COMPETITOR = {
    "Microglia": "Macrophage", "Macrophage": "Microglia",
    "Pericytes": "Endothelial", "Endothelial": "Pericytes",
}
PAIRS = [("Microglia", "Macrophage"), ("Pericytes", "Endothelial")]

CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4", "Microglia": "#17becf", "Macrophage": "#00a087",
    "Endothelial": "#2ca02c", "Pericytes": "#a65628", "Ependymal": "#984ea3",
    "Neurons": "#e377c2",
}


def knn_purity(emb, labels, k=KNN_K):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, nbr = nn.kneighbors(emb)
    return (labels[nbr[:, 1:]] == labels[:, None]).mean(1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ch.SCORES_CSV = PROVISIONAL_CSV
    adata = ch.load_adata_with_calls()
    df = pd.read_csv(PROVISIONAL_CSV)               # same row order (validated by ch)

    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    ct_l = ct[labeled]
    x_l = adata.obs["x"].to_numpy()[labeled]
    y_l = adata.obs["y"].to_numpy()[labeled]
    groups = sorted(set(ct_l))
    clean = [g for g in groups if g not in COMPETITOR]

    # per-label raw scores (labeled subset), for the pairwise gap
    score = {g: df["score_" + g].to_numpy()[labeled] for g in groups}

    # ---- global centroid gap + global purity (for clean types) ----
    emb = ch.scaled_pca(adata)
    emb_l = emb[labeled]
    centroids = np.vstack([emb_l[ct_l == g].mean(0) for g in groups])
    sims = cosine_similarity(emb_l, centroids)
    own_i = np.array([groups.index(g) for g in ct_l])
    own_sim = sims[np.arange(len(sims)), own_i]
    other = sims.copy(); other[np.arange(len(sims)), own_i] = -np.inf
    gap_global = own_sim - other.max(1)
    purity_global = knn_purity(emb_l, ct_l)

    # ---- pairwise gap + pair-space purity (for confused pairs) ----
    pairwise_gap = np.full(len(ct_l), np.nan)
    pair_purity = np.full(len(ct_l), np.nan)
    for A, B in PAIRS:
        for g, comp in ((A, B), (B, A)):
            mg = ct_l == g
            pairwise_gap[mg] = score[g][mg] - score[comp][mg]
        pm = (ct_l == A) | (ct_l == B)
        genes = [x for x in MARKER_GENES[A] + MARKER_GENES[B] if x in adata.var_names]
        sub = adata[labeled][pm][:, genes].copy()    # log-norm marker expression
        sc.pp.scale(sub, max_value=10)
        sc.pp.pca(sub, n_comps=int(min(PAIR_PCS, len(genes) - 1, sub.n_obs - 1)),
                  random_state=0)
        pair_purity[pm] = knn_purity(sub.obsm["X_pca"], ct_l[pm])

    # ---- selection (3-condition AND, per-group criterion) ----
    is_anchor = np.zeros(len(ct_l), dtype=bool)
    rows = []
    for g in groups:
        m = ct_l == g
        if g in clean:
            gp, pur, kind = gap_global, purity_global, "centroid"
        else:
            gp, pur, kind = pairwise_gap, pair_purity, "pairwise"
        cut = np.quantile(gp[m], 1 - TOP_PCT)
        sel = m & (gp >= cut) & (gp > GAP_FLOOR) & (pur >= PURITY_MIN)
        is_anchor |= sel
        rows.append({
            "group": g, "criterion": kind, "n_provisional": int(m.sum()),
            "pass_top%": int((m & (gp >= cut)).sum()),
            "pass_gap>0": int((m & (gp > GAP_FLOOR)).sum()),
            "pass_purity": int((m & (pur >= PURITY_MIN)).sum()),
            "anchors": int(sel.sum()),
            "%_of_group": round(100 * sel.sum() / max(m.sum(), 1), 1),
            "gap_anchor": round(float(gp[sel].mean()), 3) if sel.any() else np.nan,
            "purity_anchor": round(float(pur[sel].mean()), 3) if sel.any() else np.nan,
        })

    summ = pd.DataFrame(rows).sort_values("anchors", ascending=False)
    print(f"rule: marker-gate AND top {TOP_PCT:.0%} gap AND gap>{GAP_FLOOR} "
          f"AND purity(k={KNN_K})>={PURITY_MIN:.0%}; "
          f"confused pairs use pairwise gap + pair-space purity\n")
    print(summ.to_string(index=False))
    n_lab, n_anc = int(labeled.sum()), int(is_anchor.sum())
    print(f"\ntotal: {n_anc:,} anchors / {n_lab:,} provisional "
          f"({100*n_anc/n_lab:.1f}%) / {adata.n_obs:,} non-tumor "
          f"({100*n_anc/adata.n_obs:.1f}%)")

    out = pd.DataFrame({
        "x": x_l, "y": y_l, "provisional_label": ct_l,
        "gap_global": gap_global.round(4), "purity_global": purity_global.round(4),
        "pairwise_gap": pairwise_gap.round(4), "pair_purity": pair_purity.round(4),
        "is_anchor": is_anchor,
    })
    out.to_csv(f"{OUT_DIR}/anchors_pairwise.csv", index=False)
    summ.to_csv(f"{OUT_DIR}/anchors_pairwise_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=170)
    ax.scatter(x_l[~is_anchor], y_l[~is_anchor], s=1, c="#d8d8d8",
               linewidths=0, rasterized=True, label="non-anchor")
    for g in groups:
        mm = (ct_l == g) & is_anchor
        if mm.any():
            ax.scatter(x_l[mm], y_l[mm], s=3, c=CELLTYPE_COLORS.get(g, "#d62728"),
                       linewidths=0, rasterized=True, label=f"{g} ({int(mm.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_1 pair-aware anchors (n={n_anc:,})")
    ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/anchors_pairwise_spatial.png", bbox_inches="tight")
    plt.close()
    print(f"\nsaved anchors_pairwise.csv, anchors_pairwise_summary.csv, "
          f"anchors_pairwise_spatial.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
