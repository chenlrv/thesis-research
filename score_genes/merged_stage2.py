"""Stage 2 for the MERGED 5-type config (Myeloid, Vascular, Astrocytes, Ependymal,
Neurons), in ANNOTATION mode - keeps every cell, just adds a confidence flag.

  high_conf = (classifier label == marker label) AND (classifier prob >= 0.5)

The classifier is multiclass over the 5 types, trained on NON-(merged-)marker genes
(non-circular). prob>=0.5 = the winning class holds the majority of the 5-way mass.

Outputs:
  - high_confidence_labels.csv  (ALL provisional cells + confidence flag)
  - validation: per-type held-out AUC (type vs rest) on non-marker genes
  - subtype safety counts: BAM/MDM and pericyte/endothelial marker+ cells,
    provisional vs high-conf (to confirm no subtype is depleted)
  - clean_set.png : spatial map + PCA scatter of the high-confidence cells
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402
from run_score_genes_merged import MARKERS  # noqa: E402

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged"
PROB_MIN = 0.50
TOP_PCT, KNN_K, PURITY_MIN = 0.30, 30, 0.70

COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2"}
SUBTYPE = {  # within-group subtype markers, for the safety check
    "Myeloid": {"BAM": ["Lyve1", "Cd163", "Mrc1"], "MDM": ["Ccr2", "Plac8"]},
    "Vascular": {"Pericyte": ["Pdgfrb", "Rgs5", "Notch3"],
                 "Endothelial": ["Pecam1", "Cdh5", "Flt1"]},
}


def main():
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    sub = adata[labeled].copy()
    ct_l = ct[labeled]
    x_l, y_l = sub.obs["x"].to_numpy(), sub.obs["y"].to_numpy()
    groups = sorted(set(ct_l))
    var = set(sub.var_names)

    emb = ch.scaled_pca(adata)
    emb_l = emb[labeled]

    # ---- anchors (clean training seed): centroid gap + kNN purity ----
    cent = np.vstack([emb_l[ct_l == g].mean(0) for g in groups])
    Sm = cosine_similarity(emb_l, cent)
    gi = np.array([groups.index(g) for g in ct_l])
    own = Sm[np.arange(len(Sm)), gi]
    o = Sm.copy(); o[np.arange(len(Sm)), gi] = -np.inf
    gap = own - o.max(1)
    _, nbr = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb_l).kneighbors(emb_l)
    purity = (ct_l[nbr[:, 1:]] == ct_l[:, None]).mean(1)
    anchor = np.zeros(len(ct_l), bool)
    for g in groups:
        m = ct_l == g
        cut = np.quantile(gap[m], 1 - TOP_PCT)
        anchor |= m & (gap >= cut) & (gap > 0) & (purity >= PURITY_MIN)

    # ---- classifier on NON-(merged-)marker genes ----
    marker_union = {g for lst in MARKERS.values() for g in lst}
    indep = [g for g in sub.var_names if g not in marker_union]
    X = sub[:, indep].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X[anchor], ct_l[anchor])
    classes = list(clf.classes_)
    proba = clf.predict_proba(X)
    pred = np.array(classes)[proba.argmax(1)]
    pred_p = proba.max(1)

    # ---- confidence flag (keep ALL cells) ----
    high = (pred == ct_l) & (pred_p >= PROB_MIN)
    print(f"prob floor = {PROB_MIN};  marker<->classifier agreement = "
          f"{100*(pred==ct_l).mean():.1f}%")
    tab = pd.DataFrame({
        "provisional": pd.Series(ct_l).value_counts(),
        "high_conf": pd.Series(ct_l[high]).value_counts(),
    }).fillna(0).astype(int)
    tab["kept_%"] = (100 * tab["high_conf"] / tab["provisional"]).round(1)
    print("\nper-type provisional vs high-confidence:")
    print(tab.to_string())

    # ---- validation: per-type held-out AUC (type vs rest), non-marker genes ----
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print("\nvalidation - held-out AUC (type vs rest) on non-marker genes:")
    for g in groups:
        y = (ct_l == g).astype(int)
        p = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                              X, y, cv=cv, method="predict_proba")[:, 1]
        print(f"  {g:12s} vs rest: AUC = {roc_auc_score(y, p):.3f}")

    # ---- subtype safety counts (provisional vs high-conf) ----
    def expr_any(mask_cells, genes):
        genes = [gg for gg in genes if gg in var]
        if not genes or mask_cells.sum() == 0:
            return 0
        M = sub[mask_cells, genes].X
        M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        return int((M > 0).any(1).sum())

    print("\nsubtype safety - marker+ cell counts (provisional -> high-conf):")
    for parent, subs in SUBTYPE.items():
        for sname, genes in subs.items():
            prov_n = expr_any(ct_l == parent, genes)
            hc_n = expr_any((ct_l == parent) & high, genes)
            print(f"  {parent}/{sname:11s} ({'/'.join(genes)}): "
                  f"{prov_n} -> {hc_n}")

    out = pd.DataFrame({
        "x": x_l, "y": y_l, "provisional_label": ct_l,
        "predicted_label": pred, "predicted_proba": pred_p.round(4),
        "is_anchor": anchor, "high_conf": high,
    })
    out.to_csv(f"{OUT}/high_confidence_labels.csv", index=False)

    # ---- plot: clean set in tissue + PCA clusters ----
    with h5py.File(ch.SLICE, "r") as h5:
        xa = ch._read_obs_num(h5, "CenterX_global_px")
        ya = ch._read_obs_num(h5, "CenterY_global_px")
        tum = ch._read_obs_bool(h5, ch.TUMOR_COL)

    fig, ax = plt.subplots(1, 2, figsize=(17, 8), dpi=150)
    a = ax[0]
    a.scatter(xa[~tum], ya[~tum], s=0.6, c="#e6e6e6", linewidths=0, rasterized=True,
              label="other / low-conf")
    a.scatter(xa[tum], ya[tum], s=1.0, c="#f3c0c0", linewidths=0, rasterized=True,
              label="tumor")
    for g in groups:
        m = high & (ct_l == g)
        a.scatter(x_l[m], y_l[m], s=4, c=COL.get(g, "#333"), linewidths=0,
                  rasterized=True, label=f"{g} ({int(m.sum())})")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
    a.set_title("High-confidence cells in tissue", fontweight="bold")

    a = ax[1]
    for g in groups:
        m = high & (ct_l == g)
        a.scatter(emb_l[m, 0], emb_l[m, 1], s=5, c=COL.get(g, "#333"),
                  linewidths=0, alpha=0.6, rasterized=True, label=g)
    a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(markerscale=2, fontsize=9)
    a.set_title("High-confidence cells in PCA space", fontweight="bold")
    fig.suptitle("Stage 2 (merged) - the clean set", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUT}/clean_set.png", bbox_inches="tight")
    plt.close()

    print(f"\nsaved high_confidence_labels.csv ({int(high.sum()):,} high-conf "
          f"of {len(ct_l):,}) and clean_set.png -> {OUT}")


if __name__ == "__main__":
    main()
