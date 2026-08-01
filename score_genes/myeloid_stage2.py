"""Stage 2 (+ Stage 3 gap) for the COMBINED-Myeloid annotation.

Same two-witness method as the split storyline, but:
  - provisional labels come from the combined-Myeloid run (6 types),
  - the classifier's features exclude the COMBINED marker genes (pan-myeloid +
    the 5 other lineages) so it stays non-circular,
  - validation focuses on Myeloid-vs-rest (and Myeloid vs each lineage).

Steps:
  1. anchors (seed) per type: marker gate + top-30% centroid gap (>0) + kNN purity.
  2. train multiclass logistic regression on NON-marker genes using the seed.
  3. keep a cell high-confidence if marker label == classifier label & prob >= 0.8.
  4. validate: held-out AUC (Myeloid vs rest / vs each lineage) on non-marker genes.
  5. Stage-3 gap: high-confidence Myeloid separation vs provisional.
Output -> score_genes_slice1_myeloid/high_confidence_labels.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402
from run_score_genes_myeloid_combined import MARKERS  # noqa: E402

PROV = "D:/thesis-research/score_genes_slice1_myeloid/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1_myeloid"
TOP_PCT, GAP_FLOOR, KNN_K, PURITY_MIN, SAVE_T = 0.30, 0.0, 30, 0.70, 0.80


def main():
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    sub = adata[labeled].copy()
    ct_l = ct[labeled]
    groups = sorted(set(ct_l))

    emb = ch.scaled_pca(adata)
    emb_l = emb[labeled]

    # ---- 1. anchors: global centroid gap + kNN purity ----
    cent = np.vstack([emb_l[ct_l == g].mean(0) for g in groups])
    S = cosine_similarity(emb_l, cent)
    gi = np.array([groups.index(g) for g in ct_l])
    own = S[np.arange(len(S)), gi]
    o = S.copy(); o[np.arange(len(S)), gi] = -np.inf
    gap = own - o.max(1)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(emb_l)
    _, nbr = nn.kneighbors(emb_l)
    purity = (ct_l[nbr[:, 1:]] == ct_l[:, None]).mean(1)

    anchor = np.zeros(len(ct_l), bool)
    for g in groups:
        m = ct_l == g
        cut = np.quantile(gap[m], 1 - TOP_PCT)
        anchor |= m & (gap >= cut) & (gap > GAP_FLOOR) & (purity >= PURITY_MIN)
    print("anchors per type:")
    print(pd.Series(ct_l[anchor]).value_counts().to_string())

    # ---- 2. classifier on NON-(combined)marker genes ----
    marker_union = {g for lst in MARKERS.values() for g in lst}
    indep = [g for g in sub.var_names if g not in marker_union]
    X = sub[:, indep].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    print(f"\nclassifier features: {len(indep)} non-marker genes "
          f"({len(marker_union)} markers excluded)")
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(X[anchor], ct_l[anchor])
    classes = list(clf.classes_)
    proba = clf.predict_proba(X)
    pred = np.array(classes)[proba.argmax(1)]
    pred_p = proba.max(1)
    agree = pred == ct_l

    # ---- 3. high-confidence = seed OR (agree & prob>=T) ----
    high = anchor | (agree & (pred_p >= SAVE_T))
    print(f"\nmarker<->classifier agreement: {100*agree.mean():.1f}%")
    print("high-confidence per type (T=%.2f):" % SAVE_T)
    print(pd.Series(ct_l[high]).value_counts().to_string())

    # ---- 4. validation: held-out AUC on non-marker genes ----
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    print("\n=== validation: held-out AUC on non-marker genes (high-conf cells) ===")
    mye = (ct_l == "Myeloid") & high
    rest = (ct_l != "Myeloid") & high
    y = np.r_[np.ones(mye.sum()), np.zeros(rest.sum())].astype(int)
    Xv = np.vstack([X[mye], X[rest]])
    p = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                          Xv, y, cv=cv, method="predict_proba")[:, 1]
    print(f"  Myeloid ({int(mye.sum())}) vs REST ({int(rest.sum())}): "
          f"AUC = {roc_auc_score(y, p):.3f}")
    for g in [g for g in groups if g != "Myeloid"]:
        mg = (ct_l == g) & high
        if mg.sum() < 10:
            continue
        yy = np.r_[np.ones(mye.sum()), np.zeros(mg.sum())].astype(int)
        XX = np.vstack([X[mye], X[mg]])
        pp = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                               XX, yy, cv=cv, method="predict_proba")[:, 1]
        print(f"  Myeloid vs {g:12s} ({int(mg.sum())}): AUC = {roc_auc_score(yy, pp):.3f}")

    # ---- 5. Stage-3 gap: high-conf Myeloid vs provisional ----
    cent_hc = {g: emb_l[(ct_l == g) & high].mean(0) for g in groups
               if ((ct_l == g) & high).sum() > 0}
    gl = list(cent_hc)
    Sh = cosine_similarity(emb_l, np.vstack([cent_hc[g] for g in gl]))
    gih = np.array([gl.index(g) if g in gl else -1 for g in ct_l])
    mye_hc = (ct_l == "Myeloid") & high
    ownh = Sh[np.arange(len(Sh)), gih]
    oh = Sh.copy(); oh[np.arange(len(Sh)), gih] = -np.inf
    gaph = ownh - oh.max(1)
    print("\n=== Stage 3: Myeloid separation gap ===")
    print(f"  provisional Myeloid: {gap[ct_l=='Myeloid'].mean():+.3f}")
    print(f"  high-confidence Myeloid: {gaph[mye_hc].mean():+.3f}")

    out = pd.DataFrame({
        "x": sub.obs["x"].to_numpy(), "y": sub.obs["y"].to_numpy(),
        "provisional_label": ct_l, "predicted_label": pred,
        "predicted_proba": pred_p.round(4), "is_seed_anchor": anchor,
        "high_conf": high,
    })
    out.to_csv(f"{OUT}/high_confidence_labels.csv", index=False)
    print(f"\nsaved -> {OUT}/high_confidence_labels.csv "
          f"({int(high.sum()):,} high-confidence cells)")


if __name__ == "__main__":
    main()
