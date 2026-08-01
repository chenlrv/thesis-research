"""build_highconf.py - grow the high-confidence set without losing quality, by
requiring two INDEPENDENT signals to agree.

The pair-aware anchors are starved for the cleanly-separable types (Astrocytes 56,
Ependymal 215) only because they were picked with the weak global-centroid metric
- which the held-out test proved understates their separation (AUC 1.0). Most of
the 2,378 / 878 provisional astro/ependymal are genuinely good.

Method - "agreement" high-confidence:
  1. seed = current pair-aware anchors (their provisional labels).
  2. train a multiclass classifier on NON-marker genes (markers excluded ->
     independent of what defined the labels).
  3. predict on every provisional-labeled cell.
  4. keep a cell as high-confidence if  provisional_label == predicted_label
     AND predicted probability >= T  (plus all seed anchors).

This grows the separable types (the classifier confidently confirms them) and
does NOT inflate the non-separable myeloid/vascular pairs (the classifier can't
confirm them on independent genes) - so quality is preserved by construction.
A threshold sweep + a held-out re-validation show the trade-off.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch          # noqa: E402
from annotation_cell_marker_genes import MARKER_GENES  # noqa: E402

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
ANCHORS_CSV = "D:/thesis-research/score_genes_slice1/anchors_pairwise.csv"
OUT_DIR = "D:/thesis-research/score_genes_slice1"
SAVE_T = 0.80                      # threshold used for the saved label set
SWEEP = [0.50, 0.70, 0.80, 0.90]


def main():
    ch.SCORES_CSV = PROVISIONAL_CSV
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    sub = adata[labeled].copy()

    anc = pd.read_csv(ANCHORS_CSV)
    prov = anc["provisional_label"].to_numpy()
    seed = anc["is_anchor"].to_numpy()

    # independent features: every gene EXCEPT the scoring markers
    marker_union = {g for lst in MARKER_GENES.values() for g in lst}
    indep = [g for g in sub.var_names if g not in marker_union]
    X = sub[:, indep].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    # ---- train on seed anchors, predict honestly on all provisional cells ----
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(X[seed], prov[seed])
    classes = list(clf.classes_)

    # seed cells: out-of-fold proba (don't trust the model on its own training set)
    proba = clf.predict_proba(X)                       # non-seed: honest (unseen)
    oof = cross_val_predict(clf, X[seed], prov[seed],
                            cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")
    proba[seed] = oof

    pred = np.array(classes)[proba.argmax(1)]
    pred_p = proba.max(1)
    agree = pred == prov

    groups = sorted(set(prov))
    print(f"seed anchors: {int(seed.sum()):,}  |  provisional cells: {len(prov):,}")
    print(f"overall marker<->independent-genes agreement: "
          f"{100*agree.mean():.1f}%\n")

    # ---- threshold sweep ----
    print("high-confidence counts per type (seed OR agree & proba>=T):")
    header = "  type         provis   seed " + " ".join(f"T={t:.2f}" for t in SWEEP)
    print(header)
    for g in groups:
        m = prov == g
        line = f"  {g:12s} {int(m.sum()):6d} {int((m&seed).sum()):6d} "
        for t in SWEEP:
            hc = m & (seed | (agree & (pred_p >= t)))
            line += f" {int(hc.sum()):6d}"
        print(line)
    tot = "  {:12s} {:6d} {:6d} ".format("TOTAL", len(prov), int(seed.sum()))
    for t in SWEEP:
        hc = seed | (agree & (pred_p >= t))
        tot += f" {int(hc.sum()):6d}"
    print(tot)

    # ---- save labels at SAVE_T + re-validate quality on the enlarged set ----
    high_conf = seed | (agree & (pred_p >= SAVE_T))
    out = pd.DataFrame({
        "x": anc["x"], "y": anc["y"], "provisional_label": prov,
        "predicted_label": pred, "predicted_proba": pred_p.round(4),
        "is_seed_anchor": seed, "high_conf": high_conf,
        "source": np.where(seed, "seed", np.where(high_conf, "grown", "dropped")),
    })
    out.to_csv(f"{OUT_DIR}/high_confidence_labels.csv", index=False)

    print(f"\n=== quality re-check at T={SAVE_T} (held-out AUC on NON-marker genes, "
          f"enlarged sets) ===")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for A, B in [("Astrocytes", "Ependymal"), ("Microglia", "Macrophage"),
                 ("Pericytes", "Endothelial")]:
        mA = (prov == A) & high_conf
        mB = (prov == B) & high_conf
        m = mA | mB
        y = mA[m].astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            print(f"  {A} ({int(mA.sum())}) vs {B} ({int(mB.sum())}): too few")
            continue
        p = cross_val_predict(clf.__class__(max_iter=3000, class_weight="balanced"),
                              X[m], y, cv=cv, method="predict_proba")[:, 1]
        print(f"  {A} ({int(mA.sum())}) vs {B} ({int(mB.sum())}): "
              f"held-out AUC = {roc_auc_score(y, p):.3f}")

    n_hc = int(high_conf.sum())
    print(f"\nsaved high_confidence_labels.csv ({n_hc:,} high-conf cells at "
          f"T={SAVE_T}, vs {int(seed.sum()):,} seed anchors) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
