"""validate_other_types.py - same two checks as validate_anchors.py, applied to
the non-myeloid/non-vascular types.

Main remaining confusion among these is Astrocyte <-> Ependymal (astrocytes bleed
toward the ependymal/ventricular-lining core). Neurons have only 7 anchors - too
few for a held-out classifier - so the Neurons hidden-gene test is run on the
PROVISIONAL neuron calls (vs Ependymal, their nearest other), clearly flagged.

(1) single-cell co-expression of anchors (own vs competitor markers)
(2) logistic regression on NON-marker genes (markers excluded -> non-circular)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch          # noqa: E402
from annotation_cell_marker_genes import MARKER_GENES  # noqa: E402

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
ANCHORS_CSV = "D:/thesis-research/score_genes_slice1/anchors_pairwise.csv"

# (group, competitor) for the co-expression check (anchors)
COEXPR = [("Astrocytes", "Ependymal"), ("Ependymal", "Astrocytes"),
          ("Neurons", "Ependymal")]


def main():
    ch.SCORES_CSV = PROVISIONAL_CSV
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    sub = adata[labeled].copy()

    anc = pd.read_csv(ANCHORS_CSV)
    sub.obs["is_anchor"] = anc["is_anchor"].to_numpy()
    sub.obs["prov"] = anc["provisional_label"].to_numpy()
    var = set(sub.var_names)

    def detect(cells, genes):
        genes = [g for g in genes if g in var]
        if not genes:
            return np.zeros(cells.n_obs, dtype=bool)
        X = cells[:, genes].X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return (X > 0).any(1)

    # ---- (1) co-expression ----
    print("=" * 78)
    print("CHECK 1 - co-expression of anchors (own vs competitor markers)")
    print("=" * 78)
    rows = []
    for g, comp in COEXPR:
        cells = sub[(sub.obs["prov"] == g) & sub.obs["is_anchor"]].copy()
        if cells.n_obs == 0:
            rows.append({"anchor_group": g, "competitor": comp, "n": 0})
            continue
        own, other = detect(cells, MARKER_GENES[g]), detect(cells, MARKER_GENES[comp])
        rows.append({
            "anchor_group": g, "competitor": comp, "n": cells.n_obs,
            "%_express_own": round(100 * own.mean(), 1),
            "%_express_comp": round(100 * other.mean(), 1),
            "%_double_pos": round(100 * (own & other).mean(), 1),
            "%_own_only": round(100 * (own & ~other).mean(), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- (2) classifier on NON-marker genes ----
    marker_union = {g for lst in MARKER_GENES.values() for g in lst}
    indep = [g for g in sub.var_names if g not in marker_union]
    Xall = sub[:, indep].X
    Xall = Xall.toarray() if hasattr(Xall, "toarray") else np.asarray(Xall)
    prov = sub.obs["prov"].to_numpy()
    isanc = sub.obs["is_anchor"].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def run(name, mA, mB, A):
        m = mA | mB
        y = mA[m].astype(int)
        X = Xall[m]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
        bacc = balanced_accuracy_score(y, (proba >= 0.5).astype(int))
        rng = np.random.default_rng(0)
        yp = rng.permutation(y)
        auc0 = roc_auc_score(yp, cross_val_predict(clf, X, yp, cv=cv,
                                                   method="predict_proba")[:, 1])
        clf.fit(X, y)
        co = clf.coef_[0]
        up = [indep[i] for i in np.argsort(co)[::-1][:6]]
        dn = [indep[i] for i in np.argsort(co)[:6]]
        print(f"\n{name}:  CV AUC = {auc:.3f}   balanced acc = {bacc:.3f}   "
              f"(shuffled = {auc0:.3f})")
        print(f"  up: {', '.join(up)}")
        print(f"  dn: {', '.join(dn)}")

    print("\n" + "=" * 78)
    print(f"CHECK 2 - separability on {len(indep)} NON-marker genes")
    print("=" * 78)
    # Astrocyte vs Ependymal on anchors
    run("Astrocytes(anchor) vs Ependymal(anchor)",
        (prov == "Astrocytes") & isanc, (prov == "Ependymal") & isanc, "Astrocytes")
    # Neurons too few anchors -> provisional calls, flagged
    print("\n[Neurons: only 7 anchors -> test on PROVISIONAL calls, not anchors]")
    run("Neurons(provisional) vs Ependymal(provisional)",
        prov == "Neurons", prov == "Ependymal", "Neurons")


if __name__ == "__main__":
    main()
