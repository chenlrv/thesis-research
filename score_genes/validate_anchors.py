"""validate_anchors.py - independent checks that the pair-aware anchors are real,
not just tautologies of the marker genes that defined them.

(1) single-cell co-expression - at the per-cell count level (not the aggregate
    score), do anchors express their OWN lineage markers and NOT the competitor's?
    The non-circular payoff is the double-positive rate: cells with a positive
    score gap can still co-express both lineages (doublets/ambient) - that's
    invisible to the gap but visible here.

(2) held-out classifier on NON-marker genes - train logistic regression to
    separate the two confused anchor groups using every gene EXCEPT the marker
    genes used for scoring. If anchors are a genuine biological split they remain
    separable on independent genes (high CV AUC); if they're marker tautologies,
    accuracy collapses toward chance.

Reads anchors_pairwise.csv (labeled-cell order) + the log-normalized h5ad.
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
PAIRS = [("Microglia", "Macrophage"), ("Pericytes", "Endothelial")]


def main():
    ch.SCORES_CSV = PROVISIONAL_CSV
    adata = ch.load_adata_with_calls()           # log-normalized X
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"
    sub = adata[labeled].copy()

    anc = pd.read_csv(ANCHORS_CSV)
    if len(anc) != sub.n_obs:
        raise SystemExit(f"row mismatch: anchors {len(anc)} vs labeled {sub.n_obs}")
    sub.obs["is_anchor"] = anc["is_anchor"].to_numpy()
    sub.obs["prov"] = anc["provisional_label"].to_numpy()

    var = set(sub.var_names)

    def detect_frac(cells, genes):
        """per-cell: does the cell express >=1 of these genes (log-norm > 0)?"""
        genes = [g for g in genes if g in var]
        if not genes:
            return np.zeros(cells.n_obs, dtype=bool)
        X = cells[:, genes].X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        return (X > 0).any(1)

    # ---- (1) single-cell co-expression ----
    print("=" * 78)
    print("CHECK 1 - single-cell co-expression of anchors (own vs competitor markers)")
    print("=" * 78)
    rows = []
    for A, B in PAIRS:
        for g, comp in ((A, B), (B, A)):
            cells = sub[(sub.obs["prov"] == g) & sub.obs["is_anchor"]].copy()
            if cells.n_obs == 0:
                continue
            own = detect_frac(cells, MARKER_GENES[g])
            other = detect_frac(cells, MARKER_GENES[comp])
            rows.append({
                "anchor_group": g, "competitor": comp, "n": cells.n_obs,
                "%_express_own": round(100 * own.mean(), 1),
                "%_express_comp": round(100 * other.mean(), 1),
                "%_double_pos": round(100 * (own & other).mean(), 1),
                "%_own_only": round(100 * (own & ~other).mean(), 1),
                "%_neither": round(100 * (~own & ~other).mean(), 1),
            })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n(want: high %_own_only, low %_double_pos. %_neither = anchored on the "
          "score but no marker detected at single-cell level.)")

    # ---- (2) held-out classifier on NON-marker genes ----
    marker_union = {g for lst in MARKER_GENES.values() for g in lst}
    indep_genes = [g for g in sub.var_names if g not in marker_union]
    print("\n" + "=" * 78)
    print(f"CHECK 2 - logistic-regression separability on {len(indep_genes)} "
          f"NON-marker genes ({len(marker_union)} markers excluded)")
    print("=" * 78)
    Xall = sub[:, indep_genes].X
    Xall = Xall.toarray() if hasattr(Xall, "toarray") else np.asarray(Xall)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for A, B in PAIRS:
        m = ((sub.obs["prov"] == A) | (sub.obs["prov"] == B)).to_numpy() \
            & sub.obs["is_anchor"].to_numpy()
        y = (sub.obs["prov"][m] == A).to_numpy().astype(int)   # 1 = A
        X = Xall[m]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y, proba)
        bacc = balanced_accuracy_score(y, pred)
        # null: shuffled labels
        rng = np.random.default_rng(0)
        y_perm = rng.permutation(y)
        auc_null = roc_auc_score(
            y_perm, cross_val_predict(clf, X, y_perm, cv=cv,
                                      method="predict_proba")[:, 1])
        # top independent genes by |coef| (fit on all)
        clf.fit(X, y)
        coef = clf.coef_[0]
        top_A = [indep_genes[i] for i in np.argsort(coef)[::-1][:6]]
        top_B = [indep_genes[i] for i in np.argsort(coef)[:6]]
        print(f"\n{A} (n={int(y.sum())}) vs {B} (n={int((1-y).sum())}):")
        print(f"  CV AUC = {auc:.3f}   balanced acc = {bacc:.3f}   "
              f"(shuffled-label AUC = {auc_null:.3f})")
        print(f"  up in {A}: {', '.join(top_A)}")
        print(f"  up in {B}: {', '.join(top_B)}")


if __name__ == "__main__":
    main()
