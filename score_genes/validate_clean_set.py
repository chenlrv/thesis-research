"""Validate the CLEAN (high-confidence) merged set: held-out AUC (type vs rest)
on non-marker genes, using ONLY the high_conf cells - the test that was missing.
Compares against the same metric on all provisional cells.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402
from run_score_genes_merged import MARKERS  # noqa: E402

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"


def auc_vs_rest(X, labels, groups, cv):
    out = {}
    for g in groups:
        y = (labels == g).astype(int)
        if y.sum() < 10 or (1 - y).sum() < 10:
            out[g] = np.nan
            continue
        p = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                              X, y, cv=cv, method="predict_proba")[:, 1]
        out[g] = roc_auc_score(y, p)
    return out


def main():
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()
    labeled = np.asarray(adata.obs["celltype"]) != "unknown"
    sub = adata[labeled].copy()

    hc = pd.read_csv(HC)
    assert np.allclose(hc["x"].to_numpy(), sub.obs["x"].to_numpy(), atol=1e-3)
    prov = hc["provisional_label"].to_numpy()
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()

    marker_union = {g for lst in MARKERS.values() for g in lst}
    indep = [g for g in sub.var_names if g not in marker_union]
    X = sub[:, indep].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    groups = sorted(set(prov))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    full = auc_vs_rest(X, prov, groups, cv)
    clean = auc_vs_rest(X[high], prov[high], groups, cv)

    print(f"{'type':12s} {'n_prov':>7s} {'n_clean':>8s} {'AUC_all':>8s} {'AUC_clean':>10s}")
    for g in groups:
        nprov = int((prov == g).sum())
        nclean = int(((prov == g) & high).sum())
        print(f"{g:12s} {nprov:7d} {nclean:8d} {full[g]:8.3f} {clean[g]:10.3f}")

    # key pairwise on clean set (compare to earlier split astro-vs-ependymal = 1.0)
    print("\npairwise on clean set (non-marker genes):")
    for a, b in [("Astrocytes", "Ependymal"), ("Myeloid", "Vascular"),
                 ("Myeloid", "Neurons")]:
        m = high & ((prov == a) | (prov == b))
        y = (prov[m] == a).astype(int)
        p = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                              X[m], y, cv=cv, method="predict_proba")[:, 1]
        print(f"  {a} vs {b}: AUC = {roc_auc_score(y, p):.3f}")


if __name__ == "__main__":
    main()
