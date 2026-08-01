"""highconf_homogeneity.py - are the HIGH-CONFIDENCE Astrocyte / Ependymal cells
homogeneous? Re-runs the two homogeneity checks (sub-clustering + intra/inter
similarity) on the high_confidence_labels.csv set instead of the provisional set.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

PROVISIONAL_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
HC_CSV = "D:/thesis-research/score_genes_slice1/high_confidence_labels.csv"

ch.SCORES_CSV = PROVISIONAL_CSV
ch.OUT_DIR = "D:/thesis-research/score_genes_slice1"
ch.MIN_CELLS = 50          # high-conf astro set is ~87, allow it


def main():
    adata = ch.load_adata_with_calls()
    labeled = np.asarray(adata.obs["celltype"]) != "unknown"
    sub = adata[labeled].copy()

    hc = pd.read_csv(HC_CSV)
    assert np.allclose(hc["x"].to_numpy(), sub.obs["x"].to_numpy(), atol=1e-3), \
        "high_confidence_labels.csv not aligned to labeled adata order"

    sub.obs["celltype"] = np.where(
        hc["high_conf"].to_numpy(), hc["provisional_label"].to_numpy(), "unknown")
    hcsub = sub[sub.obs["celltype"] != "unknown"].copy()
    groups_all = sorted(set(hcsub.obs["celltype"]))
    print("high-confidence group sizes:")
    print(hcsub.obs["celltype"].value_counts().to_string())

    # Check 1: do the high-conf Astrocyte / Ependymal groups fragment?
    ch.subcluster_test(hcsub, ["Astrocytes", "Ependymal"])
    # Check 3: are they separated from every other high-conf group?
    ch.similarity_test(hcsub, groups_all)


if __name__ == "__main__":
    main()
