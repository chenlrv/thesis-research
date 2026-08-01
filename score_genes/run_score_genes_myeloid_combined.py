"""Combined-Myeloid provisional annotation for slice 1.

Same score_genes recipe + FDR/margin gating as run_score_genes_slice1.py, but the
Microglia and Macrophage marker sets are MERGED into one "Myeloid" set (because
the two don't separate beyond their markers - AUC 0.59). Writes a cell_scores.csv
in the SAME format so the existing pipeline (check_homogeneity, select_anchors,
build_highconf, storyline) can consume it unchanged.

Output -> score_genes_slice1_myeloid/cell_scores.csv
"""
import os
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls, MARGIN_RATIO)
from annotation_cell_marker_genes import MARKER_GENES as SPLIT  # noqa: E402

OUT_DIR = "D:/thesis-research/score_genes_slice1_myeloid"
FDR_CUTOFF = 0.05

# PAN-myeloid markers: genes ALL myeloid cells (microglia + MDM + BAM) express but
# the other lineages (neurons/astro/vascular/ependymal) do not. Using subtype-
# specific markers merged would DILUTE the score (microglia don't fire macrophage
# genes and vice-versa) and under-call myeloid - so we use shared myeloid identity
# genes instead. All confirmed present on the panel.
MYELOID = ["Csf1r", "C1qa", "C1qb", "C1qc", "Aif1", "Cd68",
           "Itgam", "Ptprc", "Tyrobp", "Fcer1g", "Trem2", "Cx3cr1"]

MARKERS = {
    "Myeloid": MYELOID,
    "Astrocytes": SPLIT["Astrocytes"],
    "Endothelial": SPLIT["Endothelial"],
    "Pericytes": SPLIT["Pericytes"],
    "Ependymal": SPLIT["Ependymal"],
    "Neurons": SPLIT["Neurons"],
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    adata = adata[~tumor].copy()
    cx, cy = cx[~tumor], cy[~tumor]
    print(f"non-tumor cells = {adata.n_obs} (removed {int(tumor.sum())} tumor)")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = list(MARKERS.keys())
    score_cols = {}
    print("\nscoring (combined Myeloid):")
    for label in labels:
        genes = [g for g in MARKERS[label] if g in adata.var_names]
        missing = [g for g in MARKERS[label] if g not in adata.var_names]
        col = "score_" + label.replace(" ", "_")
        sc.tl.score_genes(adata, gene_list=genes, score_name=col,
                          ctrl_size=50, n_bins=25)
        score_cols[label] = col
        print(f"  {label:12s}: {len(genes)} genes" +
              (f"  (missing {missing})" if missing else ""))

    S = adata.obs[[score_cols[l] for l in labels]].copy()
    S.columns = labels

    thresholds = {}
    for label in labels:
        t, *_ = mirrored_fdr_threshold(S[label].to_numpy(), fdr=FDR_CUTOFF)
        thresholds[label] = t

    res = scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO)
    calls = res["calls"]

    out = pd.DataFrame({"x": cx, "y": cy})
    for l in labels:
        out["score_" + l.replace(" ", "_")] = S[l].to_numpy()
    out["top_score_label"] = res["top_label"]
    out["second_label"] = res["second_label"]
    out["top_scaled"] = res["top_scaled"]
    out["second_scaled"] = res["second_scaled"]
    out["margin_scaled"] = res["margin_scaled"]
    out["fdr_pass"] = res["fdr_pass"]
    out["margin_pass"] = res["margin_pass"]
    out["celltype"] = calls
    out.to_csv(f"{OUT_DIR}/cell_scores.csv", index=False)

    print(f"\nannotated: {int((calls != 'unknown').sum()):,}  "
          f"unknown: {int((calls == 'unknown').sum()):,}")
    print("\nfinal labels:")
    print(pd.Series(calls).value_counts().to_string())
    print(f"\nsaved -> {OUT_DIR}/cell_scores.csv")


if __name__ == "__main__":
    main()
