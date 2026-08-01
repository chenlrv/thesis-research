"""Merged provisional annotation for slice 1: Myeloid + Vascular both combined.

5 types: Myeloid (pan-myeloid), Vascular (Pericytes + Endothelial), Astrocytes,
Ependymal, Neurons. Same score_genes + FDR/margin gating. Vascular uses the union
of the pericyte + endothelial marker sets (justified here because pericytes
co-express endothelial markers ~78% from spillover, so the union fires for both).

Output -> score_genes_slice1_merged/cell_scores.csv
"""
import os
import sys

import anndata as ad
import h5py
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls, MARGIN_RATIO)
from annotation_cell_marker_genes import MARKER_GENES as SPLIT  # noqa: E402

OUT_DIR = "D:/thesis-research/score_genes_slice1_merged"
FDR_CUTOFF = 0.05

# Lean pan-myeloid set (the Jun 6 markers): specific myeloid identity genes WITHOUT
# the ultra-high-expression complement genes (C1qa/b/c, Cd68) that widen the
# score_genes null and make the FDR gate over-conservative (which under-called
# Myeloid to ~3k). These 5 give better coverage at equal specificity.
MYELOID = ["Csf1r", "Tyrobp", "Fcer1g", "Aif1", "Cx3cr1"]

MARKERS = {
    "Myeloid": MYELOID,
    "Vascular": SPLIT["Pericytes"] + SPLIT["Endothelial"],
    "Astrocytes": SPLIT["Astrocytes"],
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
    print(f"non-tumor cells = {adata.n_obs}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = list(MARKERS.keys())
    print("\nscoring (merged):")
    for label in labels:
        genes = [g for g in MARKERS[label] if g in adata.var_names]
        col = "score_" + label
        sc.tl.score_genes(adata, gene_list=genes, score_name=col,
                          ctrl_size=50, n_bins=25)
        print(f"  {label:12s}: {len(genes)} genes")

    S = adata.obs[["score_" + l for l in labels]].copy()
    S.columns = labels
    thresholds = {l: mirrored_fdr_threshold(S[l].to_numpy(), fdr=FDR_CUTOFF)[0]
                  for l in labels}
    res = scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO)
    calls = res["calls"]

    out = pd.DataFrame({"x": cx, "y": cy})
    for l in labels:
        out["score_" + l] = S[l].to_numpy()
    out["top_score_label"] = res["top_label"]
    out["second_label"] = res["second_label"]
    out["top_scaled"] = res["top_scaled"]
    out["second_scaled"] = res["second_scaled"]
    out["margin_scaled"] = res["margin_scaled"]
    out["fdr_pass"] = res["fdr_pass"]
    out["margin_pass"] = res["margin_pass"]
    out["celltype"] = calls
    out.to_csv(f"{OUT_DIR}/cell_scores.csv", index=False)

    print(f"\nannotated: {int((calls!='unknown').sum()):,}  "
          f"unknown: {int((calls=='unknown').sum()):,}")
    print(pd.Series(calls).value_counts().to_string())
    print(f"\nsaved -> {OUT_DIR}/cell_scores.csv")


if __name__ == "__main__":
    main()
