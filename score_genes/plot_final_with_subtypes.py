"""Combined slice-1 map (OvR-abstain style) with the Myeloid group resolved into
subtypes: BAM (Lyve1 >= 3 only), MDM, Microglia; plus Vascular, Astrocytes,
Ependymal, Neurons; tumor in black; everything else grey.

Output -> score_genes_slice1_merged/classify/final_with_subtypes.png
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
SUB = f"{OUT}/ovr_myeloid_subtypes.csv"
LYVE_THRESH = 3.0
COL = {"unknown": "#dddddd", "MDM": "#00a087", "Microglia": "#17becf",
       "Vascular": "#2ca02c", "Astrocytes": "#1f77b4", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "BAM": "#d62728"}
BACKBONE = ["Vascular", "Astrocytes", "Ependymal", "Neurons"]
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    tx, ty = cx[tumor], cy[tumor]
    adata = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]
    lyve = adata[:, "Lyve1"].X
    lyve = (lyve.toarray() if sp.issparse(lyve) else np.asarray(lyve)).ravel()
    coord = {(round(float(a), 2), round(float(b), 2)): i
             for i, (a, b) in enumerate(zip(cxnt, cynt))}

    ovr = pd.read_csv(OVR)
    assert np.allclose(ovr["x"].to_numpy(), cxnt, atol=1e-3)
    final = ovr["final_label"].to_numpy()

    combined = np.full(len(cxnt), "unknown", dtype=object)
    for g in BACKBONE:
        combined[final == g] = g

    sub = pd.read_csv(SUB)
    for _, r in sub.iterrows():
        i = coord[(round(float(r["x"]), 2), round(float(r["y"]), 2))]
        s = r["subtype"]
        if s in ("MDM", "Microglia", "BAM"):
            combined[i] = s
        # myeloid-unknown stays 'unknown'

    counts = {k: int((combined == k).sum()) for k in COL}
    print("counts:", counts, "tumor:", len(tx))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(15, 10), dpi=180)
    m = combined == "unknown"
    ax.scatter(cxnt[m], cynt[m], s=1.2, c=COL["unknown"], linewidths=0,
               rasterized=True, label=f"unknown ({counts['unknown']:,})")
    # backbone + MDM/Microglia at original size
    for g in ["MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal", "Neurons"]:
        mm = combined == g
        ax.scatter(cxnt[mm], cynt[mm], s=1.6, c=COL[g], linewidths=0,
                   rasterized=True, label=f"{g} ({counts[g]:,})")
    # tumor black
    ax.scatter(tx, ty, s=1.6, c="black", linewidths=0, rasterized=True,
               label=f"tumor ({len(tx):,})")
    # BAM (Lyve1>=3) on top, enlarged so the rare subset is visible
    mb = combined == "BAM"
    ax.scatter(cxnt[mb], cynt[mb], s=1.6, c=COL["BAM"], linewidths=0,
               rasterized=True, label=f"BAM ({counts['BAM']:,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("slice1 annotation", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", markerscale=5, fontsize=9, frameon=True)
    fig.savefig(f"{OUT}/final_annotation_allbam.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved final_annotation_allbam.png -> {OUT}")


if __name__ == "__main__":
    main()
