"""Per-type spatial maps for the slice-1 annotation (positive-markers-only backbone
+ myeloid subtypes, OvR-abstain LogReg), reproducing slice1_annotation.png:
one panel per type over a grey tissue background, tumor in black.

Labels: backbone from ovr_nontumor_predictions.csv (final_label), myeloid subtypes
from ovr_myeloid_subtypes.csv (subtype); BAM = subtype BAM AND Lyve1(log-norm)>=3.

Output -> score_genes_slice1_merged/classify/slice1_per_type.png
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
BACKBONE = ["Vascular", "Astrocytes", "Ependymal", "Neurons"]
PANELS = ["BAM", "MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal",
          "Neurons", "unknown"]
COL = {"unknown": "#999999", "MDM": "#00a087", "Microglia": "#17becf",
       "Vascular": "#2ca02c", "Astrocytes": "#1f77b4", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "BAM": "#d62728"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5); var = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X); adata.var_names = pd.Index(var); adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    tx, ty = cx[tumor], cy[tumor]
    adata = adata[~tumor].copy(); cxnt, cynt = cx[~tumor], cy[~tumor]
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
        if s in ("MDM", "Microglia"):
            combined[i] = s
        elif s == "BAM" and lyve[i] >= LYVE_THRESH:
            combined[i] = "BAM"

    counts = {g: int((combined == g).sum()) for g in PANELS}
    print("counts:", counts, "tumor:", len(tx))

    fig, axes = plt.subplots(2, 4, figsize=(26, 11), dpi=150)
    for ax, g in zip(axes.ravel(), PANELS):
        m = combined == g
        ax.scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
        ax.scatter(tx, ty, s=1.0, c="black", linewidths=0, rasterized=True)
        ax.scatter(cxnt[m], cynt[m], s=3.0, c=COL[g], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (n={counts[g]:,})", fontweight="bold")
    fig.suptitle("slice1 annotation — per type (grey = non-tumor, black = tumor)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/slice1_per_type.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT}/slice1_per_type.png")


if __name__ == "__main__":
    main()
