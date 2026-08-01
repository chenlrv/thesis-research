"""Supervisor-facing figures: WHY the per-cell subtyping gate
(run_myeloid_stage2_gate.py) does not give reliable microglia/BAM/MDM calls.

Consumes that script's outputs (subtype_calls.csv) + the raw h5ads. Three issues:

  1. gate_marker_sparsity.png
     The subtype markers are detected in only a small fraction of myeloid cells
     -> there is usually no signal to score per cell.
  2. gate_composition.png
     Consequence: ~70% of cells stay 'unresolved', and microglia (which should
     dominate brain myeloid) come out a minority -> biologically implausible.
  3. (ccr2_not_enriched.png, from plot_cluster_issues_simple.py)
     The one MDM marker, Ccr2, is rare and not tumor-enriched.

Reads score_genes_myeloid_stage2_gate/<slice>/subtype_calls.csv (run that first).
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

GATE_ROOT = "D:/thesis-research/score_genes_myeloid_stage2_gate"
STAGE1_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
SLICES = ["slice_1", "slice_3"]
SLICE_LABEL = {"slice_1": "slice_1 (tumor)", "slice_3": "slice_3 (healthy)"}
SLICE_COLOR = {"slice_1": "#d62728", "slice_3": "#1f77b4"}
RAW_H5AD = {
    "slice_1": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad",
}
TUMOR_COL = "pred_tumor_XGBoost"
OUT = "D:/thesis-research/cluster_issue_plots"

SUBTYPE_MARKERS = {
    "microglia": ["TMEM119", "P2rx5"],
    "BAM": ["Mrc1", "Cd163", "Lyve1", "Pf4"],
    "MDM": ["Ccr2", "Plac8"],
}
COMP_ORDER = ["microglia", "BAM", "MDM", "myeloid_unresolved"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 150})


# --------------------------------------------------------------------------- #
def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_X(h5):
    node = h5["X"]
    if isinstance(node, h5py.Group):
        enc = str(node.attrs.get("encoding-type", ""))
        shape = tuple(node.attrs["shape"])
        data, idx, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
        M = csc_matrix((data, idx, indptr), shape=shape) if "csc" in enc \
            else csr_matrix((data, idx, indptr), shape=shape)
        return M.tocsr()
    return csr_matrix(node[...])


def _read_var(h5):
    var = h5["var"]; key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_bool(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]; cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def composition(slc):
    df = pd.read_csv(os.path.join(GATE_ROOT, slc, "subtype_calls.csv"))
    vc = df["subtype"].value_counts()
    n = len(df)
    return {k: 100 * vc.get(k, 0) / n for k in COMP_ORDER}, n


def marker_pct_detected(slc):
    """% of Stage-1 myeloid cells with >=1 raw count, per subtype marker."""
    with h5py.File(RAW_H5AD[slc], "r") as h5:
        X = _read_X(h5); vn = list(_read_var(h5)); tumor = _read_bool(h5, TUMOR_COL)
    Xn = X[~tumor]
    lut = {str(v).lower(): i for i, v in enumerate(vn)}
    s1 = pd.read_csv(os.path.join(STAGE1_ROOT, slc, "cell_scores.csv"))
    mye = s1["celltype"].to_numpy().astype(str) == "Myeloid"
    Xm = Xn[mye]
    out = {}
    for st, genes in SUBTYPE_MARKERS.items():
        for g in genes:
            if g.lower() in lut:
                col = np.asarray(Xm[:, lut[g.lower()]].todense()).ravel()
                out[(st, g)] = 100 * (col > 0).mean()
    return out


# --------------------------------------------------------------------------- #
def fig_composition():
    comps = {slc: composition(slc) for slc in SLICES}
    x = np.arange(len(COMP_ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, slc in enumerate(SLICES):
        comp, n = comps[slc]
        vals = [comp[k] for k in COMP_ORDER]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=SLICE_COLOR[slc],
                      edgecolor="black", label=f"{SLICE_LABEL[slc]} (n={n:,})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}%", ha="center",
                    va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["microglia", "BAM", "MDM", "unresolved\n(no call)"])
    ax.set_ylabel("% of myeloid cells")
    ax.set_title("Per-cell subtyping: ~70% of cells get no call, and microglia\n"
                 "(should dominate brain myeloid) come out a minority")
    ax.legend()
    fig.tight_layout()
    f = os.path.join(OUT, "gate_composition.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def fig_marker_sparsity():
    dets = {slc: marker_pct_detected(slc) for slc in SLICES}
    keys = [(st, g) for st, genes in SUBTYPE_MARKERS.items() for g in genes
            if (st, g) in dets[SLICES[0]]]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, slc in enumerate(SLICES):
        vals = [dets[slc][k] for k in keys]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=SLICE_COLOR[slc],
                      edgecolor="black", label=SLICE_LABEL[slc])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({st})" for st, g in keys])
    ax.set_ylabel("% of myeloid cells with the marker detected (≥1 count)")
    ax.set_title("Subtype markers are detected in only a small fraction of cells\n"
                 "→ usually no signal to assign a subtype per cell")
    ax.legend()
    fig.tight_layout()
    f = os.path.join(OUT, "gate_marker_sparsity.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    for slc in SLICES:
        if not os.path.exists(os.path.join(GATE_ROOT, slc, "subtype_calls.csv")):
            print(f"  {slc}: subtype_calls.csv missing -- run run_myeloid_stage2_gate.py")
            return
    print("  ", fig_marker_sparsity())
    print("  ", fig_composition())
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
