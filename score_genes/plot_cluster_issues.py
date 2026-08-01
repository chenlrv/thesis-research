"""Supervisor-facing figures: WHY the plain myeloid clustering does not yield
microglia/BAM/MDM subtypes.

Consumes the outputs of run_myeloid_stage2_cluster.py (raw and decontX) -- run
that first. Produces captioned figures that make three issues visible:

  1. depth_lockstep_<variant>_<slice>.png
     Marker families move in LOCKSTEP across clusters, and the clusters are
     ordered by library size -> clusters capture RNA content, not subtype.
  2. depth_vs_marker_<variant>_<slice>.png
     Per-cluster subtype-marker level scales with library size (Pearson r near
     1) -> no depth-independent subtype signal.
  3. ccr2_ambient_tumor_vs_healthy.png
     The MDM marker Ccr2 is ~1 transcript and no higher in tumor than healthy
     -> a recruited-MDM population cannot be defined.

Nothing here re-clusters; it only reads the CSVs (+ the raw h5ads for Ccr2).
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

VARIANTS = {
    "raw": "D:/thesis-research/score_genes_myeloid_stage2",
    "decontX": "D:/thesis-research/score_genes_myeloid_stage2_decontx",
}
SLICES = ["slice_1", "slice_3"]
SLICE_LABEL = {"slice_1": "slice_1 (tumor)", "slice_3": "slice_3 (healthy)"}
RAW_H5AD = {
    "slice_1": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad",
}
STAGE1_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
TUMOR_COL = "pred_tumor_XGBoost"
OUT = "D:/thesis-research/cluster_issue_plots"

FAMILY_MARKERS = {
    "microglia": ["TMEM119", "P2rx5"],
    "BAM": ["Mrc1", "Cd163", "Lyve1", "Pf4"],
    "MDM": ["Ccr2", "Plac8"],
}
FAMILY_COLORS = {"pan-myeloid": "#555555", "microglia": "#1f77b4", "BAM": "#ff7f0e",
                 "MDM": "#9467bd", "activation": "#2ca02c"}

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "figure.dpi": 150})


# --------------------------------------------------------------------------- #
def load_outputs(root, slc):
    d = os.path.join(root, slc)
    fz = pd.read_csv(os.path.join(d, "cluster_family_z.csv"), index_col=0)
    fz.index = fz.index.astype(int)
    mm = pd.read_csv(os.path.join(d, "cluster_marker_means.csv"), index_col=0)
    mm.index = mm.index.astype(int)
    asg = pd.read_csv(os.path.join(d, "cluster_assignments.csv"))
    depth = asg.groupby("leiden")["total_counts"].mean()
    depth.index = depth.index.astype(int)
    size = asg["leiden"].value_counts()
    size.index = size.index.astype(int)
    return fz, mm, depth, size


def resolve_cols(cols, wanted):
    lut = {c.lower(): c for c in cols}
    return [lut[w.lower()] for w in wanted if w.lower() in lut]


# --------------------------------------------------------------------------- #
def fig_lockstep(variant, slc, fz, depth, size):
    order = depth.sort_values().index.tolist()
    x = np.arange(len(order))
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(max(7, 0.6 * len(order) + 3), 8),
                                   sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for fam in fz.columns:
        axA.plot(x, fz.reindex(order)[fam], marker="o", lw=2,
                 color=FAMILY_COLORS.get(fam, None), label=fam)
    axA.axhline(0, color="k", lw=0.8, ls="--")
    axA.set_ylabel("marker-family mean-z\n(across clusters)")
    axA.legend(ncol=5, fontsize=10, loc="upper left")
    axA.set_title(f"{SLICE_LABEL[slc]} — {variant}: marker families move together "
                  f"(lockstep)\n→ clusters separate by RNA content, not subtype")

    axB.bar(x, depth.reindex(order).values, color="#888888")
    axB.set_ylabel("mean counts / cell")
    axB.set_xlabel("Leiden cluster (ordered by library size →)")
    axB.set_xticks(x)
    axB.set_xticklabels([f"c{c}\n(n={int(size.get(c, 0))})" for c in order], fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, f"depth_lockstep_{variant}_{slc}.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def fig_depth_vs_marker(variant, slc, mm, depth):
    fig, ax = plt.subplots(figsize=(7, 6))
    txt = []
    for fam, genes in FAMILY_MARKERS.items():
        cols = resolve_cols(mm.columns, genes)
        if not cols:
            continue
        y = mm[cols].mean(axis=1)
        common = y.index.intersection(depth.index)
        xv, yv = depth.loc[common].values, y.loc[common].values
        r = np.corrcoef(xv, yv)[0, 1] if len(common) > 2 else np.nan
        ax.scatter(xv, yv, s=60, color=FAMILY_COLORS.get(fam), label=f"{fam} (r={r:.2f})")
        if len(common) > 2:
            b, a = np.polyfit(xv, yv, 1)
            xs = np.array([xv.min(), xv.max()])
            ax.plot(xs, a + b * xs, color=FAMILY_COLORS.get(fam), lw=1, ls="--")
        txt.append(f"{fam}: r={r:.2f}")
    ax.set_xlabel("cluster mean counts / cell (library size)")
    ax.set_ylabel("cluster mean subtype-marker expression (log-norm)")
    ax.set_title(f"{SLICE_LABEL[slc]} — {variant}: every subtype's marker level\n"
                 f"scales with library size → no depth-independent signal")
    ax.legend(title="each point = a cluster")
    fig.tight_layout()
    f = os.path.join(OUT, f"depth_vs_marker_{variant}_{slc}.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    print(f"    depth-vs-marker r: {', '.join(txt)}")
    return f


# ---- Ccr2 ambient figure (reads raw h5ads) -------------------------------- #
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


def ccr2_among_myeloid(slc):
    with h5py.File(RAW_H5AD[slc], "r") as h5:
        X = _read_X(h5); vn = list(_read_var(h5)); tumor = _read_bool(h5, TUMOR_COL)
    Xn = X[~tumor]
    lut = {str(v).lower(): i for i, v in enumerate(vn)}
    if "ccr2" not in lut:
        return None
    ccr2 = np.asarray(Xn[:, lut["ccr2"]].todense()).ravel()
    s1 = pd.read_csv(os.path.join(STAGE1_ROOT, slc, "cell_scores.csv"))
    mye = s1["celltype"].to_numpy().astype(str) == "Myeloid"
    return ccr2[mye]


def fig_ccr2_ambient():
    data = {slc: ccr2_among_myeloid(slc) for slc in SLICES}
    data = {k: v for k, v in data.items() if v is not None}
    if not data:
        print("  Ccr2 absent from panel; skipping ambient figure")
        return None
    maxc = int(max(v.max() for v in data.values()))
    bins = np.arange(0, maxc + 2) - 0.5
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"slice_1": "#d62728", "slice_3": "#1f77b4"}
    for slc, v in data.items():
        ax.hist(v, bins=bins, density=True, alpha=0.55, color=colors.get(slc),
                label=f"{SLICE_LABEL[slc]}: {100*(v>0).mean():.0f}% >0, "
                      f"mean(>0)={v[v>0].mean():.2f}, max={int(v.max())}")
    ax.set_xlabel("Ccr2 raw counts (per myeloid cell)")
    ax.set_ylabel("fraction of myeloid cells")
    ax.set_xticks(range(0, maxc + 1))
    ax.set_title("MDM marker Ccr2 is ambient: ~1 transcript, and NOT higher in\n"
                 "tumor than healthy → recruited-MDM population not definable")
    ax.legend()
    fig.tight_layout()
    f = os.path.join(OUT, "ccr2_ambient_tumor_vs_healthy.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    for variant, root in VARIANTS.items():
        for slc in SLICES:
            d = os.path.join(root, slc)
            if not os.path.exists(os.path.join(d, "cluster_family_z.csv")):
                print(f"  {variant}/{slc}: outputs missing -- run clustering first; skipping")
                continue
            fz, mm, depth, size = load_outputs(root, slc)
            print(f"\n{variant}/{slc}: {len(fz)} clusters")
            print("   ", fig_lockstep(variant, slc, fz, depth, size))
            print("   ", fig_depth_vs_marker(variant, slc, mm, depth))
    f = fig_ccr2_ambient()
    if f:
        print("\n", f)
    print(f"\nsaved supervisor figures -> {OUT}")


if __name__ == "__main__":
    main()
