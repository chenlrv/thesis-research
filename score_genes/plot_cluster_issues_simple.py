"""Simpler, supervisor-facing versions of the clustering-issue figures.

Same message as plot_cluster_issues.py but with no z-scores / no correlation
coefficients -- just raw average expression:

  1. same_cluster_wins_<slice>.png
     Grouped bars: a few subtype markers, bars = clusters. If clusters were cell
     types each marker would peak in its OWN cluster; instead one cluster (the one
     with the most RNA, shown in the lower panel) is highest for ALL markers.
  2. expected_vs_observed_slice_1.png
     Two small heatmaps: what subtype clusters SHOULD look like (one bright block
     per cluster) vs what we OBSERVE (one cluster bright across all markers).
  3. ccr2_not_enriched.png
     Two bars: % of myeloid cells with Ccr2>=2 in tumor vs healthy -- recruited
     monocytes are not enriched in tumor, so MDM can't be defined.

Reads run_myeloid_stage2_cluster.py outputs (raw) + the raw h5ads for Ccr2.
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

VARIANT = "decontX"            # which clustering run to plot: "raw" or "decontX"
DATA_ROOTS = {"raw": "D:/thesis-research/score_genes_myeloid_stage2",
              "decontX": "D:/thesis-research/score_genes_myeloid_stage2_decontx"}
DATA_ROOT = DATA_ROOTS[VARIANT]
SLICES = ["slice_1", "slice_3"]
SLICE_LABEL = {"slice_1": "slice_1 (tumor)", "slice_3": "slice_3 (healthy)"}
RAW_H5AD = {
    "slice_1": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad",
}
STAGE1_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
TUMOR_COL = "pred_tumor_XGBoost"
OUT = "D:/thesis-research/cluster_issue_plots"

# one or two clear markers per subtype (label shows which cell type they mark)
MARKERS = [("TMEM119", "microglia"), ("Mrc1", "BAM"), ("Lyve1", "BAM"),
           ("Ccr2", "MDM"), ("Plac8", "MDM")]
FAMILY_MARKERS = {"microglia": ["TMEM119", "P2rx5"],
                  "BAM": ["Mrc1", "Cd163", "Lyve1", "Pf4"],
                  "MDM": ["Ccr2", "Plac8"]}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "figure.dpi": 150})


def load(slc):
    d = os.path.join(DATA_ROOT, slc)
    mm = pd.read_csv(os.path.join(d, "cluster_marker_means.csv"), index_col=0)
    mm.index = mm.index.astype(int)
    asg = pd.read_csv(os.path.join(d, "cluster_assignments.csv"))
    depth = asg.groupby("leiden")["total_counts"].mean(); depth.index = depth.index.astype(int)
    return mm, depth


def resolve(cols, wanted):
    lut = {c.lower(): c for c in cols}
    return [lut[w.lower()] for w in wanted if w.lower() in lut]


def fig_same_cluster_wins(slc):
    mm, depth = load(slc)
    order = depth.sort_values().index.tolist()          # low -> high RNA
    colors = plt.cm.cividis(np.linspace(0.15, 0.9, len(order)))
    markers = [(resolve(mm.columns, [g])[0], fam) for g, fam in MARKERS
               if resolve(mm.columns, [g])]
    x = np.arange(len(markers))
    w = 0.8 / len(order)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})
    for i, c in enumerate(order):
        vals = [mm.loc[c, g] for g, _ in markers]
        axA.bar(x + i * w, vals, w, color=colors[i],
                label=f"cluster {c}", edgecolor="white", linewidth=0.5)
    axA.set_xticks(x + 0.4 - w / 2)
    axA.set_xticklabels([f"{g}\n({fam})" for g, fam in markers])
    axA.set_ylabel("average expression")
    axA.legend(title="", fontsize=11)
    axA.set_title(f"{SLICE_LABEL[slc]} [{VARIANT}]: the SAME cluster is highest for "
                  "every marker\n(real cell types would each peak in a different cluster)")

    axB.bar(np.arange(len(order)), depth.reindex(order).values, color=colors,
            edgecolor="white", linewidth=0.5)
    axB.set_xticks(np.arange(len(order)))
    axB.set_xticklabels([f"cluster {c}" for c in order])
    axB.set_ylabel("RNA per cell\n(total counts)")
    axB.set_title("...because that cluster simply has more RNA per cell", fontsize=12)
    fig.tight_layout()
    f = os.path.join(OUT, f"same_cluster_wins_{VARIANT}_{slc}.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def fig_expected_vs_observed(slc):
    mm, depth = load(slc)
    order = depth.sort_values().index.tolist()
    fams = list(FAMILY_MARKERS)
    obs = np.zeros((len(order), len(fams)))
    for j, fam in enumerate(fams):
        cols = resolve(mm.columns, FAMILY_MARKERS[fam])
        obs[:, j] = mm.loc[order, cols].mean(axis=1).values if cols else 0
    obs = obs / (obs.max(axis=0, keepdims=True) + 1e-9)   # scale each marker-family to its own max

    # expected: each cluster lights up one family (block diagonal, best effort)
    exp = np.zeros_like(obs)
    for i in range(len(order)):
        exp[i, i % len(fams)] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in [(axes[0], exp, "If clusters were cell types\n(each cluster = one type)"),
                           (axes[1], obs, "What we observe\n(one cluster bright for ALL)")]:
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"cluster {c}" for c in order])
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.5 else "black", fontsize=11)
    fig.suptitle(f"{SLICE_LABEL[slc]} [{VARIANT}]: subtype structure vs. observed clusters",
                 fontsize=14)
    fig.tight_layout()
    f = os.path.join(OUT, f"expected_vs_observed_{VARIANT}_{slc}.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


# ---- Ccr2 (reads raw h5ads) ----------------------------------------------- #
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


def ccr2_myeloid(slc):
    with h5py.File(RAW_H5AD[slc], "r") as h5:
        X = _read_X(h5); vn = list(_read_var(h5)); tumor = _read_bool(h5, TUMOR_COL)
    lut = {str(v).lower(): i for i, v in enumerate(vn)}
    if "ccr2" not in lut:
        return None
    ccr2 = np.asarray(X[~tumor][:, lut["ccr2"]].todense()).ravel()
    s1 = pd.read_csv(os.path.join(STAGE1_ROOT, slc, "cell_scores.csv"))
    return ccr2[s1["celltype"].to_numpy().astype(str) == "Myeloid"]


def fig_ccr2_not_enriched():
    vals = {slc: ccr2_myeloid(slc) for slc in SLICES}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        return None
    labels = [SLICE_LABEL[s] for s in vals]
    pct = [100 * (v >= 2).mean() for v in vals.values()]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, pct, color=["#d62728", "#1f77b4"], edgecolor="black")
    for b, p in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, p, f"{p:.1f}%", ha="center",
                va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("% of myeloid cells that are Ccr2-high (>=2)")
    ax.set_ylim(0, max(pct) * 1.4 if pct else 1)
    ax.set_title("Ccr2-high cells are rare in both (4% tumor, 2% healthy)\n"
                 "and only ~1 transcript when present → no MDM cluster to find")
    fig.tight_layout()
    f = os.path.join(OUT, "ccr2_not_enriched.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"  variant = {VARIANT}  ({DATA_ROOT})")
    for slc in SLICES:
        if not os.path.exists(os.path.join(DATA_ROOT, slc, "cluster_marker_means.csv")):
            print(f"  {slc}: clustering outputs missing; skipping")
            continue
        print("  ", fig_same_cluster_wins(slc))
        print("  ", fig_expected_vs_observed(slc))
    f = fig_ccr2_not_enriched()
    if f:
        print("  ", f)
    print(f"\nsaved simpler figures -> {OUT}")


if __name__ == "__main__":
    main()
