"""Re-cluster the non-tumor cells of a slice and plot the result.

Steps (per the request):
  1. Load slice (default slice_3, the tumor-free control).
  2. Remove tumor cells by the XGBoost prediction column (pred_tumor_XGBoost).
  3. Cluster the remaining cells with a standard normalized pipeline
     (normalize_total -> log1p -> scale -> PCA -> neighbors -> Leiden).
  4. Plot the clustering: a UMAP and a spatial map (global coords), both
     colored by Leiden cluster.

Nothing is written back to the cache. Outputs -> D:/thesis-research/recluster_slice{ID}/.

Usage: python run_recluster_slice.py [slice_id] [leiden_resolution]
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import csr_matrix, csc_matrix

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "3"
LEIDEN_RES = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
OUT_DIR = f"D:/thesis-research/recluster_slice{SLICE_ID}"
TUMOR_COL = "pred_tumor_XGBoost"
N_PCS = 50


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
    return node[...]


def _read_var_names(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_obs_num(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_obs_bool(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):  # categorical-encoded
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {SLICE} ...")
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        if TUMOR_COL not in h5["obs"]:
            raise KeyError(f"{TUMOR_COL} not in obs; available: {list(h5['obs'].keys())}")
        tumor = _read_obs_bool(h5, TUMOR_COL)

    adata = ad.AnnData(X=X, obs=pd.DataFrame({"x": cx, "y": cy, "tumor": tumor}))
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    print(f"  total cells = {adata.n_obs}")
    print(f"  {TUMOR_COL} tumor calls = {int(tumor.sum())} ({100*tumor.mean():.2f}%)")

    # ---- 1) remove tumor cells by XGBoost ----
    adt = adata[~adata.obs["tumor"].to_numpy()].copy()
    print(f"  non-tumor cells carried forward = {adt.n_obs}")

    # drop control/negative probes
    keep = ~adt.var_names.str.lower().str.startswith(
        ("neg", "negprb", "blank", "falsecode", "systemcontrol"))
    adt = adt[:, keep].copy()
    print(f"  genes after dropping control probes = {adt.n_vars}")

    # ---- 2) standard normalized clustering pipeline ----
    adt.layers["counts"] = adt.X.copy()
    sc.pp.normalize_total(adt, target_sum=1e4)
    sc.pp.log1p(adt)
    adt.raw = adt
    sc.pp.scale(adt, max_value=10)
    sc.tl.pca(adt, n_comps=min(N_PCS, adt.n_vars - 1, adt.n_obs - 1))
    sc.pp.neighbors(adt, n_neighbors=15, n_pcs=min(N_PCS, adt.obsm["X_pca"].shape[1]))
    print(f"  running Leiden (resolution={LEIDEN_RES}) ...")
    try:
        sc.tl.leiden(adt, resolution=LEIDEN_RES, flavor="igraph",
                     n_iterations=2, directed=False)
    except Exception as e:
        print(f"  leiden(igraph) failed ({e}); default flavor")
        sc.tl.leiden(adt, resolution=LEIDEN_RES)
    sc.tl.umap(adt)

    clusters = adt.obs["leiden"].astype(str)
    cats = sorted(clusters.unique(), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    n_clusters = len(cats)
    print(f"\n  {n_clusters} Leiden clusters")
    vc = clusters.value_counts()
    for c in cats:
        print(f"    cluster {c:>2}: {int(vc[c]):6d}  ({100*vc[c]/len(clusters):4.1f}%)")

    # consistent colors across both plots
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))
    color = {c: cmap(i) for i, c in enumerate(cats)}
    cvec = np.array([color[c] for c in clusters])

    # ---- 3a) UMAP ----
    um = adt.obsm["X_umap"]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=160)
    ax.scatter(um[:, 0], um[:, 1], s=2, c=cvec, linewidths=0, rasterized=True)
    for c in cats:
        m = clusters.to_numpy() == c
        ax.text(um[m, 0].mean(), um[m, 1].mean(), c, fontsize=11, fontweight="bold",
                ha="center", va="center")
    ax.set_title(f"slice_{SLICE_ID} non-tumor — UMAP, Leiden res={LEIDEN_RES} "
                 f"({n_clusters} clusters)")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/clusters_umap.png", dpi=160,
                                    bbox_inches="tight"); plt.close()

    # ---- 3b) spatial map (global coords; flip Y to match slide) ----
    x = adt.obs["x"].to_numpy(); y = -adt.obs["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 10), dpi=170)
    ax.scatter(x, y, s=2, c=cvec, linewidths=0, rasterized=True)
    for c in cats:
        m = clusters.to_numpy() == c
        ax.text(x[m].mean(), y[m].mean(), c, fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_{SLICE_ID} non-tumor — spatial, Leiden res={LEIDEN_RES} "
                 f"({n_clusters} clusters)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/clusters_spatial.png", dpi=170,
                                    bbox_inches="tight"); plt.close()

    # legend reference
    fig, ax = plt.subplots(figsize=(3, 0.35 * n_clusters + 1), dpi=120)
    for i, c in enumerate(cats):
        ax.scatter([], [], c=[color[c]], label=f"cluster {c} (n={int(vc[c])})", s=40)
    ax.legend(loc="center", frameon=False); ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/clusters_legend.png", dpi=120,
                                    bbox_inches="tight"); plt.close()

    adt.obs[["x", "y", "leiden"]].to_csv(f"{OUT_DIR}/cluster_assignments.csv")

    # ---- 3c) diagnostic: is the UMAP axis library size or cell identity? ----
    # color the SAME umap by total counts and by restricted markers.
    total_counts = np.asarray(adt.layers["counts"].sum(1)).ravel()
    lut = {v.lower(): v for v in adt.raw.var_names}
    def rawval(g):
        gg = lut.get(g.lower())
        if gg is None:
            return None
        j = list(adt.raw.var_names).index(gg)
        return np.asarray(adt.raw.X[:, j].todense()).ravel()

    panels = [("__total__", np.log1p(total_counts), "log total counts")]
    for g in ["GFAP", "TMEM119", "Cx3cr1", "P2rx5", "C1qa", "Pecam1", "Sparcl1", "Col1a1"]:
        v = rawval(g)
        if v is not None:
            panels.append((g, v, f"{g} (log-norm)"))

    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.6 * nrow), dpi=150)
    for ax, (name, val, title) in zip(axes.ravel(), panels):
        order = np.argsort(val)
        vmax = np.quantile(val, 0.99) or 1.0
        sctr = ax.scatter(um[order, 0], um[order, 1], s=2, c=val[order], cmap="viridis",
                          vmin=float(np.quantile(val, 0.02)), vmax=float(vmax),
                          linewidths=0, rasterized=True)
        fig.colorbar(sctr, ax=ax, shrink=0.7)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"slice_{SLICE_ID} — UMAP colored by total counts vs markers "
                 f"(is the axis content or identity?)", fontsize=14)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/umap_diagnostic.png", dpi=150,
                                    bbox_inches="tight"); plt.close()

    # numeric: mean total counts per cluster (library-size confound check)
    print("\n=== mean total counts per cluster (library-size check) ===")
    lib = pd.Series({c: total_counts[clusters.to_numpy() == c].mean() for c in cats})
    for c in cats:
        print(f"    cluster {c:>2}: mean total counts = {lib[c]:8.1f}  (n={int(vc[c])})")
    print(f"    ratio max/min = {lib.max()/lib.min():.1f}x")

    print(f"\nSaved: clusters_umap.png, clusters_spatial.png, umap_diagnostic.png, "
          f"clusters_legend.png, cluster_assignments.csv  ->  {OUT_DIR}")


if __name__ == "__main__":
    main()
