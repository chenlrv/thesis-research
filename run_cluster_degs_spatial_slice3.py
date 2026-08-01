"""For the slice-3 re-clustering: per-cluster spatial maps (others greyed) + DEGs.

Reuses the EXACT cluster labels already computed (recluster_slice{ID}/
cluster_assignments.csv) so the clusters match the plots already on disk, then
loads expression to (a) draw one spatial plot per cluster with non-cluster cells
in grey, and (b) compute differentially expressed genes per cluster
(rank_genes_groups, Wilcoxon, one-vs-rest).

Usage: python run_cluster_degs_spatial_slice3.py [slice_id]
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "3"
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
OUT_DIR = f"D:/thesis-research/recluster_slice{SLICE_ID}"
ASSIGN_CSV = f"{OUT_DIR}/cluster_assignments.csv"
SPATIAL_DIR = f"{OUT_DIR}/per_cluster_spatial"
TUMOR_COL = "pred_tumor_XGBoost"
N_TOP = 15


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


def _read_obs_bool(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def main():
    os.makedirs(SPATIAL_DIR, exist_ok=True)
    assign = pd.read_csv(ASSIGN_CSV, index_col=0)
    print(f"loaded {len(assign)} cluster assignments")

    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        tumor = _read_obs_bool(h5, TUMOR_COL)

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    adata = adata[~tumor].copy()
    keep = ~adata.var_names.str.lower().str.startswith(
        ("neg", "negprb", "blank", "falsecode", "systemcontrol"))
    adata = adata[:, keep].copy()

    assert adata.n_obs == len(assign), \
        f"expression rows {adata.n_obs} != assignments {len(assign)}"
    adata.obs["x"] = assign["x"].to_numpy()
    adata.obs["y"] = assign["y"].to_numpy()
    adata.obs["leiden"] = pd.Categorical(assign["leiden"].astype(str).to_numpy())
    cats = sorted(adata.obs["leiden"].cat.categories, key=lambda s: int(s))

    # ---- normalize + log1p for DE ----
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---- (3) DEGs per cluster: rank_genes_groups (Wilcoxon, one-vs-rest) ----
    print("computing DEGs (Wilcoxon, one-vs-rest) ...")
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", pts=True)
    rgg = adata.uns["rank_genes_groups"]
    groups = rgg["names"].dtype.names
    rows = []
    for g in groups:
        for r in range(N_TOP):
            gene = rgg["names"][g][r]
            rows.append({
                "cluster": g, "rank": r + 1, "gene": gene,
                "log2FC": float(rgg["logfoldchanges"][g][r]),
                "pval_adj": float(rgg["pvals_adj"][g][r]),
                "score": float(rgg["scores"][g][r]),
                "pct_in": float(rgg["pts"].loc[gene, g]) if "pts" in rgg else np.nan,
            })
    degs = pd.DataFrame(rows)
    degs.to_csv(f"{OUT_DIR}/cluster_DEGs.csv", index=False)
    print(f"saved {OUT_DIR}/cluster_DEGs.csv")
    print("\n=== top DEGs per cluster ===")
    for g in cats:
        top = degs[degs["cluster"] == g].head(N_TOP)
        names = ", ".join(top["gene"].tolist())
        print(f"  cluster {g} (n={int((adata.obs['leiden']==g).sum())}): {names}")

    # dotplot of top markers
    top_genes = {g: list(rgg["names"][g][:8]) for g in cats}
    flat, seen = [], set()
    for g in cats:
        for gn in top_genes[g]:
            if gn not in seen:
                seen.add(gn); flat.append(gn)
    try:
        sc.pl.dotplot(adata, var_names=flat, groupby="leiden", show=False,
                      standard_scale="var")
        plt.savefig(f"{OUT_DIR}/cluster_DEGs_dotplot.png", dpi=160, bbox_inches="tight")
        plt.close()
        print(f"saved {OUT_DIR}/cluster_DEGs_dotplot.png")
    except Exception as e:
        print(f"dotplot skipped: {e}")

    # ---- (2) per-cluster spatial maps, non-cluster cells in grey ----
    x = adata.obs["x"].to_numpy()
    y = -adata.obs["y"].to_numpy()   # flip Y to match slide
    lab = adata.obs["leiden"].to_numpy()
    print(f"drawing {len(cats)} per-cluster spatial maps -> {SPATIAL_DIR}")
    for g in cats:
        m = lab == g
        fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
        ax.scatter(x[~m], y[~m], s=1.2, c="lightgrey", linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=2.5, c="crimson", linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice_{SLICE_ID} — cluster {g}  (n={int(m.sum())}, "
                     f"{100*m.mean():.1f}%)")
        plt.savefig(f"{SPATIAL_DIR}/cluster_{g}_spatial.png", dpi=180,
                    bbox_inches="tight")
        plt.close()
    print(f"saved {len(cats)} plots")


if __name__ == "__main__":
    main()
