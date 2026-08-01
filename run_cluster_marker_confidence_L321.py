"""Positive-selection labeling of the 13 L321 Leiden clusters, anchored on
TYPE-RESTRICTED markers (not composite signature scores).

Motivation: thresholding the composite astrocyte signature would falsely tag
stromal/meningeal clusters (which share S100b/Glul/Apoe) as astrocyte without any
GFAP. So we gate each cluster on the marker that only the target type expresses:
    astrocyte  <- GFAP
    microglia  <- TMEM119 / Cx3cr1 / P2rx5  (microglia-restricted)
A cluster is ASSIGNED to a type only if its restricted-marker expression is both
(a) a clear positive outlier across clusters (z >= ASSIGN_Z) and (b) above an
absolute detection floor (mean raw count >= FLOOR_MEAN and %positive >= FLOOR_POS).
Otherwise the cluster is 'unknown'. Nothing is written back to the cache.
"""
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix, csc_matrix

CACHE = "D:/thesis-research/resources/cache/sample_L321_adata.h5ad"
CLUSTER_COL = ("Basic.run_Neighbor.network.expression.space.1_1"
               "_cluster_Basic.run_Leiden.Clustering.1_1")
OUT_CSV = "D:/thesis-research/L321_cluster_marker_confidence.csv"

# restricted markers that anchor each call
RESTRICTED = {
    "astrocyte": ["GFAP"],
    "microglia": ["TMEM119", "Cx3cr1", "P2rx5"],
}
# context markers (printed, not used for the gate)
CONTEXT = ["Sparcl1", "Glul", "S100b", "Sox9",          # astro context
           "C1qa", "Csf1r", "Aif1",                      # pan-myeloid context
           "Mrc1", "Cd163", "Plac8", "Pecam1", "Col1a1"] # other lineages

# confidence thresholds (absolute floor + relative outlier)
ASSIGN_Z = 1.5      # restricted marker must be this many SD above the cross-cluster mean
FLOOR_MEAN = 0.10   # ... and at least this mean raw count in the cluster
FLOOR_POS = 0.05    # ... and detected in at least this fraction of the cluster's cells


def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_obs_column(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
    return _decode(node[...])


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


def main():
    with h5py.File(CACHE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        clusters = _read_obs_column(h5, CLUSTER_COL)
    valid = pd.notna(clusters)
    X, clusters = X[valid], clusters[valid].astype(str)
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    adata.obs["cluster"] = clusters

    # keep raw counts; build a normalized layer to remove the library-size axis
    adata.layers["counts"] = adata.X.copy()
    total_counts = np.asarray(adata.layers["counts"].sum(1)).ravel()
    keep = ~adata.var_names.str.lower().str.startswith(
        ("neg", "negprb", "blank", "falsecode", "systemcontrol"))
    adata = adata[:, keep].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    lut = {v.lower(): v for v in adata.var_names}
    def col(gene, layer=None):
        g = lut.get(gene.lower())
        if g is None:
            return None
        M = adata[:, g].layers["counts"] if layer == "counts" else adata[:, g].X
        return np.asarray(M.todense()).ravel()

    cl = adata.obs["cluster"].to_numpy()
    cats = sorted(set(cl), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    n_per = {c: int((cl == c).sum()) for c in cats}

    # ---- expose the library-size confound: mean total counts per cluster ----
    print("\n=== per-cluster mean TOTAL counts (library size) ===")
    lib = pd.Series({c: total_counts[cl == c].mean() for c in cats})
    for c in cats:
        print(f"  cluster {c:>2}: n={n_per[c]:6d}  mean total counts = {lib[c]:8.1f}")
    print(f"  (ratio max/min cluster = {lib.max()/lib.min():.1f}x)")

    all_markers = sum(RESTRICTED.values(), []) + CONTEXT
    present = [g for g in all_markers if g.lower() in lut]
    absent = [g for g in all_markers if g.lower() not in lut]
    if absent:
        print("absent from panel:", absent)

    # per-cluster mean raw count + % positive for every marker
    rows = []
    for g in present:
        v = col(g)
        for c in cats:
            m = cl == c
            rows.append({"gene": g, "cluster": c,
                         "mean": v[m].mean(), "pos": (v[m] > 0).mean()})
    mt = pd.DataFrame(rows)
    mean_tab = mt.pivot(index="cluster", columns="gene", values="mean").loc[cats]
    pos_tab = mt.pivot(index="cluster", columns="gene", values="pos").loc[cats]

    print("\n=== per-cluster MEAN normalized expr (log1p per-10k; library-size removed) ===")
    with pd.option_context("display.width", 240, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(mean_tab.round(3))
    print("\n=== per-cluster % positive (fraction of cells with count>0) ===")
    with pd.option_context("display.width", 240, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.2f}"):
        print(pos_tab.round(2))

    # ---- confidence gate on restricted markers ----
    # for each type: take its best restricted marker per cluster, z across clusters
    print(f"\n=== restricted-marker confidence gate "
          f"(z>={ASSIGN_Z}, mean>={FLOOR_MEAN}, pos>={FLOOR_POS}) ===")
    assign = {}
    detail = []
    for typ, genes in RESTRICTED.items():
        present_g = [g for g in genes if g in mean_tab.columns]
        if not present_g:
            continue
        sub = mean_tab[present_g]
        z = (sub - sub.mean(0)) / (sub.std(0) + 1e-9)
        # cluster's strongest restricted marker for this type
        best_gene = z.idxmax(1)
        best_z = z.max(1)
        for c in cats:
            g = best_gene[c]
            zc, mc, pc = best_z[c], mean_tab.loc[c, g], pos_tab.loc[c, g]
            ok = (zc >= ASSIGN_Z) and (mc >= FLOOR_MEAN) and (pc >= FLOOR_POS)
            detail.append({"cluster": c, "type": typ, "marker": g,
                           "z": round(zc, 2), "mean": round(mc, 3),
                           "pos": round(pc, 3), "pass": ok})
            if ok:
                assign.setdefault(c, []).append((typ, zc))

    final = {}
    for c in cats:
        if c in assign:
            # if both types somehow pass, take the higher-z one
            final[c] = sorted(assign[c], key=lambda t: -t[1])[0][0]
        else:
            final[c] = "unknown"

    dd = pd.DataFrame(detail)
    print(dd.to_string(index=False))

    res = pd.DataFrame({"cluster": cats, "n": [n_per[c] for c in cats],
                        "label": [final[c] for c in cats]})
    res.to_csv(OUT_CSV, index=False)
    print("\n=== FINAL positive-selection labels (restricted-marker gated) ===")
    print(res.to_string(index=False))
    n_assigned = (res["label"] != "unknown").sum()
    print(f"\nassigned {n_assigned}/{len(cats)} clusters; "
          f"{len(cats)-n_assigned} -> unknown")
    print("saved", OUT_CSV)


if __name__ == "__main__":
    main()
