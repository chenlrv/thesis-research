"""check_homogeneity.py - how homogeneous are the slice-1 score_genes groups?

Two consistency checks (no marker-coverage assumptions):

  (1) sub-clustering test - Leiden each group on its OWN within-group PCA. A
      homogeneous group stays one cluster; a heterogeneous one fragments into
      separated subclusters (high silhouette) with distinct DE genes. We report
      n_subclusters, the fraction in the largest, the subcluster silhouette
      (how separated the split is - low => the split is noise, not structure),
      and the top DE genes distinguishing the subclusters.

  (3) intra- vs inter-group similarity - in scaled-PCA space, compare how close a
      group's cells sit to their OWN centroid (intra, cosine) vs to the nearest
      OTHER group's centroid. A homogeneous, well-separated group has high intra
      and a large gap to the nearest other.

Re-loads the same h5ad and recipe (normalize_total -> log1p) used to make the
calls, re-attaches `celltype` from cell_scores.csv (row order is preserved by the
original script), and excludes tumor. Tumor is already absent from cell_scores.
"""
import os

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
SCORES_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
OUT_DIR = "D:/thesis-research/score_genes_slice1"
TUMOR_COL = "pred_tumor_XGBoost"

N_PCS = 50
LEIDEN_RES = 1.0          # resolution for the per-group sub-clustering
MIN_CELLS = 100           # groups smaller than this are not split-tested
MIN_SUBCLUSTER_FRAC = 0.02  # ignore Leiden specks below this fraction of a group
SEED = 0

sc.settings.verbosity = 0


# ---- h5ad readers (match run_score_genes_slice1.py exactly for row alignment) ----
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
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def scaled_pca(a, n_comps=N_PCS):
    """z-score genes (clipped) then PCA; returns the embedding only."""
    a = a.copy()
    sc.pp.scale(a, max_value=10)
    n_comps = int(min(n_comps, a.n_vars - 1, a.n_obs - 1))
    sc.pp.pca(a, n_comps=n_comps, random_state=SEED)
    return a.obsm["X_pca"]


def load_adata_with_calls():
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

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    df = pd.read_csv(SCORES_CSV)
    if len(df) != adata.n_obs:
        raise SystemExit(
            f"row mismatch: cell_scores.csv has {len(df)} rows but non-tumor "
            f"adata has {adata.n_obs}. Re-run run_score_genes_slice1.py first."
        )
    # sanity-check that rows really align (same coordinates, same order)
    if not (np.allclose(df["x"].to_numpy(), cx, atol=1e-3)
            and np.allclose(df["y"].to_numpy(), cy, atol=1e-3)):
        raise SystemExit("coordinate mismatch: cell_scores.csv rows are not aligned "
                         "to the h5ad non-tumor order.")
    adata.obs["celltype"] = pd.Categorical(df["celltype"].to_numpy())
    adata.obs["x"], adata.obs["y"] = cx, cy
    return adata


# ---- check (1): sub-clustering test ------------------------------------------
def subcluster_test(adata, groups):
    rows = []
    print("\n" + "=" * 70)
    print("CHECK 1 - sub-clustering test (does the group fragment?)")
    print("=" * 70)
    for g in groups:
        sub = adata[adata.obs["celltype"] == g].copy()
        n = sub.n_obs
        if n < MIN_CELLS:
            print(f"\n{g}: n={n} < {MIN_CELLS}, skipped")
            rows.append({"group": g, "n_cells": n, "n_subclusters": np.nan,
                         "frac_largest": np.nan, "silhouette": np.nan,
                         "top_split_degs": ""})
            continue

        sub.obsm["X_pca"] = scaled_pca(sub)
        sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca", random_state=SEED)
        sc.tl.leiden(sub, resolution=LEIDEN_RES, flavor="igraph",
                     n_iterations=2, directed=False, random_state=SEED)

        sizes = sub.obs["leiden"].value_counts()
        keep = sizes[sizes >= MIN_SUBCLUSTER_FRAC * n]   # drop specks
        n_sub = int(len(keep))
        frac_largest = float(sizes.iloc[0] / n)

        sil, degs = np.nan, ""
        if n_sub > 1:
            mask_keep = sub.obs["leiden"].isin(keep.index)
            emb = sub.obsm["X_pca"][mask_keep.to_numpy()]
            lab = sub.obs["leiden"][mask_keep].to_numpy()
            sil = float(silhouette_score(emb, lab))
            # what separates the subclusters (top DE gene per kept subcluster)
            sc.tl.rank_genes_groups(sub, "leiden", method="wilcoxon",
                                    groups=list(keep.index))
            names = sub.uns["rank_genes_groups"]["names"]
            degs = "; ".join(f"{c}:{names[c][0]}/{names[c][1]}"
                             for c in keep.index)

        verdict = ("HOMOGENEOUS" if n_sub == 1 or frac_largest >= 0.9
                   or (np.isfinite(sil) and sil < 0.1)
                   else "heterogeneous?")
        sil_str = f"{sil:.2f}" if np.isfinite(sil) else "n/a"
        print(f"\n{g}: n={n}  subclusters(>={MIN_SUBCLUSTER_FRAC:.0%}): {n_sub}  "
              f"largest={frac_largest:.0%}  silhouette={sil_str}  -> {verdict}")
        if degs:
            print(f"   split markers: {degs}")
        rows.append({"group": g, "n_cells": n, "n_subclusters": n_sub,
                     "frac_largest": round(frac_largest, 3),
                     "silhouette": round(sil, 3) if np.isfinite(sil) else np.nan,
                     "top_split_degs": degs})
    return pd.DataFrame(rows)


# ---- check (3): intra- vs inter-group similarity -----------------------------
def similarity_test(adata, groups):
    print("\n" + "=" * 70)
    print("CHECK 3 - intra- vs inter-group similarity (cosine, scaled-PCA)")
    print("=" * 70)
    emb = scaled_pca(adata)
    centroids = np.vstack([emb[(adata.obs["celltype"] == g).to_numpy()].mean(0)
                           for g in groups])
    cc = cosine_similarity(centroids)          # group x group centroid cosine
    cc_df = pd.DataFrame(cc, index=groups, columns=groups)

    rows = []
    for i, g in enumerate(groups):
        m = (adata.obs["celltype"] == g).to_numpy()
        sims = cosine_similarity(emb[m], centroids)   # cells x groups
        intra = float(sims[:, i].mean())
        others = np.delete(sims, i, axis=1)
        other_groups = [x for j, x in enumerate(groups) if j != i]
        nearest_other = float(others.max(1).mean())
        nearest_group = other_groups[int(np.argmax(others.mean(0)))]
        rows.append({"group": g, "n": int(m.sum()),
                     "intra_cos": round(intra, 3),
                     "nearest_other_cos": round(nearest_other, 3),
                     "gap": round(intra - nearest_other, 3),
                     "nearest_other_group": nearest_group})
    sim_df = pd.DataFrame(rows).sort_values("gap")
    print("\n(small gap or high nearest_other_cos => group overlaps its neighbor)")
    print(sim_df.to_string(index=False))

    # centroid-correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
    for a in range(len(groups)):
        for b in range(len(groups)):
            ax.text(b, a, f"{cc[a, b]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="centroid cosine")
    ax.set_title("group centroid similarity (scaled-PCA)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/homogeneity_centroid_similarity.png", bbox_inches="tight")
    plt.close()
    return sim_df, cc_df


def main():
    import argparse
    global SCORES_CSV, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=SCORES_CSV, help="cell_scores.csv to read")
    ap.add_argument("--out", default=OUT_DIR, help="dir for outputs")
    args = ap.parse_args()
    SCORES_CSV, OUT_DIR = args.scores, args.out

    os.makedirs(OUT_DIR, exist_ok=True)
    adata = load_adata_with_calls()
    counts = adata.obs["celltype"].value_counts()
    groups = [g for g in counts.index if g != "unknown"]
    print(f"non-tumor cells: {adata.n_obs}")
    print("group sizes:\n" + counts.to_string())

    sub_df = subcluster_test(adata, groups)
    sim_df, cc_df = similarity_test(adata, groups)

    sub_df.to_csv(f"{OUT_DIR}/homogeneity_subclusters.csv", index=False)
    sim_df.to_csv(f"{OUT_DIR}/homogeneity_similarity.csv", index=False)
    print(f"\nsaved homogeneity_subclusters.csv, homogeneity_similarity.csv, "
          f"homogeneity_centroid_similarity.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
