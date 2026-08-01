"""Stage 2 (CLUSTERING ONLY -- no annotation): subcluster the current Stage-1
Myeloid calls and characterize the structure before any subtype labels are
committed.

Stage 1 writes score_genes_myeloid_stage1/<slice>/cell_scores.csv using the
current broad annotation method (score_genes + mirrored-FDR + MAD-scaled margin).
This script consumes exactly the cells labeled "Myeloid" there, verifies row
alignment by x/y coordinates, then on that myeloid subset only:
    drop control probes and reporter genes -> normalize/log1p -> HVG -> scale
    -> PCA -> neighbors -> Leiden -> UMAP.

Outputs per slice (NO subtype labels assigned):
    - console: n Stage-1 Myeloid cells, n clusters + sizes
    - umap_clusters.png, umap_qc.png (library size / complexity)
    - spatial_clusters.png (clusters in tissue, tumor in grey)
    - dotplot_markers.png (candidate markers x cluster: mean expr + % positive)
    - cluster_marker_heatmap.png (z of mean log-norm across clusters)
    - cluster_marker_means.csv, cluster_reporter_stats.csv,
      cluster_assignments.csv

Reporter genes (tdTomato/GFP) are never used for clustering. They are summarized
per cluster afterward as orthogonal validation, especially for MDM-like clusters.
"""
import os

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

USE_DECONTX = False  # True -> cluster on decontX-corrected counts (run_decontx_correct.py first)
SLICES_RAW = {
    "slice_1": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad",
}
SLICES_DECONTX = {
    "slice_1": "D:/thesis-research/resources/cache/decontx/slice_1_decontx.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/decontx/slice_3_decontx.h5ad",
}
SLICES = SLICES_DECONTX if USE_DECONTX else SLICES_RAW
OUT_ROOT = "D:/thesis-research/score_genes_myeloid_stage2" + ("_decontx" if USE_DECONTX else "")
STAGE1_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
STAGE1_MYELOID_LABEL = "Myeloid"
TUMOR_COL = "pred_tumor_XGBoost"

# clustering params (tune LEIDEN_RES to merge/split clusters)
N_HVG = 500
N_PCS = 30
N_NEIGHBORS = 15
LEIDEN_RES = 0.5   # lower -> fewer/broader clusters (1.0 over-split, 0.3 gave only 2)
MIN_CELLS = 100   # skip clustering if fewer Stage-1 Myeloid cells than this

# candidate markers -- for CHARACTERIZATION ONLY (never used to label clusters)
MARKER_PANEL = {
    "pan-myeloid": ["Csf1r", "Aif1", "Tyrobp", "Fcer1g", "C1qa", "Ptprc", "Cd68",
                    "Cx3cr1", "Selplg"],
    "microglia":   ["TMEM119", "P2rx5"],
    "BAM":         ["Mrc1", "Cd163", "Lyve1", "Pf4"],
    "MDM":         ["Ccr2", "Plac8"],
    "activation":  ["Cd14", "Apoe", "Trem2", "Cd74", "Spp1", "Gpnmb"],
}
REPORTER_GENES = ["tdTomato", "GFP"]
CONTROL_PREFIXES = ("neg", "negprb", "blank", "falsecode", "systemcontrol")


# --------------------------------------------------------------------------- #
# h5py loaders (shared with the Stage-1 script)
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


def resolve_genes(genes, var_names):
    lut = {str(v).lower(): str(v) for v in var_names}
    return [lut[g.lower()] for g in genes if g.lower() in lut]


def _dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def load_stage1_myeloid_mask(name, cx, cy):
    """Load current Stage-1 Myeloid calls and verify row alignment by coordinates."""
    csv = os.path.join(STAGE1_ROOT, name, "cell_scores.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"Stage-1 output missing: {csv}\n"
            "Run run_score_genes_myeloid_stage1.py first."
        )
    df = pd.read_csv(csv)
    if len(df) != len(cx):
        raise ValueError(
            f"{name}: Stage-1 rows ({len(df)}) != non-tumor cells ({len(cx)}); "
            "regenerate Stage 1 before clustering."
        )
    if not (np.allclose(df["x"].to_numpy(), cx) and np.allclose(df["y"].to_numpy(), cy)):
        raise ValueError(
            f"{name}: Stage-1 x/y do not match current non-tumor coords; "
            "cell ordering mismatch."
        )
    return df["celltype"].to_numpy().astype(str) == STAGE1_MYELOID_LABEL


def gene_counts(adata, gene):
    """Return one raw/corrected count vector from adata.layers['counts'] if present."""
    present = resolve_genes([gene], adata.var_names)
    if not present:
        return None, None
    source = adata.layers["counts"] if "counts" in adata.layers else adata.X
    idx = int(np.where(adata.var_names == present[0])[0][0])
    values = source[:, idx]
    return _dense(values).ravel(), present[0]


def cluster_myeloid(mye):
    """Re-process and Leiden-cluster the myeloid subset.

    Reporter genes are excluded from clustering features, but their counts stay
    in obs columns that were attached before this function.
    """
    var_lower = mye.var_names.str.lower()
    reporter_lower = {g.lower() for g in REPORTER_GENES}
    keep = (~var_lower.str.startswith(CONTROL_PREFIXES)) & (~var_lower.isin(reporter_lower))
    mye = mye[:, keep].copy()
    sc.pp.normalize_total(mye, target_sum=1e4)
    sc.pp.log1p(mye)
    mye.layers["lognorm"] = mye.X.copy()
    sc.pp.highly_variable_genes(mye, n_top_genes=min(N_HVG, mye.n_vars - 1))
    sc.pp.scale(mye, max_value=10)
    sc.tl.pca(mye, n_comps=min(N_PCS, mye.n_obs - 1, mye.n_vars - 1))
    sc.pp.neighbors(mye, n_neighbors=N_NEIGHBORS, use_rep="X_pca")
    try:
        sc.tl.leiden(mye, resolution=LEIDEN_RES, flavor="igraph",
                     n_iterations=2, directed=False)
    except Exception as e:
        print(f"  leiden(igraph) failed ({e}); using default flavor")
        sc.tl.leiden(mye, resolution=LEIDEN_RES)
    return mye


def run_slice(name, path):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {name} ===")

    with h5py.File(path, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)

    cx_all, cy_all, tumor_all = cx.copy(), cy.copy(), tumor.copy()
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata = adata[~tumor].copy()
    cx, cy = cx[~tumor], cy[~tumor]
    print(f"non-tumor cells = {adata.n_obs}")

    stage1_myeloid = load_stage1_myeloid_mask(name, cx, cy)
    print(f"Stage-1 Myeloid cells = {int(stage1_myeloid.sum()):,} "
          f"({100 * stage1_myeloid.mean():.1f}% of non-tumor)")

    mye = adata[stage1_myeloid].copy()
    cxm, cym = cx[stage1_myeloid], cy[stage1_myeloid]
    if mye.n_obs < MIN_CELLS:
        print(f"  only {mye.n_obs} Stage-1 Myeloid cells (< {MIN_CELLS}); skipping clustering")
        return

    # Reporter genes are validation metadata only; they are excluded from HVG/PCA.
    for gene in REPORTER_GENES:
        values, resolved = gene_counts(mye, gene)
        if values is None:
            mye.obs[f"{gene}_counts"] = np.nan
            mye.obs[f"{gene}_pos"] = False
            print(f"  reporter {gene}: absent from panel")
        else:
            mye.obs[f"{gene}_counts"] = values
            mye.obs[f"{gene}_pos"] = values > 0
            print(f"  reporter {resolved} > 0: {int((values > 0).sum()):,} "
                  f"({100 * (values > 0).mean():.1f}% of Stage-1 Myeloid)")

    mye.X = mye.layers["counts"].copy()
    mye = cluster_myeloid(mye)
    mye.obs["total_counts"] = _dense(mye.layers["counts"].sum(1)).ravel()
    mye.obs["n_genes"] = _dense((mye.layers["counts"] > 0).sum(1)).ravel()

    clusters = mye.obs["leiden"].astype(str).to_numpy()
    uniq = sorted(set(clusters), key=int)
    print(f"  {mye.n_obs} cells -> {len(uniq)} Leiden clusters (res={LEIDEN_RES})")
    print(mye.obs["leiden"].value_counts().sort_index().to_string())

    reporter_rows = []
    for c in uniq:
        m = clusters == c
        row = {"leiden": c, "n_cells": int(m.sum())}
        for gene in REPORTER_GENES:
            counts_col = f"{gene}_counts"
            if counts_col not in mye.obs:
                continue
            vals = mye.obs[counts_col].to_numpy(dtype=float)[m]
            row[f"{gene}_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
            row[f"{gene}_pct_gt0"] = float(100 * np.nanmean(vals > 0)) if len(vals) else np.nan
            row[f"{gene}_pct_ge2"] = float(100 * np.nanmean(vals >= 2)) if len(vals) else np.nan
            row[f"{gene}_max"] = float(np.nanmax(vals)) if len(vals) else np.nan
        reporter_rows.append(row)
    reporter_df = pd.DataFrame(reporter_rows)
    reporter_df.to_csv(f"{out_dir}/cluster_reporter_stats.csv", index=False)
    print("\n=== reporter support per cluster (validation only; not used for clustering) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(reporter_df.round(2).to_string(index=False))

    cmap = plt.get_cmap("tab20")
    color_of = {c: cmap(i % 20) for i, c in enumerate(uniq)}

    # ---- UMAP ----
    have_umap = True
    try:
        sc.tl.umap(mye)
    except Exception as e:
        print(f"  umap failed ({e}); skipping umap plots")
        have_umap = False

    if have_umap:
        U = mye.obsm["X_umap"]
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        for c in uniq:
            m = clusters == c
            ax.scatter(U[m, 0], U[m, 1], s=4, color=color_of[c], linewidths=0,
                       rasterized=True, label=f"{c} ({int(m.sum())})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title(f"{name} Stage-1 Myeloid: {len(uniq)} Leiden clusters")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3,
                  fontsize=7, title="cluster")
        plt.tight_layout(); plt.savefig(f"{out_dir}/umap_clusters.png",
                                        bbox_inches="tight"); plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        for ax, key in zip(axes, ["total_counts", "n_genes"]):
            v = mye.obs[key].to_numpy()
            sctr = ax.scatter(U[:, 0], U[:, 1], s=4, c=v, cmap="viridis",
                              linewidths=0, rasterized=True,
                              vmax=np.quantile(v, 0.99))
            fig.colorbar(sctr, ax=ax, shrink=0.7)
            ax.set_title(key); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name} myeloid UMAP -- QC (watch for library-size clusters)")
        plt.tight_layout(); plt.savefig(f"{out_dir}/umap_qc.png",
                                        bbox_inches="tight"); plt.close()

    # ---- spatial ----
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.0, c="#f1c9c9",
               linewidths=0, rasterized=True, label="tumor")
    ax.scatter(cx, cy, s=0.5, c="#ededed", linewidths=0, rasterized=True)
    for c in uniq:
        m = clusters == c
        ax.scatter(cxm[m], cym[m], s=5, color=color_of[c], linewidths=0,
                   rasterized=True, label=f"c{c} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name} Stage-1 Myeloid Leiden clusters in space")
    ax.legend(loc="lower right", markerscale=3, fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(f"{out_dir}/spatial_clusters.png",
                                    bbox_inches="tight"); plt.close()

    # ---- candidate-marker characterization (NO labels) ----
    groups = {grp: resolve_genes(gs, mye.var_names)
              for grp, gs in MARKER_PANEL.items()}
    groups = {g: v for g, v in groups.items() if v}
    sc.settings.figdir = out_dir
    try:
        sc.pl.dotplot(mye, groups, groupby="leiden", layer="lognorm",
                      standard_scale="var", show=False, save="_markers.png")
        print(f"  marker dotplot -> {out_dir}/dotplot_markers.png")
    except Exception as e:
        print(f"  dotplot failed ({e})")

    allmarkers = [g for v in groups.values() for g in v]
    var_idx = {g: i for i, g in enumerate(mye.var_names)}
    M = _dense(mye.layers["lognorm"][:, [var_idx[g] for g in allmarkers]])
    L = pd.DataFrame(M, columns=allmarkers)
    L["leiden"] = clusters
    cm = L.groupby("leiden").mean().loc[uniq]
    cm.to_csv(f"{out_dir}/cluster_marker_means.csv")

    z = (cm - cm.mean(0)) / (cm.std(0) + 1e-9)
    fig, ax = plt.subplots(figsize=(0.42 * len(allmarkers) + 3,
                                    0.45 * len(uniq) + 2), dpi=150)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(allmarkers)))
    ax.set_xticklabels(allmarkers, rotation=90, fontsize=7)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels([f"c{c} (n={int((clusters == c).sum())})" for c in uniq])
    # group separators
    x0 = 0
    for v in groups.values():
        x0 += len(v)
        if x0 < len(allmarkers):
            ax.axvline(x0 - 0.5, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="z of mean log-norm (across clusters)")
    ax.set_title(f"{name} cluster x candidate marker  (NO labels assigned)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/cluster_marker_heatmap.png",
                                    bbox_inches="tight"); plt.close()

    # ---- are the clusters actually differentiated? -------------------------- #
    # (a) UNBIASED differential expression: all genes, not just our candidates,
    #     so this can't be accused of circular "we only fed in markers" reasoning.
    mye.X = mye.layers["lognorm"].copy()          # DE on log-norm, not scaled X
    sc.tl.rank_genes_groups(mye, "leiden", method="wilcoxon", use_raw=False)
    rgg = mye.uns["rank_genes_groups"]
    de_rows = []
    print("\n=== unbiased top DE genes per cluster (Wilcoxon, log-norm) ===")
    for c in uniq:
        nm, lfc, padj = rgg["names"][c], rgg["logfoldchanges"][c], rgg["pvals_adj"][c]
        n_sig = int(np.sum((padj < 0.05) & (lfc > 1)))
        print(f"  c{c} (n={int((clusters == c).sum())}, {n_sig} sig markers "
              f"[adj-p<0.05 & log2fc>1]): " + ", ".join(nm[:8]))
        for g, l, p in zip(nm, lfc, padj):
            de_rows.append({"cluster": c, "gene": g, "log2fc": float(l),
                            "pval_adj": float(p)})
    pd.DataFrame(de_rows).to_csv(f"{out_dir}/cluster_de_genes.csv", index=False)
    try:
        sc.pl.rank_genes_groups_dotplot(mye, n_genes=5, standard_scale="var",
                                        show=False, save="_de.png")
        print(f"  DE dotplot -> {out_dir}/dotplot_de.png")
    except Exception as e:
        print(f"  DE dotplot failed ({e})")

    # (b) which annotation marker-family dominates each cluster (DESCRIPTIVE only,
    #     not a committed label) -- microglia/BAM/MDM mean-z + the separation margin.
    fam_z = pd.DataFrame(index=uniq)
    for grp, gs in groups.items():
        fam_z[grp] = z[gs].mean(axis=1)
    subtype_fams = [f for f in ("microglia", "BAM", "MDM") if f in fam_z.columns]
    print("\n=== marker-family mean-z per cluster (descriptive characterization) ===")
    print(fam_z.round(2).to_string())
    print("dominant SUBTYPE family per cluster (microglia/BAM/MDM):")
    for c in uniq:
        sub = fam_z.loc[c, subtype_fams]
        dom = sub.idxmax()
        margin = sub.max() - sub.drop(dom).max() if len(subtype_fams) > 1 else np.nan
        print(f"  c{c}: {dom} (z={sub[dom]:.2f}, margin over next={margin:.2f})")
    fam_z.to_csv(f"{out_dir}/cluster_family_z.csv")

    assignments = pd.DataFrame({
        "x": cxm, "y": cym,
        "leiden": clusters,
        "total_counts": mye.obs["total_counts"].to_numpy(),
        "n_genes": mye.obs["n_genes"].to_numpy(),
    })
    for gene in REPORTER_GENES:
        for suffix in ("counts", "pos"):
            col = f"{gene}_{suffix}"
            if col in mye.obs:
                assignments[col] = mye.obs[col].to_numpy()
    assignments.to_csv(f"{out_dir}/cluster_assignments.csv", index=False)

    print(f"  saved outputs -> {out_dir}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    sc.settings.verbosity = 1
    for name, path in SLICES.items():
        run_slice(name, path)


if __name__ == "__main__":
    main()
