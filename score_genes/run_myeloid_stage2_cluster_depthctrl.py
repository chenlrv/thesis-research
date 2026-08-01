"""Stage 2 (DEPTH-CONTROLLED clustering + tentative annotation).

Sibling of run_myeloid_stage2_cluster.py. The plain clustering split the myeloid
cells by LIBRARY SIZE, not subtype: in cluster_family_z the families moved in
lockstep (one cluster high on microglia AND BAM AND MDM, another low on all),
and slice_1 cluster-0 DE was cell-cycle/housekeeping/ambient (Stmn1, Tyms, Ppia,
Meg3). So those clusters were not labelable.

This version removes the depth axis before clustering, then labels clusters only
when one subtype family clearly dominates:

  consume Stage-1 Myeloid cells -> (decontX counts by default)
  -> QC-filter low-count/low-complexity cells (they form the 'everything-low'
     cluster) -> normalize/log1p -> HVG -> REGRESS OUT total_counts + n_genes
     -> scale -> PCA -> neighbors -> Leiden -> UMAP
  -> characterize (unchanged) -> tentative per-cluster label by family-z margin.

Annotation rule (margin gate): a cluster is called microglia / BAM / MDM only if
that family's mean-z is positive AND beats the next subtype family by
ANNOT_MARGIN_MIN. A cluster high on everything (residual depth) has a small
margin -> 'myeloid_unresolved'. MDM is kept as a candidate family but, per the
per-cell gate (Ccr2 ambient-level), is not expected to win anywhere.

Reporter genes (tdTomato/GFP) are never used for clustering -- summarized per
cluster afterward as orthogonal validation only.
"""
import os

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

USE_DECONTX = True   # depth-controlled run defaults to decontX (run_decontx_correct.py first)
SLICES_RAW = {
    "slice_1": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad",
}
SLICES_DECONTX = {
    "slice_1": "D:/thesis-research/resources/cache/decontx/slice_1_decontx.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/decontx/slice_3_decontx.h5ad",
}
SLICES = SLICES_DECONTX if USE_DECONTX else SLICES_RAW
OUT_ROOT = "D:/thesis-research/score_genes_myeloid_stage2_depthctrl" + \
           ("_decontx" if USE_DECONTX else "")
STAGE1_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
STAGE1_MYELOID_LABEL = "Myeloid"
TUMOR_COL = "pred_tumor_XGBoost"

# ---- QC filter (drops the low-quality cells that formed the 'everything-low'
#      cluster). Count floor is RELATIVE (percentile) so it adapts to the raw vs
#      decontX count scale -- an absolute floor over-dropped decontX cells (it
#      removed ~75% of healthy myeloid). Tune QC_COUNT_PCTL from the printed
#      percentiles. ----
QC_COUNT_PCTL = 10   # drop cells below this percentile of total_counts
MIN_GENES = 5        # hard floor: a cell with fewer detected genes can't cluster

# ---- depth control: regress these obs columns out before scale/PCA ----
REGRESS_OUT = ["total_counts", "n_genes"]

# clustering params
N_HVG = 500
N_PCS = 30
N_NEIGHBORS = 15
LEIDEN_RES = 1.0   # higher than the plain run: depth-controlled data needs more
                   # resolution to split BAM out (0.5 gave only 2 clusters)
MIN_CELLS = 100

# annotation: assign a subtype to a cluster only if its family-z is meaningfully
# positive (>= ANNOT_MIN_Z, not merely > 0 -- a flat ~0 family must not win as
# 'least negative') AND beats the next subtype family by ANNOT_MARGIN_MIN.
ANNOT_MIN_Z = 0.3
ANNOT_MARGIN_MIN = 0.3
UNRESOLVED_LABEL = "myeloid_unresolved"

MARKER_PANEL = {
    "pan-myeloid": ["Csf1r", "Aif1", "Tyrobp", "Fcer1g", "C1qa", "Ptprc", "Cd68",
                    "Cx3cr1", "Selplg"],
    "microglia":   ["TMEM119", "P2rx5"],
    "BAM":         ["Mrc1", "Cd163", "Lyve1", "Pf4"],
    "MDM":         ["Ccr2", "Plac8"],
    "activation":  ["Cd14", "Apoe", "Trem2", "Cd74", "Spp1", "Gpnmb"],
}
SUBTYPE_FAMILIES = ["microglia", "BAM", "MDM"]
LABEL_COLORS = {
    "microglia": "#1f77b4",
    "BAM": "#ff7f0e",
    "MDM": "#9467bd",
    UNRESOLVED_LABEL: "#bdbdbd",
}
REPORTER_GENES = ["tdTomato", "GFP"]
CONTROL_PREFIXES = ("neg", "negprb", "blank", "falsecode", "systemcontrol")
TUMOR_COLOR = "#f1c9c9"


# --------------------------------------------------------------------------- #
# h5py loaders (shared with the Stage-1 / plain Stage-2 scripts)
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
    csv = os.path.join(STAGE1_ROOT, name, "cell_scores.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"Stage-1 output missing: {csv}\nRun run_score_genes_myeloid_stage1.py first."
        )
    df = pd.read_csv(csv)
    if len(df) != len(cx):
        raise ValueError(
            f"{name}: Stage-1 rows ({len(df)}) != non-tumor cells ({len(cx)}); "
            "regenerate Stage 1 before clustering."
        )
    if not (np.allclose(df["x"].to_numpy(), cx) and np.allclose(df["y"].to_numpy(), cy)):
        raise ValueError(
            f"{name}: Stage-1 x/y do not match current non-tumor coords; ordering mismatch."
        )
    return df["celltype"].to_numpy().astype(str) == STAGE1_MYELOID_LABEL


def gene_counts(adata, gene):
    present = resolve_genes([gene], adata.var_names)
    if not present:
        return None, None
    source = adata.layers["counts"] if "counts" in adata.layers else adata.X
    idx = int(np.where(adata.var_names == present[0])[0][0])
    return _dense(source[:, idx]).ravel(), present[0]


def cluster_myeloid(mye):
    """Re-process and Leiden-cluster the myeloid subset, regressing out depth.

    Reporter / control probes are excluded from features. total_counts and
    n_genes must already be in mye.obs (used by sc.pp.regress_out).
    """
    var_lower = mye.var_names.str.lower()
    reporter_lower = {g.lower() for g in REPORTER_GENES}
    keep = (~var_lower.str.startswith(CONTROL_PREFIXES)) & (~var_lower.isin(reporter_lower))
    mye = mye[:, keep].copy()
    sc.pp.normalize_total(mye, target_sum=1e4)
    sc.pp.log1p(mye)
    mye.layers["lognorm"] = mye.X.copy()
    sc.pp.highly_variable_genes(mye, n_top_genes=min(N_HVG, mye.n_vars - 1))
    if REGRESS_OUT:
        sc.pp.regress_out(mye, REGRESS_OUT)   # <-- removes the library-size axis
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


def annotate_clusters(fam_z, uniq):
    """Tentative per-cluster label by subtype family-z margin.

    Returns {cluster: (label, dom_family, margin)}.
    """
    fams = [f for f in SUBTYPE_FAMILIES if f in fam_z.columns]
    out = {}
    for c in uniq:
        sub = fam_z.loc[c, fams]
        dom = sub.idxmax()
        nextmax = sub.drop(dom).max() if len(fams) > 1 else -np.inf
        margin = float(sub[dom] - nextmax)
        label = dom if (sub[dom] >= ANNOT_MIN_Z and margin >= ANNOT_MARGIN_MIN) \
            else UNRESOLVED_LABEL
        out[c] = (label, dom, margin)
    return out


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
        print(f"  only {mye.n_obs} Stage-1 Myeloid cells (< {MIN_CELLS}); skipping")
        return

    # ---- reporter counts (validation only; attached before any gene filtering) ----
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

    # ---- depth metrics (over all genes) + QC filter ----
    total_counts = _dense(mye.layers["counts"].sum(1)).ravel()
    n_genes = _dense((mye.layers["counts"] > 0).sum(1)).ravel()
    print(f"  counts/cell  p10={np.percentile(total_counts,10):.0f} "
          f"median={np.median(total_counts):.0f} p90={np.percentile(total_counts,90):.0f}; "
          f"genes/cell  p10={np.percentile(n_genes,10):.0f} median={np.median(n_genes):.0f}")
    count_thr = np.percentile(total_counts, QC_COUNT_PCTL)
    qc = (total_counts >= count_thr) & (n_genes >= MIN_GENES)
    print(f"  QC filter (counts>=p{QC_COUNT_PCTL}={count_thr:.0f} & genes>={MIN_GENES}): "
          f"keep {int(qc.sum()):,} / {mye.n_obs:,}  (drop {int((~qc).sum()):,})")
    mye = mye[qc].copy()
    cxm, cym = cxm[qc], cym[qc]
    mye.obs["total_counts"] = total_counts[qc]
    mye.obs["n_genes"] = n_genes[qc]
    if mye.n_obs < MIN_CELLS:
        print(f"  only {mye.n_obs} cells after QC (< {MIN_CELLS}); skipping")
        return

    mye.X = mye.layers["counts"].copy()
    mye = cluster_myeloid(mye)

    clusters = mye.obs["leiden"].astype(str).to_numpy()
    uniq = sorted(set(clusters), key=int)
    print(f"  {mye.n_obs} cells -> {len(uniq)} Leiden clusters (res={LEIDEN_RES}, "
          f"regress_out={REGRESS_OUT})")
    print(mye.obs["leiden"].value_counts().sort_index().to_string())

    # ---- reporter support per cluster ----
    reporter_rows = []
    for c in uniq:
        m = clusters == c
        row = {"leiden": c, "n_cells": int(m.sum())}
        for gene in REPORTER_GENES:
            cc = f"{gene}_counts"
            if cc not in mye.obs:
                continue
            vals = mye.obs[cc].to_numpy(dtype=float)[m]
            row[f"{gene}_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
            row[f"{gene}_pct_gt0"] = float(100 * np.nanmean(vals > 0)) if len(vals) else np.nan
            row[f"{gene}_max"] = float(np.nanmax(vals)) if len(vals) else np.nan
        reporter_rows.append(row)
    reporter_df = pd.DataFrame(reporter_rows)
    reporter_df.to_csv(f"{out_dir}/cluster_reporter_stats.csv", index=False)

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
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title(f"{name} myeloid (depth-controlled): {len(uniq)} clusters")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3,
                  fontsize=7, title="cluster")
        plt.tight_layout(); plt.savefig(f"{out_dir}/umap_clusters.png",
                                        bbox_inches="tight"); plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        for ax, key in zip(axes, ["total_counts", "n_genes"]):
            v = mye.obs[key].to_numpy()
            sctr = ax.scatter(U[:, 0], U[:, 1], s=4, c=v, cmap="viridis", linewidths=0,
                              rasterized=True, vmax=np.quantile(v, 0.99))
            fig.colorbar(sctr, ax=ax, shrink=0.7)
            ax.set_title(key); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name} myeloid UMAP -- QC (depth should NO LONGER define clusters)")
        plt.tight_layout(); plt.savefig(f"{out_dir}/umap_qc.png",
                                        bbox_inches="tight"); plt.close()

    # ---- spatial: Leiden clusters in tissue ----
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.0, c=TUMOR_COLOR,
               linewidths=0, rasterized=True, label="tumor")
    ax.scatter(cx, cy, s=0.5, c="#ededed", linewidths=0, rasterized=True)
    for c in uniq:
        m = clusters == c
        ax.scatter(cxm[m], cym[m], s=5, color=color_of[c], linewidths=0,
                   rasterized=True, label=f"c{c} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name} myeloid Leiden clusters in space (depth-controlled)")
    ax.legend(loc="lower right", markerscale=3, fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(f"{out_dir}/spatial_clusters.png",
                                    bbox_inches="tight"); plt.close()

    # ---- candidate-marker characterization ----
    groups = {grp: resolve_genes(gs, mye.var_names) for grp, gs in MARKER_PANEL.items()}
    groups = {g: v for g, v in groups.items() if v}
    sc.settings.figdir = out_dir
    try:
        sc.pl.dotplot(mye, groups, groupby="leiden", layer="lognorm",
                      standard_scale="var", show=False, save="_markers.png")
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

    fig, ax = plt.subplots(figsize=(0.42 * len(allmarkers) + 3, 0.45 * len(uniq) + 2), dpi=150)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(allmarkers))); ax.set_xticklabels(allmarkers, rotation=90, fontsize=7)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels([f"c{c} (n={int((clusters == c).sum())})" for c in uniq])
    x0 = 0
    for v in groups.values():
        x0 += len(v)
        if x0 < len(allmarkers):
            ax.axvline(x0 - 0.5, color="k", lw=0.8)
    fig.colorbar(im, ax=ax, shrink=0.6, label="z of mean log-norm (across clusters)")
    ax.set_title(f"{name} cluster x marker (depth-controlled)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/cluster_marker_heatmap.png",
                                    bbox_inches="tight"); plt.close()

    # ---- unbiased DE ----
    mye.X = mye.layers["lognorm"].copy()
    sc.tl.rank_genes_groups(mye, "leiden", method="wilcoxon", use_raw=False)
    rgg = mye.uns["rank_genes_groups"]
    de_rows = []
    print("\n=== unbiased top DE genes per cluster (Wilcoxon, log-norm) ===")
    for c in uniq:
        nm, lfc, padj = rgg["names"][c], rgg["logfoldchanges"][c], rgg["pvals_adj"][c]
        n_sig = int(np.sum((padj < 0.05) & (lfc > 1)))
        print(f"  c{c} (n={int((clusters == c).sum())}, {n_sig} sig markers): "
              + ", ".join(nm[:8]))
        for g, l, p in zip(nm, lfc, padj):
            de_rows.append({"cluster": c, "gene": g, "log2fc": float(l), "pval_adj": float(p)})
    pd.DataFrame(de_rows).to_csv(f"{out_dir}/cluster_de_genes.csv", index=False)

    # ---- family-z + tentative annotation by margin ----
    fam_z = pd.DataFrame(index=uniq)
    for grp, gs in groups.items():
        fam_z[grp] = z[gs].mean(axis=1)
    fam_z.to_csv(f"{out_dir}/cluster_family_z.csv")
    annot = annotate_clusters(fam_z, uniq)

    print("\n=== marker-family mean-z per cluster ===")
    print(fam_z.round(2).to_string())
    print(f"\n=== tentative cluster labels (margin gate >= {ANNOT_MARGIN_MIN}) ===")
    label_of = {}
    annot_rows = []
    for c in uniq:
        label, dom, margin = annot[c]
        label_of[c] = label
        n = int((clusters == c).sum())
        print(f"  c{c}: {label:18s} (dom={dom}, margin={margin:.2f}, n={n})")
        annot_rows.append({"leiden": c, "n_cells": n, "label": label,
                           "dominant_family": dom, "margin": margin})
    pd.DataFrame(annot_rows).to_csv(f"{out_dir}/cluster_subtype_annotation.csv", index=False)

    cell_labels = np.array([label_of[c] for c in clusters], dtype=object)
    print("\n=== cell composition by tentative label ===")
    vc = pd.Series(cell_labels).value_counts()
    for k, v in vc.items():
        print(f"  {k:18s} {v:6,} ({100 * v / len(cell_labels):5.1f}%)")

    # ---- labeled spatial + umap ----
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.0, c=TUMOR_COLOR,
               linewidths=0, rasterized=True, label="tumor")
    ax.scatter(cx, cy, s=0.5, c="#ededed", linewidths=0, rasterized=True)
    for lbl in list(LABEL_COLORS):
        m = cell_labels == lbl
        if not m.any():
            continue
        ax.scatter(cxm[m], cym[m], s=5, color=LABEL_COLORS[lbl], linewidths=0,
                   rasterized=True, label=f"{lbl} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name} myeloid tentative subtypes (cluster-level, margin-gated)")
    ax.legend(loc="lower right", markerscale=3, fontsize=7)
    plt.tight_layout(); plt.savefig(f"{out_dir}/spatial_subtypes.png",
                                    bbox_inches="tight"); plt.close()

    if have_umap:
        U = mye.obsm["X_umap"]
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        for lbl in list(LABEL_COLORS):
            m = cell_labels == lbl
            if not m.any():
                continue
            ax.scatter(U[m, 0], U[m, 1], s=5, color=LABEL_COLORS[lbl], linewidths=0,
                       rasterized=True, label=f"{lbl} ({int(m.sum())})")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        ax.set_title(f"{name} tentative subtypes on UMAP")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3, fontsize=7)
        plt.tight_layout(); plt.savefig(f"{out_dir}/umap_subtypes.png",
                                        bbox_inches="tight"); plt.close()

    # ---- per-cell output ----
    assignments = pd.DataFrame({
        "x": cxm, "y": cym, "leiden": clusters, "label": cell_labels,
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
