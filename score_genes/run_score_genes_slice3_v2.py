"""Fresh-start slice-3 annotation: 5-type merged backbone (no GFAP).

Per-cell score_genes -> mirrored-FDR + MAD-scaled 1.5x margin decision -> spatial
plots per type. Also reports the per-type marker-coverage distribution among
confident (FDR-passing) cells, to calibrate the coverage gate `k_T` (open question).

Reuses the proven loader/decision logic from run_score_genes_slice3.py, but with the
finalized 5-type marker dict and a coverage diagnostic. Non-destructive (reads the
h5ad, writes only to OUT_DIR).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
import sys

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

SLICE_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TUMOR_SLICES = {1, 2, 5, 6}  # control slices are 3, 4
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
OUT_DIR = f"D:/thesis-research/score_genes_slice{SLICE_ID}_v2"
TUMOR_COL = "pred_tumor_XGBoost"
FDR_CUTOFF = 0.05
MARGIN_RATIO = 1.5
# coverage gate: min # markers (raw count >= 1) the winning type must show.
# Calibrated from p10 of confident cells (compact sets 2, vascular 4).
COVERAGE_K = {"Astrocytes": 2, "Neurons": 2, "Myeloid": 2, "Ependymal": 2, "Vascular": 4}

# Ependymal: the epithelial genes (Epcam/Krt8/18/19/Cldn4) are shared with
# lung-adeno TUMOR, so on tumor slices we keep only the lineage-safe core to
# avoid mis-tagging residual/mis-segmented tumor as ependymal.
_EPENDYMAL_FULL = ["Ttr", "Adgrv1", "Cd24a", "Krt8", "Krt18", "Krt19", "Cldn4", "Epcam"]
_EPENDYMAL_SAFE = ["Ttr", "Adgrv1", "Cd24a"]

MARKER_GENES = {
    "Astrocytes": ["Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9"],
    "Neurons": [
        "Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1",
        "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst",
    ],
    "Myeloid": ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"],
    "Ependymal": _EPENDYMAL_SAFE if SLICE_ID in TUMOR_SLICES else _EPENDYMAL_FULL,
    "Vascular": [
        "Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Tie1", "Esam", "Slc2a1",
        "Clec14a", "Adgrl4", "Eng", "Icam2", "Ramp2", "Vwf",
        "Rgs5", "Pdgfrb", "Notch3", "Vtn",
    ],
}
CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4",
    "Neurons": "#e377c2",
    "Myeloid": "#00a087",
    "Ependymal": "#984ea3",
    "Vascular": "#2ca02c",
}
TUMOR_COLOR = "#000000"


# ---- h5ad readers (copied from run_score_genes_slice3.py; proven on these files) ----
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


def mirrored_fdr_threshold(scores, fdr=0.05):
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.sort(scores[scores > 0]))
    for t in candidates:
        n_called = int(np.sum(scores >= t))
        n_mirror = int(np.sum(scores <= -t))
        if n_mirror / max(n_called, 1) <= fdr:
            return float(t), n_called, n_mirror, float(n_mirror / max(n_called, 1))
    return np.inf, 0, 0, np.nan


def scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO):
    labels = list(S.columns)
    raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, axis=0)), axis=0)
    mad = np.where(mad > 0, mad, raw.std(axis=0))
    mad = np.where(mad > 0, mad, 1.0)
    scaled = raw / mad
    order = np.argsort(-scaled, axis=1)
    lab = np.asarray(labels)
    top_label = lab[order[:, 0]]
    second_label = lab[order[:, 1]]
    top = np.take_along_axis(scaled, order[:, :1], 1)[:, 0]
    second = np.take_along_axis(scaled, order[:, 1:2], 1)[:, 0]
    margin_pass = (top > 0) & ((second <= 0) | (top >= ratio * second))
    top_raw = raw[np.arange(raw.shape[0]), order[:, 0]]
    fdr_pass = top_raw >= np.array([thresholds[l] for l in top_label])
    calls = np.where(margin_pass & fdr_pass, top_label, "unknown").astype(object)
    return {
        "calls": calls, "top_label": top_label, "second_label": second_label,
        "top_scaled": top, "second_scaled": second, "margin_scaled": top - second,
        "fdr_pass": fdr_pass, "margin_pass": margin_pass,
    }


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)

    cx_all, cy_all, tumor_all = cx.copy(), cy.copy(), tumor.copy()
    # subset rows on the sparse matrix BEFORE building AnnData (avoids a full-object
    # copy) and cast to float32 -> lower peak memory on the big tumor slices.
    keep = ~tumor
    X = X[keep]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    cx, cy = cx[keep], cy[keep]
    print(f"non-tumor cells = {adata.n_obs:,} (removed {int(tumor.sum()):,} tumor)")

    labels = list(MARKER_GENES.keys())

    # ---- marker coverage (# markers with raw count >= 1), on RAW counts ----
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    Xraw = adata.X.tocsr()
    coverage = {}
    for label in labels:
        idx = [var_idx[g] for g in MARKER_GENES[label] if g in var_idx]
        det = (Xraw[:, idx] > 0)
        coverage[label] = np.asarray(det.sum(axis=1)).ravel()

    # ---- recipe: normalize -> log1p, score on log-normalized data ----
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    score_cols = {}
    print("\nscoring (score_genes, ctrl_size=50, n_bins=25, random_state=0):")
    for label in labels:
        genes = [g for g in MARKER_GENES[label] if g in adata.var_names]
        missing = [g for g in MARKER_GENES[label] if g not in adata.var_names]
        col = "score_" + label
        sc.tl.score_genes(adata, gene_list=genes, score_name=col,
                          ctrl_size=50, n_bins=25, random_state=0)
        score_cols[label] = col
        print(f"  {label:12s}: {len(genes)}/{len(MARKER_GENES[label])} genes"
              + (f"  MISSING {missing}" if missing else ""))

    S = adata.obs[[score_cols[l] for l in labels]].copy()
    S.columns = labels

    # ---- mirrored-FDR thresholds ----
    thresholds, thr_rows = {}, []
    for label in labels:
        t, n_called, n_mirror, est = mirrored_fdr_threshold(S[label].to_numpy(), FDR_CUTOFF)
        thresholds[label] = t
        thr_rows.append({"label": label, "threshold": t, "n_ge": n_called,
                         "n_mirror": n_mirror, "est_fdr": est})
    thr_df = pd.DataFrame(thr_rows)
    print(f"\n=== mirrored-FDR thresholds (FDR <= {FDR_CUTOFF}) ===")
    print(thr_df.round(4).to_string(index=False))

    res = scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO)
    fdr, mar = res["fdr_pass"], res["margin_pass"]

    # ---- v1: FDR + margin only (the previous run) ----
    calls_v1 = np.where(mar & fdr, res["top_label"], "unknown").astype(object)

    # ---- v2: sequential rule adding the coverage gate ----
    #   not FDR -> unknown ; FDR & not margin -> ambiguous ;
    #   FDR & margin & cov<k -> low_markers_coverage ; else -> label
    cov_mat = np.column_stack([coverage[l] for l in labels])
    lab_to_col = {l: i for i, l in enumerate(labels)}
    top_col = np.array([lab_to_col[l] for l in res["top_label"]])
    cov_top = cov_mat[np.arange(cov_mat.shape[0]), top_col]
    k_top = np.array([COVERAGE_K[l] for l in res["top_label"]])
    cov_pass = cov_top >= k_top

    calls_v2 = np.full(len(fdr), "unknown", dtype=object)
    calls_v2[fdr & ~mar] = "ambiguous"
    calls_v2[fdr & mar & ~cov_pass] = "low_markers_coverage"
    keep = fdr & mar & cov_pass
    calls_v2[keep] = res["top_label"][keep]
    calls = calls_v2

    special = ["unknown", "ambiguous", "low_markers_coverage"]
    v1_ann = calls_v1 != "unknown"
    v2_ann = ~np.isin(calls_v2, special)
    print(f"\n=== gates: FDR<= {FDR_CUTOFF}, margin {MARGIN_RATIO}x, coverage k={COVERAGE_K} ===")
    print(f"  FDR pass:     {int(fdr.sum()):>7,} ({100*fdr.mean():5.1f}%)")
    print(f"  margin pass:  {int(mar.sum()):>7,} ({100*mar.mean():5.1f}%)")
    print(f"  cov pass(top):{int(cov_pass.sum()):>7,} ({100*cov_pass.mean():5.1f}%)")
    print(f"\n  annotated v1 (FDR+margin):         {int(v1_ann.sum()):>7,} ({100*v1_ann.mean():5.1f}%)")
    print(f"  annotated v2 (+coverage):          {int(v2_ann.sum()):>7,} ({100*v2_ann.mean():5.1f}%)")
    print(f"  label -> low_markers_coverage:     {int((v1_ann & (calls_v2=='low_markers_coverage')).sum()):>7,}")
    print(f"  unknown -> ambiguous:              {int((~v1_ann & (calls_v2=='ambiguous')).sum()):>7,}")

    cmp = pd.DataFrame({
        "v1_FDR+margin": pd.Series(calls_v1).value_counts(),
        "v2_+coverage": pd.Series(calls_v2).value_counts(),
    }).fillna(0).astype(int)
    cmp["delta"] = cmp["v2_+coverage"] - cmp["v1_FDR+margin"]
    print("\n=== per-category counts: v1 vs v2 ===")
    print(cmp.sort_values("v1_FDR+margin", ascending=False).to_string())

    # ---- coverage diagnostic: distribution among confident (FDR-passing) cells ----
    print("\n=== marker coverage among FDR-passing cells (to set k_T) ===")
    cov_rows = []
    for label in labels:
        present = sum(g in var_idx for g in MARKER_GENES[label])
        # cells whose winning label is this type AND passed FDR
        sel = (res["top_label"] == label) & res["fdr_pass"]
        c = coverage[label][sel]
        if c.size:
            p10, p25, med = np.percentile(c, [10, 25, 50])
        else:
            p10 = p25 = med = np.nan
        cov_rows.append({"label": label, "set_size": present, "n_conf": int(sel.sum()),
                         "cov_p10": p10, "cov_p25": p25, "cov_median": med,
                         "suggested_k": int(max(2, np.nan_to_num(p10, nan=2)))})
    cov_df = pd.DataFrame(cov_rows)
    print(cov_df.to_string(index=False))

    # ---- save table ----
    out = pd.DataFrame({"x": cx, "y": cy})
    for l in labels:
        out["score_" + l] = S[l].to_numpy()
        out["cov_" + l] = coverage[l]
    out["top_score_label"] = res["top_label"]
    out["top_scaled"] = res["top_scaled"]
    out["second_scaled"] = res["second_scaled"]
    out["fdr_pass"] = res["fdr_pass"]
    out["margin_pass"] = res["margin_pass"]
    out["cov_top"] = cov_top
    out["cov_pass"] = cov_pass
    out["celltype_v1"] = calls_v1
    out["celltype_v2"] = calls_v2
    out.to_csv(f"{OUT_DIR}/cell_scores.csv", index=False)
    thr_df.to_csv(f"{OUT_DIR}/mirrored_fdr_thresholds.csv", index=False)
    cov_df.to_csv(f"{OUT_DIR}/coverage_diagnostic.csv", index=False)

    x, y = cx, cy
    ncols, nrows = 3, int(np.ceil(len(labels) / 3))

    # (1) per-type spatial score maps (diverging, centered at 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()
    for ax, l in zip(axes, labels):
        v = S[l].to_numpy()
        vmax = np.quantile(np.abs(v), 0.99) or 1.0
        o = np.argsort(np.abs(v))
        sc_ = ax.scatter(x[o], y[o], s=2, c=v[o], cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
        fig.colorbar(sc_, ax=ax, shrink=0.7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{l}  (%>0: {100*(v>0).mean():.0f}%)")
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle(f"slice_{SLICE_ID} per-cell score_genes in space (red = high)", fontsize=15)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/score_spatial.png", bbox_inches="tight"); plt.close()

    # (2) per-type FINAL-LABEL spatial maps
    for label in labels:
        mask = calls == label
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
        ax.scatter(x, y, s=1.0, c="#d8d8d8", linewidths=0, rasterized=True, label="other non-tumor")
        ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=2.0, c=TUMOR_COLOR,
                   linewidths=0, rasterized=True, label="pred tumor")
        ax.scatter(x[mask], y[mask], s=2.6, c=CELLTYPE_COLORS.get(label, "#d62728"),
                   linewidths=0, rasterized=True, label=label)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice_{SLICE_ID} {label}  (n={int(mask.sum()):,}, "
                     f"{100*mask.sum()/len(tumor_all):.1f}% of all cells)")
        ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/label_spatial_{safe_name(label)}.png", bbox_inches="tight")
        plt.close()

    # (3) combined label map (v2, incl. ambiguous / low_markers_coverage)
    extra = {"ambiguous": "#ff7f0e", "low_markers_coverage": "#bcbd22"}
    fig, ax = plt.subplots(figsize=(9, 8), dpi=170)
    ax.scatter(x, y, s=1.0, c="#eaeaea", linewidths=0, rasterized=True, label="unknown")
    if tumor_all.any():
        ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.6, c=TUMOR_COLOR,
                   linewidths=0, rasterized=True, label=f"tumor (n={int(tumor_all.sum()):,})")
    for cat, col in extra.items():
        mask = calls == cat
        ax.scatter(x[mask], y[mask], s=1.6, c=col, linewidths=0, rasterized=True, label=cat)
    for label in labels:
        mask = calls == label
        ax.scatter(x[mask], y[mask], s=2.2, c=CELLTYPE_COLORS[label],
                   linewidths=0, rasterized=True, label=label)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_{SLICE_ID} annotation (5-type, FDR+margin+coverage)")
    ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/label_spatial_ALL.png", bbox_inches="tight"); plt.close()

    # (3b) before/after comparison: v1 (FDR+margin) vs v2 (+coverage), per type
    fig, axes = plt.subplots(len(labels), 2, figsize=(13, 3.0 * len(labels)), dpi=140)
    for r, label in enumerate(labels):
        for c, (cl, tag) in enumerate([(calls_v1, "v1 FDR+margin"),
                                       (calls_v2, "v2 +coverage")]):
            ax = axes[r, c]
            m = cl == label
            ax.scatter(x, y, s=0.4, c="#e8e8e8", linewidths=0, rasterized=True)
            ax.scatter(x[m], y[m], s=1.8, c=CELLTYPE_COLORS[label],
                       linewidths=0, rasterized=True)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label} — {tag}  (n={int(m.sum()):,})", fontsize=9)
    fig.suptitle(f"slice_{SLICE_ID}: FDR+margin (left) vs +coverage gate (right)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/compare_v1_v2.png", bbox_inches="tight"); plt.close()

    # (4) score histograms with thresholds
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=140)
    axes = np.asarray(axes).ravel()
    for ax, l in zip(axes, labels):
        ax.hist(S[l], bins=60, color="#4c72b0")
        ax.axvline(0, c="k", ls="--", lw=1)
        if np.isfinite(thresholds[l]):
            ax.axvline(thresholds[l], c="crimson", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"{l}  thr={thresholds[l]:.3f}", fontsize=10)
    for ax in axes[len(labels):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/score_histograms.png", bbox_inches="tight"); plt.close()

    print(f"\nsaved -> {OUT_DIR}\n  score_spatial.png, label_spatial_<type>.png, "
          f"label_spatial_ALL.png, score_histograms.png,\n  cell_scores.csv, "
          f"mirrored_fdr_thresholds.csv, coverage_diagnostic.csv")


if __name__ == "__main__":
    main()
