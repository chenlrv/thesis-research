"""score_genes recipe on slice 1 with combined FDR + margin celltype calls.

normalize_total -> log1p -> sc.tl.score_genes per label
(markers from annotation_cell_marker_genes.MARKER_GENES). Each non-tumor cell is
assigned only if it passes BOTH a 5% mirrored-FDR gate and a MAD-scaled margin
gate (top score >= 1.5x the runner-up, the "50% rule"); else 'unknown'.
"""
import os
import re

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

from annotation_cell_marker_genes import MARKER_GENES

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUT_DIR = "D:/thesis-research/score_genes_slice1"
TUMOR_COL = "pred_tumor_XGBoost"
FDR_CUTOFF = 0.05

CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4",
    "Microglia": "#17becf",
    "Macrophage": "#00a087",
    "Endothelial": "#2ca02c",
    "Pericytes": "#a65628",
    "Ependymal": "#984ea3",
    "Neurons": "#e377c2",
}
TUMOR_COLOR = "#ff0000"


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
        est_fdr = n_mirror / max(n_called, 1)
        if est_fdr <= fdr:
            return float(t), n_called, n_mirror, float(est_fdr)
    return np.inf, 0, 0, np.nan


MARGIN_RATIO = 1.5  # 50% rule: keep the call only if the top score is >=
                    # MARGIN_RATIO x the runner-up (MAD-scaled, sign-safe).


def scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO):
    """Assign each cell to a celltype via two combined gates.

    FDR gate    - the chosen label's raw score >= its mirrored-FDR threshold.
    margin gate - scale every label by its own MAD so all spreads are equal
        (and 0 is preserved), then keep the call only if the top scaled score
        beats the runner-up by >= `ratio`. Sign-safe: the top must be > 0, and
        a runner-up <= 0 makes the winner automatically unambiguous.

    A cell failing either gate is 'unknown'. Returns a results dict.
    """
    labels = list(S.columns)
    raw = S.to_numpy(float)
    mad = 1.4826 * np.median(np.abs(raw - np.median(raw, axis=0)), axis=0)
    fallback = raw.std(axis=0)
    mad = np.where(mad > 0, mad, fallback)
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
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    adata = adata[~tumor].copy()
    cx, cy = cx[~tumor], cy[~tumor]
    print(f"non-tumor cells = {adata.n_obs} (removed {int(tumor.sum())} tumor)")

    # ---- recipe: normalize -> log1p, then score on log-normalized data ----
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = list(MARKER_GENES.keys())
    score_cols = {}
    print("\nscoring (score_genes, ctrl_size=50, n_bins=25):")
    for label in labels:
        genes = [g for g in MARKER_GENES[label] if g in adata.var_names]
        missing = [g for g in MARKER_GENES[label] if g not in adata.var_names]
        col = "score_" + label.replace(" ", "_")
        sc.tl.score_genes(adata, gene_list=genes, score_name=col,
                          ctrl_size=50, n_bins=25)
        score_cols[label] = col
        print(f"  {label:16s}: {len(genes)} genes" +
              (f"  (missing {missing})" if missing else ""))

    S = adata.obs[[score_cols[l] for l in labels]].copy()
    S.columns = labels

    # ---- summary stats ----
    print("\n=== per-label score summary ===")
    summ = pd.DataFrame({
        "mean": S.mean(), "std": S.std(),
        "min": S.min(), "max": S.max(),
        "%>0": 100 * (S > 0).mean(),
    }).round(3)
    print(summ)
    print("\n=== cross-score correlation (Pearson) ===")
    corr = S.corr()
    print(corr.round(2))

    # ---- mirrored-FDR thresholds + top-score labels ----
    threshold_rows = []
    thresholds = {}
    for label in labels:
        t, n_called, n_mirror, est_fdr = mirrored_fdr_threshold(
            S[label].to_numpy(), fdr=FDR_CUTOFF
        )
        thresholds[label] = t
        threshold_rows.append({
            "label": label,
            "threshold": t,
            "n_score_ge_threshold": n_called,
            "n_score_le_minus_threshold": n_mirror,
            "estimated_fdr": est_fdr,
        })

    threshold_df = pd.DataFrame(threshold_rows)
    print(f"\n=== mirrored-FDR thresholds (FDR <= {FDR_CUTOFF}) ===")
    print(threshold_df.round(4).to_string(index=False))

    res = scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO)
    calls = res["calls"]

    print(f"\n=== assignment: FDR (<= {FDR_CUTOFF}) AND "
          f"{int((MARGIN_RATIO - 1) * 100)}% margin "
          f"(top >= {MARGIN_RATIO}x runner-up, MAD-scaled) ===")
    print(f"  FDR gate pass:    {int(res['fdr_pass'].sum()):>7,} "
          f"({100 * res['fdr_pass'].mean():5.1f}%)")
    print(f"  margin gate pass: {int(res['margin_pass'].sum()):>7,} "
          f"({100 * res['margin_pass'].mean():5.1f}%)")
    print(f"  both (annotated): {int((calls != 'unknown').sum()):>7,} "
          f"({100 * (calls != 'unknown').mean():5.1f}%)")
    print("\n=== final labels ===")
    print(pd.Series(calls).value_counts().to_string())

    # which label pairs are the near-ties that fail the margin gate
    failed = (~res["margin_pass"]) & (res["top_scaled"] > 0)
    if failed.any():
        pairs = pd.Series(
            [tuple(sorted((a, b)))
             for a, b in zip(res["top_label"][failed], res["second_label"][failed])]
        ).value_counts().head(10)
        print("\ntop colliding label pairs (fail margin gate, top > 0):")
        print(pairs.to_string())

    out = pd.DataFrame({"x": cx, "y": cy})
    for l in labels:
        out["score_" + l.replace(" ", "_")] = S[l].to_numpy()
    out["top_score_label"] = res["top_label"]
    out["second_label"] = res["second_label"]
    out["top_scaled"] = res["top_scaled"]
    out["second_scaled"] = res["second_scaled"]
    out["margin_scaled"] = res["margin_scaled"]
    out["fdr_pass"] = res["fdr_pass"]
    out["margin_pass"] = res["margin_pass"]
    out["celltype"] = calls
    out.to_csv(f"{OUT_DIR}/cell_scores.csv", index=False)
    threshold_df.to_csv(f"{OUT_DIR}/mirrored_fdr_thresholds.csv", index=False)

    # ---- figures ----
    # (1) score distributions
    ncols = 3
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=140)
    axes = np.asarray(axes).ravel()
    for ax, l in zip(axes, labels):
        n_pass = int((S[l] >= thresholds[l]).sum())
        n_filtered = int(((S[l] > 0) & (S[l] < thresholds[l])).sum())
        ax.hist(S[l], bins=60, color="#4c72b0")
        ax.axvline(0, c="k", ls="--", lw=1)
        if np.isfinite(thresholds[l]):
            ax.axvline(thresholds[l], c="crimson", ls="-", lw=1)
        ax.set_yscale("log")
        ax.set_title(
            f"{l}\n"
            f"pass >= threshold: {n_pass:,}\n"
            f"0 < filtered < threshold: {n_filtered:,}",
            fontsize=10,
        )
        ax.set_xlabel("score")
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle(
        "slice_1 per-cell score_genes distributions",
        fontsize=14,
    )
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/score_histograms.png",
                                    bbox_inches="tight"); plt.close()

    # (2) spatial score maps (diverging, centered at 0)
    x, y = cx, cy
    x_all, y_all = cx_all, cy_all
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()
    for ax, l in zip(axes, labels):
        v = S[l].to_numpy()
        vmax = np.quantile(np.abs(v), 0.99) or 1.0
        order = np.argsort(np.abs(v))     # strong scores on top
        sctr = ax.scatter(x[order], y[order], s=2, c=v[order], cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
        fig.colorbar(sctr, ax=ax, shrink=0.7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{l}  (%>0: {100*(v>0).mean():.0f}%)")
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle("slice_1 per-cell scores in space (red = high)", fontsize=15)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/score_spatial.png",
                                    bbox_inches="tight"); plt.close()

    # (3) one final-label spatial result plot per cell type
    for label in labels:
        mask = calls == label
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
        ax.scatter(x, y, s=1.0, c="#d0d0d0", linewidths=0, rasterized=True,
                   label="other non-tumor")
        ax.scatter(x_all[tumor_all], y_all[tumor_all], s=2.4, c=TUMOR_COLOR,
                   linewidths=0, rasterized=True, label="tumor")
        ax.scatter(x[mask], y[mask], s=2.4,
                   c=CELLTYPE_COLORS.get(label, "#d62728"), linewidths=0,
                   rasterized=True, label=label)
        pct_all = 100 * mask.sum() / len(tumor_all)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice_1 {label} cells (n={int(mask.sum())}, {pct_all:.1f}% all cells)")
        ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/result_spatial_{safe_name(label)}.png",
                    bbox_inches="tight")
        plt.close()

    # (4) cross-score correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("cross-score correlation")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/score_correlation.png",
                                    bbox_inches="tight"); plt.close()

    # (5) assignment-margin diagnostics (MAD-scaled top vs runner-up)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].hist(res["margin_scaled"], bins=80, color="#4c72b0")
    axes[0].set_yscale("log"); axes[0].set_xlabel("MAD-scaled  top - second")
    axes[0].set_title("scaled top-2 margin")
    pos = res["second_scaled"] > 0
    ratio = res["top_scaled"][pos] / res["second_scaled"][pos]
    axes[1].hist(np.clip(ratio, 0, 6), bins=80, color="#55a868")
    axes[1].axvline(MARGIN_RATIO, c="crimson", ls="--", lw=1, label=f"{MARGIN_RATIO}x")
    axes[1].set_yscale("log"); axes[1].set_xlabel("top / second  (second > 0)")
    axes[1].set_title("top : runner-up ratio"); axes[1].legend()
    fig.suptitle("slice_1 assignment margin (FDR + ratio gate)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/score_margin_hist.png",
                                    bbox_inches="tight"); plt.close()

    print(f"\nsaved cell_scores.csv, mirrored_fdr_thresholds.csv, "
          f"score_histograms.png, score_spatial.png, score_correlation.png, "
          f"score_margin_hist.png, and result_spatial_<celltype>.png "
          f"files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
