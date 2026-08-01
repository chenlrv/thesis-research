"""Broad celltype annotation for slices 1 and 3, Microglia+Macrophage merged
into one 'Myeloid' label.

Same paradigm as run_score_genes_slice1.py: normalize_total -> log1p ->
sc.tl.score_genes per label, then assign each non-tumor cell only if it passes a
5% mirrored-FDR gate AND a MAD-scaled 50% margin gate (top >= 1.5x runner-up).
Additionally, a cell called 'Myeloid' must be CD45 (Ptprc) positive to keep that
call (REQUIRE_CD45). No raw-count co-detection gate.
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


SLICES = {
    "slice_1": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad",
    "slice_3": "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad",
}
OUT_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
TUMOR_COL = "pred_tumor_XGBoost"
FDR_CUTOFF = 0.05
# CD45 (Ptprc) myeloid gate: a cell called "Myeloid" by the score_genes + FDR +
# margin paradigm must also be CD45-positive (raw count > 0) to keep that call.
# CAUTION: microglia are CD45-low, so this can drop true myeloid cells -- the CD45+
# rate among Myeloid calls (printed below) shows the cost. Set False to disable.
IMMUNE_GATE_GENE = "Ptprc"
REQUIRE_CD45 = False
# 50% rule: a cell keeps its call only if its top score is >= MARGIN_RATIO x the
# runner-up, after scaling every label by its MAD so all spreads are equal. This
# margin gate is combined with the mirrored-FDR gate (same paradigm as slice1).
MARGIN_RATIO = 1.5

MARKER_GENES_STAGE1 = {
    "Astrocytes": [
        "GFAP", "Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9",
    ],
    "Myeloid": [
        "Csf1r", "Tyrobp", "Fcer1g", "Aif1", "Cx3cr1",
    ],
    "Endothelial": [
        "Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Tie1", "Esam", "Vwf",
        "Slc2a1", "Clec14a", "Adgrl4", "Ldb2", "Icam2", "Eng", "Cd34",
        "Ramp2", "Klf2",
    ],
    "Pericytes": [
        "Rgs5", "Pdgfrb", "Notch3", "Vtn",
    ],
    "Ependymal": [
        "Adgrv1", "Cd24a", "Ttr", "Epcam", "Krt8", "Krt18", "Krt19", "Cldn4",
    ],
    "Neurons": [
        "Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1",
        "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst",
    ],
}

CELLTYPE_COLORS = {
    "Astrocytes": "#1f77b4",
    "Myeloid": "#00a087",
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


def raw_vs_scaled_argmax(S, scaled_top_label):
    """Raw-score argmax winner vs the MAD-scaled argmax winner, per cell.

    The final calls choose the winner on MAD-scaled scores. MAD-scaling divides
    each label by its own spread, which can promote a label with a tight (rare)
    null and demote an abundant one -- so a large, directional disagreement with
    the raw-score winner is the signal that scaling is reshaping the annotation
    (and, since Stage 2 consumes the Myeloid call, the subtype counts downstream).
    Returns (raw_top_label, flip_mask).
    """
    labels = np.asarray(S.columns)
    raw_top = labels[S.to_numpy(float).argmax(axis=1)]
    return raw_top, raw_top != np.asarray(scaled_top_label)


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def resolve_genes(genes, var_names):
    """Resolve marker names case-insensitively against panel gene symbols."""
    lookup = {str(g).lower(): str(g) for g in var_names}
    return [lookup[g.lower()] for g in genes if g.lower() in lookup]


def raw_expr_vector(adata, gene):
    """Return raw counts for one gene from the current AnnData matrix."""
    present = resolve_genes([gene], adata.var_names)
    if not present:
        return None, None
    X = adata[:, present[0]].X
    if hasattr(X, "toarray"):
        values = X.toarray().ravel()
    else:
        values = np.asarray(X).ravel()
    return values, present[0]


def run_slice(slice_name, slice_path):
    out_dir = os.path.join(OUT_ROOT, slice_name)
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(slice_path, "r") as h5:
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
    print(f"\n=== {slice_name} ===")
    print(f"non-tumor cells = {adata.n_obs} (removed {int(tumor.sum())} tumor)")

    # CD45 (Ptprc) raw counts -- the myeloid confirmation gate (applied to the
    # Myeloid call after scoring, below). Read here while X is still raw counts.
    cd45_vals, cd45_name = raw_expr_vector(adata, IMMUNE_GATE_GENE)
    cd45_pos = (cd45_vals > 0) if cd45_vals is not None \
        else np.zeros(adata.n_obs, dtype=bool)
    if cd45_vals is None:
        print(f"\n{IMMUNE_GATE_GENE} (CD45) absent from panel -- CD45 gate disabled")
    else:
        print(f"\nCD45 ({cd45_name}) raw count > 0: {int(cd45_pos.sum()):,} "
              f"({100 * cd45_pos.mean():.1f}% of non-tumor)")

    # score_genes expects normalized/log1p magnitudes for expression-matched controls.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    labels = list(MARKER_GENES_STAGE1.keys())
    score_cols = {}
    print("\nscoring (score_genes, ctrl_size=50, n_bins=25):")
    for label in labels:
        genes = resolve_genes(MARKER_GENES_STAGE1[label], adata.var_names)
        missing = [g for g in MARKER_GENES_STAGE1[label] if g.lower() not in
                   {v.lower() for v in adata.var_names}]
        col = "score_" + label.replace(" ", "_")
        sc.tl.score_genes(adata, gene_list=genes, score_name=col,
                          ctrl_size=50, n_bins=25)
        score_cols[label] = col
        print(f"  {label:16s}: {len(genes)} genes" +
              (f"  (missing {missing})" if missing else ""))

    S = adata.obs[[score_cols[l] for l in labels]].copy()
    S.columns = labels

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
    failed = (~res["margin_pass"]) & (res["top_scaled"] > 0)
    if failed.any():
        pairs = pd.Series(
            [tuple(sorted((a, b)))
             for a, b in zip(res["top_label"][failed], res["second_label"][failed])]
        ).value_counts().head(8)
        print("  top colliding pairs (fail margin gate, top > 0):")
        print(pairs.to_string())

    # ---- CD45 myeloid gate: a Myeloid call must be CD45+ (immune) ----
    myeloid_call = res["calls"] == "Myeloid"
    cd45_in_myeloid = (100 * cd45_pos[myeloid_call].mean()) if myeloid_call.any() else 0.0
    cd45_drop = myeloid_call & ~cd45_pos
    print(f"\n=== CD45 myeloid gate (REQUIRE_CD45={REQUIRE_CD45}) ===")
    print(f"  Myeloid score calls: {int(myeloid_call.sum()):,}; "
          f"CD45+ {int((myeloid_call & cd45_pos).sum()):,} ({cd45_in_myeloid:.1f}%); "
          f"CD45- {int(cd45_drop.sum()):,}")
    if REQUIRE_CD45 and cd45_drop.any():
        calls = calls.copy()
        calls[cd45_drop] = "unknown"
        print(f"  dropped {int(cd45_drop.sum()):,} CD45- Myeloid calls -> unknown")

    # ---- raw-argmax vs MAD-scaled-argmax diagnostic ----
    # Does dividing each label by its MAD change which label wins? If the winner
    # flips heavily in one direction, the full-distribution MAD denominator (which
    # mixes noise spread with real signal) is reshaping the calls toward tight-null
    # labels -- a recall/precision risk that propagates into Stage 2.
    raw_top, flip = raw_vs_scaled_argmax(S, res["top_label"])
    scaled_top = np.asarray(res["top_label"])
    pos = res["top_scaled"] > 0
    annotated = calls != "unknown"
    flip_annot_pct = 100 * flip[annotated].mean() if annotated.any() else 0.0
    net = (pd.Series(scaled_top[pos]).value_counts()
           .subtract(pd.Series(raw_top[pos]).value_counts(), fill_value=0)
           .reindex(labels).fillna(0).astype(int))
    print("\n=== raw-argmax vs MAD-scaled-argmax ===")
    print(f"  winner flips (all non-tumor):       {int(flip.sum()):>7,} "
          f"({100 * flip.mean():5.1f}%)")
    print(f"  winner flips (scaled top > 0):      {int((flip & pos).sum()):>7,} "
          f"({100 * (flip & pos).mean():5.1f}%)")
    print(f"  winner flips among annotated cells: {int((flip & annotated).sum()):>7,} "
          f"({flip_annot_pct:5.1f}% of annotated)")
    print("  net argmax change per label (scaled - raw, scaled top > 0; "
          "+ MAD boosts / - MAD suppresses):")
    print(net.to_string())
    mye_raw, mye_scaled = raw_top == "Myeloid", scaled_top == "Myeloid"
    print(f"  Myeloid argmax: raw {int(mye_raw.sum()):,} -> scaled "
          f"{int(mye_scaled.sum()):,} (lost {int((mye_raw & ~mye_scaled).sum()):,}, "
          f"gained {int((~mye_raw & mye_scaled).sum()):,})")
    if flip.any():
        moves = pd.Series([f"{r} -> {s}" for r, s in zip(raw_top[flip], scaled_top[flip])])
        print("  top winner flips (raw -> scaled):")
        print(moves.value_counts().head(8).to_string())

    print("\n=== final top-score labels ===")
    print(pd.Series(calls).value_counts().to_string())

    out = pd.DataFrame({"x": cx, "y": cy})
    for label in labels:
        out["score_" + label.replace(" ", "_")] = S[label].to_numpy()
    out["top_score_label"] = res["top_label"]
    out["raw_top_label"] = raw_top
    out["argmax_flip"] = flip
    out["second_label"] = res["second_label"]
    out["top_scaled"] = res["top_scaled"]
    out["second_scaled"] = res["second_scaled"]
    out["margin_scaled"] = res["margin_scaled"]
    out["fdr_pass"] = res["fdr_pass"]
    out["margin_pass"] = res["margin_pass"]
    out["cd45_pos"] = cd45_pos
    # celltype = FDR gate AND 50% MAD-scaled margin gate; a Myeloid call also
    # requires CD45+ (see REQUIRE_CD45).
    out["celltype"] = calls
    out.to_csv(f"{out_dir}/cell_scores.csv", index=False)
    threshold_df.to_csv(f"{out_dir}/mirrored_fdr_thresholds.csv", index=False)

    ncols = 3
    nrows = int(np.ceil(len(labels) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), dpi=140)
    axes = np.asarray(axes).ravel()
    for ax, label in zip(axes, labels):
        n_pass = int((S[label] >= thresholds[label]).sum())
        n_filtered = int(((S[label] > 0) & (S[label] < thresholds[label])).sum())
        ax.hist(S[label], bins=60, color="#4c72b0")
        ax.axvline(0, c="k", ls="--", lw=1)
        if np.isfinite(thresholds[label]):
            ax.axvline(thresholds[label], c="crimson", ls="-", lw=1)
        ax.set_yscale("log")
        ax.set_title(
            f"{label}\n"
            f"pass >= threshold: {n_pass:,}\n"
            f"0 < filtered < threshold: {n_filtered:,}",
            fontsize=10,
        )
        ax.set_xlabel("score")
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle(f"{slice_name} per-cell score_genes distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_histograms.png", bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), dpi=150)
    axes = np.asarray(axes).ravel()
    for ax, label in zip(axes, labels):
        v = S[label].to_numpy()
        vmax = np.quantile(np.abs(v), 0.99) or 1.0
        order = np.argsort(np.abs(v))
        sctr = ax.scatter(cx[order], cy[order], s=2, c=v[order], cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
        fig.colorbar(sctr, ax=ax, shrink=0.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}  (%>0: {100 * (v > 0).mean():.0f}%)")
    for ax in axes[len(labels):]:
        ax.axis("off")
    fig.suptitle(f"{slice_name} per-cell scores in space (red = high)", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_spatial.png", bbox_inches="tight")
    plt.close()

    for label in labels:
        mask = calls == label
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
        ax.scatter(cx, cy, s=1.0, c="#d0d0d0", linewidths=0, rasterized=True,
                   label="other non-tumor")
        ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=2.4, c=TUMOR_COLOR,
                   linewidths=0, rasterized=True, label="tumor")
        ax.scatter(cx[mask], cy[mask], s=2.4,
                   c=CELLTYPE_COLORS.get(label, "#d62728"), linewidths=0,
                   rasterized=True, label=label)
        pct_all = 100 * mask.sum() / len(tumor_all)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{slice_name} {label} cells (n={int(mask.sum())}, {pct_all:.1f}% all cells)")
        ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/result_spatial_{safe_name(label)}.png",
                    bbox_inches="tight")
        plt.close()

    # ---- combined annotation map: every cell colored by final argmax label ----
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    unk = calls == "unknown"
    ax.scatter(cx[unk], cy[unk], s=0.6, c="#e8e8e8", linewidths=0, rasterized=True,
               label=f"unknown ({int(unk.sum()):,})")
    ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.5, c=TUMOR_COLOR,
               linewidths=0, rasterized=True, label=f"tumor ({int(tumor_all.sum()):,})")
    for lbl in labels:
        m = calls == lbl
        if not m.any():
            continue
        ax.scatter(cx[m], cy[m], s=2.2, c=CELLTYPE_COLORS.get(lbl, "#333333"),
                   linewidths=0, rasterized=True, label=f"{lbl} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{slice_name} annotated cells "
                 f"(FDR + {int((MARGIN_RATIO - 1) * 100)}% margin)")
    ax.legend(loc="lower right", markerscale=5, frameon=True, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/result_spatial_all_labels.png", bbox_inches="tight")
    plt.close()

    # ---- CD45 split of the Myeloid score call (microglia are CD45-low) ----
    if cd45_vals is not None:
        mc = res["calls"] == "Myeloid"   # pre-CD45-gate Myeloid argmax calls
        fig, ax = plt.subplots(figsize=(8, 7), dpi=170)
        ax.scatter(cx, cy, s=1.0, c="#d0d0d0", linewidths=0, rasterized=True,
                   label="other non-tumor")
        ax.scatter(cx[mc & ~cd45_pos], cy[mc & ~cd45_pos], s=2.4, c="#f39c12",
                   linewidths=0, rasterized=True,
                   label=f"Myeloid CD45- ({int((mc & ~cd45_pos).sum()):,})")
        ax.scatter(cx[mc & cd45_pos], cy[mc & cd45_pos], s=2.4, c="#00a087",
                   linewidths=0, rasterized=True,
                   label=f"Myeloid CD45+ ({int((mc & cd45_pos).sum()):,})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{slice_name} Myeloid score call split by CD45 ({cd45_name})")
        ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/myeloid_call_cd45_split.png", bbox_inches="tight")
        plt.close()

    # ---- assignment-margin diagnostics (MAD-scaled top vs runner-up) ----
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
    fig.suptitle(f"{slice_name} assignment margin (FDR + ratio gate)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_margin_hist.png", bbox_inches="tight")
    plt.close()

    # ---- raw vs MAD-scaled argmax: net per-label change in the winner ----
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bar_colors = ["#55a868" if v >= 0 else "#c44e52" for v in net.values]
    ax.bar(range(len(net)), net.values, color=bar_colors)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(len(net)))
    ax.set_xticklabels(net.index, rotation=45, ha="right")
    ax.set_ylabel("scaled argmax - raw argmax  (cells)")
    ax.set_title(f"{slice_name} MAD-scaling effect on argmax winner\n"
                 f"(+ MAD boosts / - MAD suppresses; scaled top > 0; "
                 f"{int(flip.sum()):,} cells flip)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/argmax_raw_vs_scaled.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("cross-score correlation")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/score_correlation.png", bbox_inches="tight")
    plt.close()

    print(f"\nsaved outputs -> {out_dir}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for slice_name, slice_path in SLICES.items():
        run_slice(slice_name, slice_path)


if __name__ == "__main__":
    main()
