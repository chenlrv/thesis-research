"""Stage 2: subtype the Stage-1 myeloid cells with the SAME machinery as Stage 1.

Stage 1 (run_score_genes_myeloid_stage1.py) calls broad cell types with
score_genes + mirrored-FDR + MAD-scaled 1.5x margin (competitive across 6 types)
and writes per-cell labels to score_genes_myeloid_stage1/<slice>/cell_scores.csv.
This script reads that file, takes exactly the cells Stage 1 labeled 'Myeloid',
and runs the IDENTICAL paradigm one level down on the 3 subtypes:

  normalize_total -> log1p -> sc.tl.score_genes per subtype, then assign each cell
  only if it passes a 5% mirrored-FDR gate AND a MAD-scaled 50% margin gate
  (top >= 1.5x runner-up). A cell failing either gate -> 'myeloid_unresolved'.

This makes both stages methodologically consistent (broad -> fine, same engine):
the call now uses marker *magnitude* (not just presence) and includes the same
separation check, instead of the old argmax-over-detected-counts gate.

Runs on RAW counts -> normalize within the myeloid subset (decontX not needed: a
microglia cell's TMEM119 is native, not ambient). Subtype sets are specificity-
scrutinised for DISCRIMINATION within the myeloid compartment (not just myeloid-
ness): Cx3cr1 and Selplg dropped from microglia (pan-myeloid / pan-leukocyte,
don't discriminate); Apoe and Cd14 dropped from MDM (Apoe astrocyte cross-
reactive; Cd14 activation-induced and not MDM-specific -- it over-called MDM in
the healthy slice_3, 68% of MDM calls Cd14+ but 59% Ccr2-, giving more MDM in
healthy than tumor tissue). MDM now rests on Ccr2 (+ sparse Plac8); microglia on
TMEM119 + P2rx5 -- both conservative, as the panel lacks the specific microglia
markers (P2ry12, Sall1, Hexb, Siglech all absent).

Caveat unchanged by the method swap: microglia<->MDM is the fragile boundary
(microglia markers are all homeostatic and drop on tumor activation) -- a panel
limitation, which the cluster-level variant (run_myeloid_stage2_cluster.py)
addresses with aggregation rather than per-cell calls.
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
OUT_ROOT = "D:/thesis-research/score_genes_myeloid_stage2_gate"
STAGE1_ROOT = "D:/thesis-research/score_genes_myeloid_stage1"
STAGE1_MYELOID_LABEL = "Myeloid"
TUMOR_COL = "pred_tumor_XGBoost"
FDR_CUTOFF = 0.05
# 50% rule: keep a call only if its top score is >= MARGIN_RATIO x the runner-up,
# after MAD-scaling every subtype so spreads are comparable (same as Stage 1).
MARGIN_RATIO = 1.5
UNRESOLVED_LABEL = "myeloid_unresolved"

SUBTYPE_MARKERS = {
    "BAM":       ["Mrc1", "Cd163", "Lyve1", "Pf4"],
    "MDM":       ["Ccr2", "Plac8"],
    "microglia": ["TMEM119", "P2rx5"],
}

COLORS = {
    "microglia": "#1f77b4",
    "BAM": "#ff7f0e",
    "MDM": "#9467bd",
    UNRESOLVED_LABEL: "#bdbdbd",
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


def resolve_genes(genes, var_names):
    lut = {str(v).lower(): str(v) for v in var_names}
    return [lut[g.lower()] for g in genes if g.lower() in lut]


def count_detected_genes(adata, genes):
    present = resolve_genes(genes, adata.var_names)
    if not present:
        return np.zeros(adata.n_obs, dtype=int), present
    X = adata[:, present].X
    detected = X > 0
    n = np.asarray(detected.sum(axis=1)).ravel() if hasattr(detected, "sum") \
        else detected.sum(axis=1)
    return n.astype(int), present


def _dense(M):
    return M.toarray() if hasattr(M, "toarray") else np.asarray(M)


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def load_stage1_myeloid_mask(name, cx, cy):
    """Boolean mask of cells Stage 1 labeled 'Myeloid', aligned to this script's
    non-tumor cell order.

    Both stages read the same h5ad and drop tumor with the same `~tumor` mask, so
    Stage 1's cell_scores.csv rows line up 1:1 with the non-tumor cells here. We
    verify length AND x/y before trusting the order, and fail loudly otherwise so
    a stale/regenerated Stage 1 can never silently mis-map onto these cells.
    """
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
            "regenerate Stage 1 (h5ad or tumor column changed)."
        )
    if not (np.allclose(df["x"].to_numpy(), cx) and np.allclose(df["y"].to_numpy(), cy)):
        raise ValueError(
            f"{name}: Stage-1 x/y do not match current non-tumor coords -- "
            "cell ordering mismatch; regenerate Stage 1."
        )
    return df["celltype"].to_numpy().astype(str) == STAGE1_MYELOID_LABEL


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


def scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO, unresolved_label="unknown"):
    """Assign each cell to a subtype via two combined gates (same as Stage 1).

    FDR gate    - the chosen subtype's raw score >= its mirrored-FDR threshold.
    margin gate - scale every subtype by its own MAD so spreads are equal (0
        preserved), then keep the call only if the top scaled score beats the
        runner-up by >= `ratio`. Sign-safe: the top must be > 0, and a runner-up
        <= 0 makes the winner automatically unambiguous.

    A cell failing either gate -> `unresolved_label`. Returns a results dict.
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

    calls = np.where(margin_pass & fdr_pass, top_label, unresolved_label).astype(object)
    return {
        "calls": calls, "top_label": top_label, "second_label": second_label,
        "top_scaled": top, "second_scaled": second, "margin_scaled": top - second,
        "fdr_pass": fdr_pass, "margin_pass": margin_pass,
    }


def plot_overlap_before(out_dir, tag, res, ratio=MARGIN_RATIO):
    """Mirror of Stage 1's overlap_before_threshold.png: how many myeloid cells
    have one clearly-dominant subtype vs >=2 close subtypes, BEFORE any gate."""
    top = res["top_scaled"]
    second = res["second_scaled"]
    top_label = res["top_label"]
    second_label = res["second_label"]
    n = max(len(top), 1)

    no_group = top <= 0
    ambiguous = (top > 0) & (second > 0) & (top < ratio * second)
    clear = (top > 0) & ~ambiguous
    buckets = [
        ("One clear subtype", int(clear.sum()), "#00a087"),
        ("≥2 close subtypes\n(ambiguous)", int(ambiguous.sum()), "#e74c3c"),
        ("No subtype positive", int(no_group.sum()), "#bbbbbb"),
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), dpi=150,
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    names = [b[0] for b in buckets]
    vals = [b[1] for b in buckets]
    cols = [b[2] for b in buckets]
    bars = axA.bar(names, vals, color=cols, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        axA.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n{100 * v / n:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axA.set_ylabel("number of myeloid cells")
    axA.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    axA.set_title(f"Per cell: is one subtype's score clearly highest?\n"
                  f"(close = top subtype < {ratio}× the runner-up)", fontsize=11)
    axA.margins(x=0.05)

    groups = sorted(set(top_label) | set(second_label))
    idx = {g: i for i, g in enumerate(groups)}
    K = len(groups)
    M = np.zeros((K, K), int)
    for a, b in zip(top_label[ambiguous], second_label[ambiguous]):
        i, j = idx[a], idx[b]
        M[i, j] += 1
        M[j, i] += 1
    np.fill_diagonal(M, 0)
    im = axB.imshow(M, cmap="Reds")
    axB.set_xticks(range(K)); axB.set_xticklabels(groups, rotation=45, ha="right")
    axB.set_yticks(range(K)); axB.set_yticklabels(groups)
    for i in range(K):
        for j in range(K):
            if M[i, j]:
                axB.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=8,
                         color="white" if M[i, j] > M.max() * 0.5 else "black")
    fig.colorbar(im, ax=axB, shrink=0.8, label="# ambiguous cells")
    axB.set_title(f"Which subtype pairs have close scores?\n"
                  f"({int(ambiguous.sum()):,} ambiguous cells)", fontsize=11)

    fig.suptitle(f"{tag} — score overlap between myeloid subtypes (BEFORE threshold)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "overlap_before_threshold.png"), bbox_inches="tight")
    plt.close()


def plot_assignment_after(out_dir, tag, calls, res, unresolved_label, ratio=MARGIN_RATIO):
    """Mirror of Stage 1's assignment_after_threshold.png: final subtype counts,
    and why the leftover cells are unresolved (weak vs ambiguous)."""
    annotated = calls != unresolved_label
    fdr = res["fdr_pass"].astype(bool)
    margin = res["margin_pass"].astype(bool)
    # every unresolved cell is either weak (fails FDR) or ambiguous (passes FDR
    # but fails the margin) -- exclusive and covering all unresolved cells.
    weak = (~annotated) & (~fdr)
    ambiguous_unk = (~annotated) & fdr & (~margin)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), dpi=150,
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    vc = pd.Series(calls[annotated]).value_counts()
    cats = list(vc.index) + [unresolved_label]
    counts = list(vc.values) + [int((~annotated).sum())]
    colors = ["#4c72b0"] * len(vc) + ["#bbbbbb"]
    bars = axA.bar(range(len(cats)), counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, counts):
        axA.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}",
                 ha="center", va="bottom", fontsize=9)
    axA.set_ylabel("number of cells")
    axA.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)
    axA.set_xticks(range(len(cats)))
    axA.set_xticklabels(cats, rotation=45, ha="right")
    axA.set_title(f"Final assignment after FDR + {int((ratio - 1) * 100)}% margin\n"
                  f"({int(annotated.sum()):,} annotated, "
                  f"{int((~annotated).sum()):,} {unresolved_label})", fontsize=11)

    reasons = [
        ("Ambiguous\n(≥2 close subtypes)", int(ambiguous_unk.sum()), "#e74c3c"),
        ("Weak signal\n(below FDR)", int(weak.sum()), "#f39c12"),
    ]
    rn = [r[0] for r in reasons]
    rv = [r[1] for r in reasons]
    rc = [r[2] for r in reasons]
    bars = axB.bar(rn, rv, color=rc, edgecolor="black", linewidth=0.5)
    tot_unk = max(int((~annotated).sum()), 1)
    for bar, v in zip(bars, rv):
        axB.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n{100 * v / tot_unk:.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axB.set_ylabel(f"number of {unresolved_label} cells")
    axB.set_ylim(0, max(rv) * 1.18 if max(rv) > 0 else 1)
    axB.set_title(f"Why cells were left '{unresolved_label}'", fontsize=11)

    fig.suptitle(f"{tag} — assignment AFTER threshold (ambiguous → {unresolved_label})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "assignment_after_threshold.png"), bbox_inches="tight")
    plt.close()


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
    adata = adata[~tumor].copy()
    cx, cy = cx[~tumor], cy[~tumor]

    # ---- Stage-1 myeloid calls (consumed directly; NOT re-derived) ----
    myeloid = load_stage1_myeloid_mask(name, cx, cy)
    print(f"non-tumor = {adata.n_obs}; Stage-1 Myeloid = {int(myeloid.sum()):,} "
          f"({100 * myeloid.mean():.1f}% of non-tumor)")

    mye = adata[myeloid].copy()
    cxm, cym = cx[myeloid], cy[myeloid]
    subtypes = list(SUBTYPE_MARKERS)

    # ---- marker detection rate (raw counts) -- sparsity context ----
    print("\nsubtype marker detection among myeloid cells (raw counts):")
    for st in subtypes:
        ndet, present = count_detected_genes(mye, SUBTYPE_MARKERS[st])
        print(f"  {st:11s} {present}: {100 * (ndet >= 1).mean():5.1f}% have >=1, "
              f"mean detected = {ndet.mean():.2f}")

    # ---- score_genes (same engine as Stage 1, normalized within myeloid) ----
    sc.pp.normalize_total(mye, target_sum=1e4)
    sc.pp.log1p(mye)
    score_cols = {}
    print("\nscoring (score_genes, ctrl_size=50, n_bins=25):")
    for st in subtypes:
        genes = resolve_genes(SUBTYPE_MARKERS[st], mye.var_names)
        missing = [g for g in SUBTYPE_MARKERS[st]
                   if g.lower() not in {v.lower() for v in mye.var_names}]
        col = "score_" + st
        sc.tl.score_genes(mye, gene_list=genes, score_name=col, ctrl_size=50, n_bins=25)
        score_cols[st] = col
        print(f"  {st:11s}: {len(genes)} genes" +
              (f"  (missing {missing})" if missing else ""))

    S = mye.obs[[score_cols[st] for st in subtypes]].copy()
    S.columns = subtypes

    print("\n=== per-subtype score summary ===")
    summ = pd.DataFrame({
        "mean": S.mean(), "std": S.std(), "min": S.min(), "max": S.max(),
        "%>0": 100 * (S > 0).mean(),
    }).round(3)
    print(summ)
    corr = S.corr()
    print("\n=== cross-score correlation (Pearson) ===")
    print(corr.round(2))

    # ---- mirrored-FDR thresholds ----
    thresholds = {}
    rows = []
    for st in subtypes:
        t, nc, nm, est_fdr = mirrored_fdr_threshold(S[st].to_numpy(), fdr=FDR_CUTOFF)
        thresholds[st] = t
        rows.append({"subtype": st, "threshold": t, "n_score_ge_threshold": nc,
                     "n_score_le_minus_threshold": nm, "estimated_fdr": est_fdr})
    threshold_df = pd.DataFrame(rows)
    print(f"\n=== mirrored-FDR thresholds (FDR <= {FDR_CUTOFF}) ===")
    print(threshold_df.round(4).to_string(index=False))

    # ---- FDR + MAD-scaled margin gate (identical to Stage 1) ----
    res = scaled_margin_calls(S, thresholds, ratio=MARGIN_RATIO,
                              unresolved_label=UNRESOLVED_LABEL)
    labels = res["calls"]
    assigned = labels != UNRESOLVED_LABEL

    print(f"\n=== assignment: FDR (<= {FDR_CUTOFF}) AND "
          f"{int((MARGIN_RATIO - 1) * 100)}% margin (top >= {MARGIN_RATIO}x runner-up) ===")
    print(f"  FDR gate pass:    {int(res['fdr_pass'].sum()):>7,} "
          f"({100 * res['fdr_pass'].mean():5.1f}%)")
    print(f"  margin gate pass: {int(res['margin_pass'].sum()):>7,} "
          f"({100 * res['margin_pass'].mean():5.1f}%)")
    print(f"  both (assigned):  {int(assigned.sum()):>7,} "
          f"({100 * assigned.mean():5.1f}%)")

    # ---- composition ----
    comp = pd.Series(labels).value_counts()
    print("\n=== Stage-2 subtype composition (within myeloid) ===")
    for lbl in comp.index:
        print(f"  {lbl:20s}: {comp[lbl]:7d}  ({100 * comp[lbl] / comp.sum():5.1f}%)")
    n_micro = int(comp.get("microglia", 0))
    n_mac = int(comp.get("BAM", 0) + comp.get("MDM", 0))
    print(f"microglia : (BAM+MDM) = {n_micro}:{n_mac} = "
          f"{n_micro / max(n_mac, 1):.2f}  (healthy brain: >> 1)")

    # ---- validation: % positive of each marker per assigned label ----
    all_markers = [g for genes in SUBTYPE_MARKERS.values()
                   for g in resolve_genes(genes, mye.var_names)]
    var_idx = {g: i for i, g in enumerate(mye.var_names)}
    Mpos = _dense(mye.X[:, [var_idx[g] for g in all_markers]]) > 0
    val = pd.DataFrame(Mpos, columns=all_markers)
    val["label"] = labels
    row_order = [s for s in subtypes if s in set(labels)] + \
                ([UNRESOLVED_LABEL] if UNRESOLVED_LABEL in set(labels) else [])
    table = (100 * val.groupby("label")[all_markers].mean()).round(1).reindex(row_order)
    print("\n=== % positive per marker, by assigned label (diagonal blocks should dominate) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table)

    # ---- save ----
    out = pd.DataFrame({"x": cxm, "y": cym, "subtype": labels})
    for st in subtypes:
        out["score_" + st] = S[st].to_numpy()
    out["top_score_label"] = res["top_label"]
    out["second_label"] = res["second_label"]
    out["top_scaled"] = res["top_scaled"]
    out["second_scaled"] = res["second_scaled"]
    out["margin_scaled"] = res["margin_scaled"]
    out["fdr_pass"] = res["fdr_pass"]
    out["margin_pass"] = res["margin_pass"]
    out.to_csv(f"{out_dir}/subtype_calls.csv", index=False)
    threshold_df.to_csv(f"{out_dir}/mirrored_fdr_thresholds.csv", index=False)
    table.to_csv(f"{out_dir}/label_marker_positivity.csv")

    # ---- threshold diagnostic figures (mirror Stage 1) ----
    plot_overlap_before(out_dir, name, res)
    plot_assignment_after(out_dir, name, labels, res, UNRESOLVED_LABEL)

    # ---- spatial: combined ----
    fig, ax = plt.subplots(figsize=(10, 9), dpi=170)
    ax.scatter(cx, cy, s=0.5, c="#ededed", linewidths=0, rasterized=True)
    ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.2, c=TUMOR_COLOR,
               linewidths=0, rasterized=True, label=f"tumor ({int(tumor_all.sum()):,})")
    for lbl in [UNRESOLVED_LABEL, "microglia", "BAM", "MDM"]:
        m = labels == lbl
        if not m.any():
            continue
        ax.scatter(cxm[m], cym[m], s=3, c=COLORS[lbl], linewidths=0,
                   rasterized=True, label=f"{lbl} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name} myeloid subtypes (score_genes + FDR + margin)")
    ax.legend(loc="lower right", markerscale=4, fontsize=7)
    plt.tight_layout(); plt.savefig(f"{out_dir}/spatial_subtypes.png",
                                    bbox_inches="tight"); plt.close()

    # ---- spatial: one per subtype ----
    for lbl in subtypes:
        m = labels == lbl
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        ax.scatter(cx, cy, s=0.6, c="#e0e0e0", linewidths=0, rasterized=True)
        ax.scatter(cx_all[tumor_all], cy_all[tumor_all], s=1.5, c=TUMOR_COLOR,
                   linewidths=0, rasterized=True, label="tumor")
        ax.scatter(cxm[m], cym[m], s=4, c=COLORS[lbl], linewidths=0,
                   rasterized=True, label=f"{lbl} (n={int(m.sum())})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name} {lbl} (n={int(m.sum())})")
        ax.legend(loc="lower right", markerscale=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/spatial_{safe_name(lbl)}.png", bbox_inches="tight")
        plt.close()

    print(f"\nsaved outputs -> {out_dir}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for name, path in SLICES.items():
        run_slice(name, path)


if __name__ == "__main__":
    main()
