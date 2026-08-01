"""Two supervisor-facing figures from a score_genes cell_scores.csv:

  1) overlap_before_threshold.png  -- how many cells have >1 group with close
     scores, BEFORE any threshold is applied (purely score separation), plus
     which group pairs overlap.
  2) assignment_after_threshold.png -- what the FDR + 50% margin threshold did:
     final labels, and why the leftover cells are 'unknown' (weak vs ambiguous).

"Close / ambiguous" = the cell's best group does not beat its runner-up by the
MARGIN_RATIO factor, on MAD-scaled scores (so every group is on one comparable
scale). These columns (top_scaled, second_scaled, fdr_pass, margin_pass) are
already written by the run_score_genes_* scripts.

Usage:  python plot_overlap_before_after.py [path/to/cell_scores.csv]
Default: score_genes_myeloid_stage1/slice_1/cell_scores.csv
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MARGIN_RATIO = 1.5   # the "50% rule": a clear winner beats the runner-up by >=1.5x

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "D:/thesis-research/score_genes_myeloid_stage1/slice_1/cell_scores.csv"


def main():
    df = pd.read_csv(CSV)
    out_dir = os.path.dirname(CSV)
    tag = os.path.basename(out_dir) or "data"
    n = len(df)

    top = df["top_scaled"].to_numpy()
    second = df["second_scaled"].to_numpy()
    top_label = df["top_score_label"].to_numpy()
    second_label = df["second_label"].to_numpy()

    # ---- three mutually-exclusive buckets, BEFORE any FDR/threshold -----------
    no_group = top <= 0                                   # nothing scores positive
    ambiguous = (top > 0) & (second > 0) & (top < MARGIN_RATIO * second)
    clear = (top > 0) & ~ambiguous                        # one dominant group
    buckets = [
        ("One clear group", int(clear.sum()), "#00a087"),
        (f"≥2 close groups\n(ambiguous)", int(ambiguous.sum()), "#e74c3c"),
        ("No group positive", int(no_group.sum()), "#bbbbbb"),
    ]

    print(f"{tag}: {n:,} cells")
    for name, c, _ in buckets:
        print(f"  {name.splitlines()[0]:22s}: {c:>7,} ({100*c/n:4.1f}%)")

    # =====================================================================
    # FIGURE 1 -- BEFORE threshold: how much group overlap is there?
    # =====================================================================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), dpi=150,
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # Panel A: the three buckets
    names = [b[0] for b in buckets]
    vals = [b[1] for b in buckets]
    cols = [b[2] for b in buckets]
    bars = axA.bar(names, vals, color=cols, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        axA.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n{100*v/n:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axA.set_ylabel("number of cells")
    axA.set_ylim(0, max(vals) * 1.18)
    axA.set_title(f"Per cell: is one group's score clearly highest?\n"
                  f"(close = top group < {MARGIN_RATIO}× the runner-up)",
                  fontsize=11)
    axA.margins(x=0.05)

    # Panel B: which group pairs overlap (among ambiguous cells)
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
                axB.text(j, i, f"{M[i, j]:,}", ha="center", va="center",
                         fontsize=8,
                         color="white" if M[i, j] > M.max() * 0.5 else "black")
    fig.colorbar(im, ax=axB, shrink=0.8, label="# ambiguous cells")
    axB.set_title(f"Which group pairs have close scores?\n"
                  f"({int(ambiguous.sum()):,} ambiguous cells)", fontsize=11)

    fig.suptitle(f"{tag} — score overlap between cell-type groups "
                 f"(BEFORE threshold)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f1 = os.path.join(out_dir, "overlap_before_threshold.png")
    plt.savefig(f1, bbox_inches="tight"); plt.close()

    # =====================================================================
    # FIGURE 2 -- AFTER threshold: what the FDR + margin rule decided
    # =====================================================================
    calls = df["celltype"].to_numpy()
    annotated = calls != "unknown"
    fdr = df["fdr_pass"].to_numpy().astype(bool)
    margin = df["margin_pass"].to_numpy().astype(bool)
    # every unknown cell is either weak (fails FDR) or ambiguous (passes FDR but
    # fails the margin) -- these are exclusive and cover all unknowns.
    weak = (~annotated) & (~fdr)
    ambiguous_unk = (~annotated) & fdr & (~margin)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), dpi=150,
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: final label counts (groups + unknown)
    vc = pd.Series(calls[annotated]).value_counts()
    cats = list(vc.index) + ["unknown"]
    counts = list(vc.values) + [int((~annotated).sum())]
    colors = ["#4c72b0"] * len(vc) + ["#bbbbbb"]
    bars = axA.bar(cats, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, counts):
        axA.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}",
                 ha="center", va="bottom", fontsize=9)
    axA.set_ylabel("number of cells")
    axA.set_ylim(0, max(counts) * 1.15)
    axA.set_xticklabels(cats, rotation=45, ha="right")
    axA.set_title(f"Final assignment after FDR + {int((MARGIN_RATIO-1)*100)}% "
                  f"margin\n({int(annotated.sum()):,} annotated, "
                  f"{int((~annotated).sum()):,} unknown)", fontsize=11)

    # Panel B: why the unknown cells are unknown
    reasons = [
        ("Ambiguous\n(≥2 close groups)", int(ambiguous_unk.sum()), "#e74c3c"),
        ("Weak signal\n(below FDR)", int(weak.sum()), "#f39c12"),
    ]
    rn = [r[0] for r in reasons]
    rv = [r[1] for r in reasons]
    rc = [r[2] for r in reasons]
    bars = axB.bar(rn, rv, color=rc, edgecolor="black", linewidth=0.5)
    tot_unk = max(int((~annotated).sum()), 1)
    for bar, v in zip(bars, rv):
        axB.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}\n{100*v/tot_unk:.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    axB.set_ylabel("number of unknown cells")
    axB.set_ylim(0, max(rv) * 1.18)
    axB.set_title("Why cells were left 'unknown'", fontsize=11)

    fig.suptitle(f"{tag} — assignment AFTER threshold "
                 f"(ambiguous cells → unknown)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f2 = os.path.join(out_dir, "assignment_after_threshold.png")
    plt.savefig(f2, bbox_inches="tight"); plt.close()

    print(f"\nsaved:\n  {f1}\n  {f2}")


if __name__ == "__main__":
    main()
