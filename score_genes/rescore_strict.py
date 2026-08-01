"""rescore_strict.py - re-derive the slice-1 celltype calls under stricter gates.

The per-cell score_genes scores do NOT depend on the FDR / margin gates - only the
mirrored-FDR thresholds and the final assignment do. So we recompute calls
directly from the existing cell_scores.csv (no h5ad reload, identical scores):

    baseline : FDR <= 0.05, margin ratio 1.5  (already in cell_scores.csv)
    strict   : FDR <= 0.01, margin ratio 2.0

Writes a new cell_scores.csv (same columns, updated gate/celltype) to a separate
dir so the baseline stays intact for comparison.
"""
import os

import numpy as np
import pandas as pd

BASE_CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
OUT_DIR = "D:/thesis-research/score_genes_slice1_strict"
FDR_CUTOFF = 0.01
MARGIN_RATIO = 2.0


def mirrored_fdr_threshold(scores, fdr=0.05):
    """Smallest score t with #(<=-t)/#(>=t) <= fdr (matches run_score_genes)."""
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.sort(scores[scores > 0]))
    for t in candidates:
        n_called = int(np.sum(scores >= t))
        n_mirror = int(np.sum(scores <= -t))
        if n_mirror / max(n_called, 1) <= fdr:
            return float(t)
    return np.inf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(BASE_CSV)

    score_cols = [c for c in df.columns if c.startswith("score_")]
    labels = [c[len("score_"):] for c in score_cols]
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    S = df[score_cols].to_numpy(float)

    top_label = df["top_score_label"].to_numpy()
    unknown_top = set(top_label) - set(labels)
    if unknown_top:
        raise SystemExit(f"top_score_label values not in score columns: {unknown_top}")

    # stricter mirrored-FDR thresholds, per label
    thresholds = {lab: mirrored_fdr_threshold(df[col].to_numpy(), FDR_CUTOFF)
                  for lab, col in zip(labels, score_cols)}

    # gates: margin from stored MAD-scaled top/second; FDR on the winner's raw score
    top_scaled = df["top_scaled"].to_numpy()
    second_scaled = df["second_scaled"].to_numpy()
    margin_pass = (top_scaled > 0) & (
        (second_scaled <= 0) | (top_scaled >= MARGIN_RATIO * second_scaled))

    idx = np.array([lab_to_idx[l] for l in top_label])
    top_raw = S[np.arange(len(S)), idx]
    thr_arr = np.array([thresholds[l] for l in top_label])
    fdr_pass = top_raw >= thr_arr

    calls = np.where(margin_pass & fdr_pass, top_label, "unknown").astype(object)

    out = df.copy()
    out["fdr_pass"], out["margin_pass"], out["celltype"] = fdr_pass, margin_pass, calls
    out.to_csv(f"{OUT_DIR}/cell_scores.csv", index=False)

    # ---- report: strict vs baseline ----
    base = df["celltype"]
    n = len(df)
    print(f"strict gates: FDR <= {FDR_CUTOFF}, margin ratio {MARGIN_RATIO}  (n={n:,})")
    print("\nmirrored-FDR thresholds (strict):")
    for lab in labels:
        print(f"  {lab:12s} {thresholds[lab]:.4f}")

    print(f"\n  FDR gate pass:    {int(fdr_pass.sum()):>7,} ({100*fdr_pass.mean():5.1f}%)")
    print(f"  margin gate pass: {int(margin_pass.sum()):>7,} ({100*margin_pass.mean():5.1f}%)")
    print(f"  both (annotated): {int((calls!='unknown').sum()):>7,} "
          f"({100*(calls!='unknown').mean():5.1f}%)")

    cmp = pd.DataFrame({
        "baseline": base.value_counts(),
        "strict": pd.Series(calls).value_counts(),
    }).fillna(0).astype(int)
    cmp["delta"] = cmp["strict"] - cmp["baseline"]
    print("\nper-label counts (baseline 0.05/1.5  vs  strict 0.01/2.0):")
    print(cmp.sort_values("baseline", ascending=False).to_string())

    # how many baseline-called cells were demoted to unknown vs kept/changed
    was_called = base.to_numpy() != "unknown"
    kept_same = was_called & (calls == base.to_numpy())
    demoted = was_called & (calls == "unknown")
    flipped = was_called & (calls != "unknown") & (calls != base.to_numpy())
    print(f"\nof {int(was_called.sum()):,} baseline-called cells: "
          f"{int(kept_same.sum()):,} kept same label, "
          f"{int(demoted.sum()):,} demoted to unknown, "
          f"{int(flipped.sum()):,} flipped label")
    print(f"\nsaved -> {OUT_DIR}/cell_scores.csv")


if __name__ == "__main__":
    main()
