"""Per-label precision metrics from a slice's cell_scores.csv (v2 labels).

(a) own-marker signal    : median # own markers detected; % above the gate minimum.
(b) contamination, two definitions (report both, they tell different stories):
    - contam_marker : % of labeled-T cells detecting >= k of a competitor's markers.
                      AMBIENT-SENSITIVE (broadly-expressed genes like Meg3 inflate it).
    - contam_fdr    : % of labeled-T cells whose competitor SCORE clears that
                      competitor's mirrored-FDR threshold. AMBIENT-ROBUST (score_genes
                      subtracts background; the threshold is calibrated to the null).
                      This is the trustworthy precision number.

pandas-only. Usage: python measure_precision.py <slice_id>
"""
import sys

import numpy as np
import pandas as pd

K = {"Astrocytes": 2, "Neurons": 2, "Myeloid": 2, "Ependymal": 2, "Vascular": 4}
TYPES = list(K)


def main(sl):
    d = f"D:/thesis-research/score_genes_slice{sl}_v2"
    df = pd.read_csv(f"{d}/cell_scores.csv")
    thr = pd.read_csv(f"{d}/mirrored_fdr_thresholds.csv").set_index("label")["threshold"]
    lab = df["celltype_v2"].to_numpy()
    rows = []
    for T in TYPES:
        m = lab == T
        n = int(m.sum())
        if n == 0:
            continue
        own = df.loc[m, f"cov_{T}"].to_numpy()
        cm = np.zeros(n, bool)
        cf = np.zeros(n, bool)
        by_fdr = {}
        for U in TYPES:
            if U == T:
                continue
            cm |= df.loc[m, f"cov_{U}"].to_numpy() >= K[U]
            posf = df.loc[m, f"score_{U}"].to_numpy() >= thr[U]
            cf |= posf
            by_fdr[U] = int(posf.sum())
        worst = max(by_fdr, key=by_fdr.get)
        rows.append({
            "label": T, "n": n,
            "own_median": float(np.median(own)),
            "own_pct_gt_min": round(100 * (own > K[T]).mean(), 1),
            "contam_marker%": round(100 * cm.sum() / n, 1),
            "contam_fdr%": round(100 * cf.sum() / n, 1),
            "top_fdr_contam": f"{worst}({by_fdr[worst]})",
        })
    print(f"=== slice {sl}: precision metrics (v2 labels) ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "3")
