"""
Shared metric logic for segmentation QC. Reused identically by the baseline
(current segmentation) and the re-segmented (Baysor / Cellpose) cell x gene
matrices so the comparison is apples-to-apples.

Mirrors agents/check_negative_findings.py:
  - GFP<->tdTomato and GFP<->Cx3cr1 Pearson r with a min-count threshold sweep
  - Lyve1 breadth (% positive among cells; % of Lyve1+ lacking both Mrc1 & Cd163)
  - microglia / MDM / BAM marker-purity metric (co-expression "leakage")
All Pearson r computed with reductions only (no BLAS matmul -> MKL-safe).
"""
import numpy as np

# ---- marker sets (panel-validated; Ly6c2 absent from panel) ----
MICROGLIA = ["Cx3cr1", "TMEM119", "C1qa", "C1qb", "Hexb"]
MDM = ["Ccr2", "Plac8", "Vim", "S100a6"]   # MDM-leaning; Apoe dropped per spec
BAM = ["Mrc1", "Cd163", "Lyve1"]


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    sx = float(np.sqrt((xm * xm).sum()))
    sy = float(np.sqrt((ym * ym).sum()))
    if sx == 0 or sy == 0:
        return float("nan")
    return float((xm * ym).sum() / (sx * sy))


def _g(mat, genes, gene):
    """Return the column vector for `gene` from a (cells x genes) ndarray given
    the ordered `genes` list, or None if absent."""
    if gene not in genes:
        return None
    return np.asarray(mat[:, genes.index(gene)], dtype=np.float64).ravel()


def threshold_sweep(a, b, thrs=(1, 2, 3, 5)):
    out = {}
    for thr in thrs:
        mm = (a >= thr) | (b >= thr)
        out[thr] = (int(mm.sum()), pearson(a[mm], b[mm]))
    return out


def marker_purity(mat, genes):
    """Cross-population co-expression 'leakage'. For each pair of lineages
    (micro/MDM/BAM), among cells positive (>=1 count) for a lineage's core
    markers, what fraction are ALSO positive for the *other* lineage's markers?
    Lower = cleaner separation. We summarise with a single mean cross-leakage
    rate (lower is better) and the per-lineage on/off-target score.

    Definition per cell: lineage score = number of that lineage's markers with
    >=1 count. A cell is 'assigned' to the lineage with the max score (ties ->
    none). Purity = mean over assigned cells of (own score) / (own + other).
    """
    sets = {"microglia": MICROGLIA, "MDM": MDM, "BAM": BAM}
    # build score per lineage
    scores = {}
    for name, gl in sets.items():
        cols = [_g(mat, genes, g) for g in gl]
        cols = [c for c in cols if c is not None]
        if not cols:
            scores[name] = np.zeros(mat.shape[0])
        else:
            scores[name] = np.sum([(c >= 1).astype(float) for c in cols], axis=0)
    S = np.vstack([scores["microglia"], scores["MDM"], scores["BAM"]]).T  # cells x 3
    names = ["microglia", "MDM", "BAM"]
    total = S.sum(axis=1)
    assigned = total > 0
    amax = S.argmax(axis=1)
    own = S[np.arange(S.shape[0]), amax]
    others = total - own
    # purity: own / total, only for cells with a positive assignment & no tie
    is_tie = (S == S.max(axis=1, keepdims=True)).sum(axis=1) > 1
    valid = assigned & ~is_tie
    purity = float(np.mean(own[valid] / total[valid])) if valid.sum() else float("nan")
    # cross-leakage: among assigned cells, mean fraction of off-target signal
    leak = float(np.mean(others[valid] / total[valid])) if valid.sum() else float("nan")
    # per-lineage: among cells assigned to lineage, mean off-target marker count
    per = {}
    for i, nm in enumerate(names):
        sel = valid & (amax == i)
        per[nm] = {
            "n_assigned": int(sel.sum()),
            "mean_own_markers": float(own[sel].mean()) if sel.sum() else float("nan"),
            "mean_offtarget_markers": float(others[sel].mean()) if sel.sum() else float("nan"),
        }
    return {"purity_own_frac": purity, "cross_leakage_frac": leak,
            "n_valid": int(valid.sum()), "per_lineage": per}


def compute_all(mat, genes, label=""):
    """mat: (n_cells, n_genes) ndarray of raw counts. genes: ordered gene list."""
    gfp = _g(mat, genes, "GFP")
    tdt = _g(mat, genes, "tdTomato")
    cx3 = _g(mat, genes, "Cx3cr1")
    lyve = _g(mat, genes, "Lyve1")
    mrc1 = _g(mat, genes, "Mrc1")
    cd163 = _g(mat, genes, "Cd163")
    total = mat.sum(axis=1)

    res = {"label": label, "n_cells": int(mat.shape[0]),
           "median_total_counts": float(np.median(total))}

    # F1 GFP vs tdTomato
    if gfp is not None and tdt is not None:
        m = (gfp > 0) | (tdt > 0)
        res["F1_full_r"] = pearson(gfp[m], tdt[m])
        res["F1_sweep"] = threshold_sweep(gfp, tdt)
        res["F1_co_pos"] = int(((gfp > 0) & (tdt > 0)).sum())
        res["F1_gfp_only"] = int(((gfp > 0) & (tdt == 0)).sum())
        res["F1_tdt_only"] = int(((gfp == 0) & (tdt > 0)).sum())
        res["F1_gfp_pos"] = int((gfp > 0).sum())
        res["F1_tdt_pos"] = int((tdt > 0).sum())

    # F2 GFP vs Cx3cr1
    if gfp is not None and cx3 is not None:
        m2 = (gfp > 0) | (cx3 > 0)
        res["F2_full_r"] = pearson(gfp[m2], cx3[m2])
        res["F2_sweep"] = threshold_sweep(gfp, cx3)

    # F3 Lyve1 breadth
    if lyve is not None:
        pos = lyve > 0
        res["F3_lyve_pct_pos"] = float(pos.mean() * 100)
        res["F3_lyve_max"] = int(lyve.max())
        res["F3_lyve_mean_in_pos"] = float(lyve[pos].mean()) if pos.sum() else 0.0
        if mrc1 is not None and cd163 is not None:
            bam = (mrc1 > 0) | (cd163 > 0)
            res["F3_lyve_pct_in_bam"] = float(pos[bam].mean() * 100) if bam.sum() else float("nan")
            res["F3_lyve_pct_in_nonbam"] = float(pos[~bam].mean() * 100) if (~bam).sum() else float("nan")
            res["F3_lyve_pct_lacking_bam"] = float((pos & ~bam).sum() / max(1, pos.sum()) * 100)

    # marker purity
    res["purity"] = marker_purity(mat, genes)
    return res


def fmt_report(res):
    L = [f"=== {res['label']} (n={res['n_cells']:,}, median total counts="
         f"{res['median_total_counts']:.0f}) ==="]
    if "F1_full_r" in res:
        L.append(f"[F1] GFP vs tdTomato full r = {res['F1_full_r']:+.3f}")
        L.append("     sweep: " + "  ".join(
            f">={t}:n={n},r={r:+.3f}" for t, (n, r) in res["F1_sweep"].items()))
        L.append(f"     co+:{res['F1_co_pos']} gfp_only:{res['F1_gfp_only']} "
                 f"tdt_only:{res['F1_tdt_only']} (gfp+:{res['F1_gfp_pos']} tdt+:{res['F1_tdt_pos']})")
    if "F2_full_r" in res:
        L.append(f"[F2] GFP vs Cx3cr1 full r = {res['F2_full_r']:+.3f}")
        L.append("     sweep: " + "  ".join(
            f">={t}:n={n},r={r:+.3f}" for t, (n, r) in res["F2_sweep"].items()))
    if "F3_lyve_pct_pos" in res:
        L.append(f"[F3] Lyve1+ = {res['F3_lyve_pct_pos']:.1f}% (max/cell={res['F3_lyve_max']}, "
                 f"mean-in-pos={res['F3_lyve_mean_in_pos']:.2f})")
        if "F3_lyve_pct_in_bam" in res:
            L.append(f"     Lyve1+ in BAM-like {res['F3_lyve_pct_in_bam']:.1f}% vs non-BAM "
                     f"{res['F3_lyve_pct_in_nonbam']:.1f}%; "
                     f"{res['F3_lyve_pct_lacking_bam']:.0f}% of Lyve1+ lack both Mrc1&Cd163")
    p = res["purity"]
    L.append(f"[Purity] own-signal frac = {p['purity_own_frac']:.3f}  "
             f"cross-leakage = {p['cross_leakage_frac']:.3f}  (n_valid={p['n_valid']:,})")
    for nm, d in p["per_lineage"].items():
        L.append(f"     {nm:9s} n={d['n_assigned']:5d}  own={d['mean_own_markers']:.2f} "
                 f"off-target={d['mean_offtarget_markers']:.2f}")
    return "\n".join(L)
