"""
First-pass probe-reliability diagnostics.
Run via: conda run -n thesis_research python agents/check_negative_findings.py

Checks the three negative findings on non-tumor cells, slices 1 & 3:
  1. GFP vs tdTomato negatively correlated
  2. GFP vs Cx3cr1 negatively correlated
  3. Lyve1 positivity higher/broader than expected for a BAM marker

Plus an ambient/segmentation-spillover diagnostic:
  - correlation as a function of count threshold (ambient dominates low-count cells)
  - per-cell decontx_contamination in tag+ vs tag- cells
  - Lyve1 co-expression with bona-fide BAM markers (Mrc1/Cd163)
"""
import anndata as ad
import numpy as np
import scipy.sparse as sp

DECONTX = r"d:/thesis-research/resources/cache/decontx/slice_{}_decontx.h5ad"


def col(a, gene):
    if gene not in a.var_names:
        return None
    x = a[:, gene].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()


def pearson(x, y):
    # Manual Pearson r using reductions only (avoids BLAS/MKL matmul, which can
    # crash on a mis-resolved numpy build via np.corrcoef/np.dot).
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


def tumor_mask(a):
    tv = a.obs["pred_tumor_XGBoost"]
    if tv.dtype == bool:
        return tv.to_numpy()
    if np.issubdtype(tv.dtype, np.number):
        return (tv.to_numpy() > 0.5)
    return tv.astype(str).str.lower().isin(["1", "true", "tumor"]).to_numpy()


for slc in (1, 3):
    print("=" * 70)
    print(f"SLICE {slc}")
    print("=" * 70)
    a = ad.read_h5ad(DECONTX.format(slc))
    nt = ~tumor_mask(a)
    sub = a[nt].copy()
    print(f"non-tumor cells: {sub.n_obs:,} / {a.n_obs:,}")

    gfp = col(sub, "GFP")
    tdt = col(sub, "tdTomato")
    cx3 = col(sub, "Cx3cr1")
    lyve = col(sub, "Lyve1")
    mrc1 = col(sub, "Mrc1")
    cd163 = col(sub, "Cd163")
    cont = sub.obs["decontx_contamination"].to_numpy(dtype=float) \
        if "decontx_contamination" in sub.obs else None
    total = np.asarray(sub.X.sum(axis=1)).ravel()

    # ---- Finding 1: GFP vs tdTomato ----
    m = (gfp > 0) | (tdt > 0)
    print(f"\n[F1] GFP vs tdTomato  (cells with >=1 of either: n={int(m.sum()):,})")
    print(f"     full Pearson r = {pearson(gfp[m], tdt[m]):+.3f}")
    print("     r by min-count threshold (ambient dominates low counts):")
    for thr in (1, 2, 3, 5):
        mm = (gfp >= thr) | (tdt >= thr)
        print(f"       >= {thr}: n={int(mm.sum()):6d}  r={pearson(gfp[mm], tdt[mm]):+.3f}")
    print(f"     co-positive GFP+&tdT+: {int(((gfp>0)&(tdt>0)).sum()):,}  "
          f"GFP-only: {int(((gfp>0)&(tdt==0)).sum()):,}  "
          f"tdT-only: {int(((gfp==0)&(tdt>0)).sum()):,}")

    # ---- Finding 2: GFP vs Cx3cr1 ----
    if cx3 is not None:
        m2 = (gfp > 0) | (cx3 > 0)
        print(f"\n[F2] GFP vs Cx3cr1  (n={int(m2.sum()):,})")
        print(f"     full Pearson r = {pearson(gfp[m2], cx3[m2]):+.3f}")
        for thr in (1, 2, 3, 5):
            mm = (gfp >= thr) | (cx3 >= thr)
            print(f"       >= {thr}: n={int(mm.sum()):6d}  r={pearson(gfp[mm], cx3[mm]):+.3f}")

    # ---- Finding 3: Lyve1 breadth ----
    if lyve is not None:
        pos = lyve > 0
        print(f"\n[F3] Lyve1 positivity: {pos.mean()*100:.1f}% of non-tumor cells, "
              f"max/cell={int(lyve.max())}, mean-in-pos={lyve[pos].mean():.2f}")
        if mrc1 is not None and cd163 is not None:
            bam = (mrc1 > 0) | (cd163 > 0)
            print(f"     Lyve1+ rate in BAM-like (Mrc1+|Cd163+, n={int(bam.sum()):,}): "
                  f"{pos[bam].mean()*100:.1f}%")
            print(f"     Lyve1+ rate in NON-BAM         (n={int((~bam).sum()):,}): "
                  f"{pos[~bam].mean()*100:.1f}%")
            print(f"     -> {(pos & ~bam).sum()/max(1,pos.sum())*100:.0f}% of all Lyve1+ "
                  f"cells lack both Mrc1 and Cd163 (breadth/ambient signature)")

    # ---- Ambient diagnostic via decontx_contamination & total counts ----
    if cont is not None:
        print(f"\n[Ambient] mean decontx_contamination: GFP+ {cont[gfp>0].mean():.3f} "
              f"vs GFP- {cont[gfp==0].mean():.3f}")
        if lyve is not None:
            print(f"          mean decontx_contamination: Lyve1+ {cont[lyve>0].mean():.3f} "
                  f"vs Lyve1- {cont[lyve==0].mean():.3f}")
    print(f"[Ambient] median total counts: GFP+ {np.median(total[gfp>0]):.0f} "
          f"vs GFP- {np.median(total[gfp==0]):.0f}")
    print()
