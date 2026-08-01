"""Confidence / ambiguity analysis for score_genes cell-type assignment.

Reads score_genes_slice1/cell_scores.csv (per-cell per-label scores) and:

  1. defines "closeness" relative to each cell type's own score spread
     (per-label robust z = (score - median) / MAD), so a margin means the
     same thing across labels;
  2. visualizes how many cells are near-ties for >1 group as a function of
     the closeness tolerance, and which label pairs collide;
  3. proposes a *calibrated* margin-of-confidence rule based on per-label
     2-component Gaussian mixtures (probability of the signature being "on"),
     and compares the resulting calls with the mirrored-FDR calls.

No sklearn dependency: the 1-D GMM is fit by EM in numpy.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = "D:/thesis-research/score_genes_slice1/cell_scores.csv"
OUT = "D:/thesis-research/score_genes_slice1"

# margin-of-confidence rule (probability units, see technique below)
P_MIN = 0.50   # top signature must be "on" with at least this probability
P_MARGIN = 0.50  # top prob must beat 2nd prob by at least this much


# ------------------------------------------------------------------ helpers
def _gauss(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def gmm1d(x, iters=300, tol=1e-7):
    """Fit a 2-component 1-D Gaussian mixture by EM.

    Returns the posterior probability that each point belongs to the
    HIGHER-mean component (interpreted as "signature is on"), plus the
    fitted (weights, means, vars) and the index of the high component.
    """
    x = np.asarray(x, float)
    m = np.median(x)
    lo, hi = x[x <= m], x[x > m]
    mu = np.array([lo.mean() if lo.size else m - 1.0,
                   hi.mean() if hi.size else m + 1.0])
    var = np.array([max(lo.var(), 1e-4) if lo.size else 1.0,
                    max(hi.var(), 1e-4) if hi.size else 1.0])
    w = np.array([0.5, 0.5])
    ll_old = -np.inf
    for _ in range(iters):
        p0 = w[0] * _gauss(x, mu[0], var[0])
        p1 = w[1] * _gauss(x, mu[1], var[1])
        s = p0 + p1 + 1e-300
        r0, r1 = p0 / s, p1 / s
        n0, n1 = r0.sum() + 1e-9, r1.sum() + 1e-9
        w = np.array([n0, n1]) / len(x)
        mu = np.array([(r0 * x).sum() / n0, (r1 * x).sum() / n1])
        var = np.array([(r0 * (x - mu[0]) ** 2).sum() / n0,
                        (r1 * (x - mu[1]) ** 2).sum() / n1])
        var = np.maximum(var, 1e-4)
        ll = np.log(s).sum()
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    h = int(np.argmax(mu))
    ph = w[h] * _gauss(x, mu[h], var[h])
    pl = w[1 - h] * _gauss(x, mu[1 - h], var[1 - h])
    return ph / (ph + pl + 1e-300), w, mu, var, h


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV)
    score_cols = [c for c in df.columns if c.startswith("score_")]
    labels = [c[len("score_"):] for c in score_cols]
    S = df[score_cols].to_numpy(float)        # cells x labels
    n, K = S.shape
    called_fdr = (df["celltype"].to_numpy() != "unknown")
    print(f"{n:,} cells x {K} labels; {called_fdr.sum():,} FDR-called")

    # ---- (1) closeness defined per cell type: robust z (median / MAD) -----
    med = np.median(S, axis=0)
    mad = 1.4826 * np.median(np.abs(S - med), axis=0)
    fallback = S.std(axis=0)
    mad = np.where(mad > 0, mad, fallback)
    mad = np.where(mad > 0, mad, 1.0)
    Zr = (S - med) / mad                      # robust z per label
    order = np.argsort(-Zr, axis=1)
    z1 = np.take_along_axis(Zr, order[:, :1], 1)[:, 0]
    z2 = np.take_along_axis(Zr, order[:, 1:2], 1)[:, 0]
    margin_z = z1 - z2
    top_idx, sec_idx = order[:, 0], order[:, 1]
    lab = np.asarray(labels)

    # ---- (2a) ambiguity sweep: #cells with >=2 labels within delta of top -
    deltas = np.linspace(0, 3, 121)
    frac_all, frac_called = [], []
    for d in deltas:
        within = (Zr >= (z1[:, None] - d)).sum(axis=1)
        amb = within >= 2
        frac_all.append(100 * amb.mean())
        frac_called.append(100 * amb[called_fdr].mean())

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax.plot(deltas, frac_all, lw=2, label="all non-tumor cells")
    ax.plot(deltas, frac_called, lw=2, label="FDR-called cells")
    for dref in (0.5, 1.0):
        ax.axvline(dref, color="grey", ls=":", lw=1)
    ax.set_xlabel("closeness tolerance  delta  (in per-label robust-z / MAD units)")
    ax.set_ylabel("% cells with >=2 labels within delta of their top")
    ax.set_title("Ambiguity sweep: how many cells are near-ties for >1 group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/confidence_ambiguity_sweep.png", bbox_inches="tight")
    plt.close()

    # ---- (2b) confusability heatmap among near-tie cells (delta = 0.5) ----
    d0 = 0.5
    near = margin_z < d0
    M = np.zeros((K, K), int)
    for i, j in zip(top_idx[near], sec_idx[near]):
        M[i, j] += 1
    Msym = M + M.T                           # collisions are unordered
    np.fill_diagonal(Msym, 0)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    im = ax.imshow(Msym, cmap="magma")
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(K)); ax.set_yticklabels(labels)
    for i in range(K):
        for j in range(K):
            if Msym[i, j]:
                ax.text(j, i, f"{Msym[i, j]:,}", ha="center", va="center",
                        fontsize=8, color="white")
    fig.colorbar(im, ax=ax, shrink=0.7, label="# near-tie cells")
    ax.set_title(f"Which groups collide  (robust-z margin < {d0})\n"
                 f"{int(near.sum()):,} near-tie cells total")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confidence_confusability.png", bbox_inches="tight")
    plt.close()

    # ---- (3) calibrated margin via per-label 2-component GMM -> P(on) ------
    P = np.zeros_like(S)
    gmm_params = []
    for k in range(K):
        ph, w, mu, var, h = gmm1d(S[:, k])
        P[:, k] = ph
        gmm_params.append((w, mu, var, h))
    porder = np.argsort(-P, axis=1)
    p1 = np.take_along_axis(P, porder[:, :1], 1)[:, 0]
    p2 = np.take_along_axis(P, porder[:, 1:2], 1)[:, 0]
    p_top_idx = porder[:, 0]
    pmargin = p1 - p2

    confident = (p1 >= P_MIN) & (pmargin >= P_MARGIN)
    call_gmm = np.where(confident, lab[p_top_idx], "unknown")

    # reason for unknown
    reason = np.full(n, "ok", object)
    reason[(p1 < P_MIN)] = "nothing_on"
    reason[(p1 >= P_MIN) & (pmargin < P_MARGIN)] = "ambiguous"

    # ---- (3 fig) per-label GMM fit + implied P(on)=0.5 threshold ----------
    ncols = 3
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), dpi=140)
    axes = np.asarray(axes).ravel()
    grid = None
    for ax, k in zip(axes, range(K)):
        x = S[:, k]
        w, mu, var, h = gmm_params[k]
        ax.hist(x, bins=80, density=True, color="#cccccc")
        grid = np.linspace(x.min(), x.max(), 400)
        for c in range(2):
            ax.plot(grid, w[c] * _gauss(grid, mu[c], var[c]),
                    lw=1.6, color="#00a087" if c == h else "#888888")
        # score at which P(on) crosses 0.5
        ph = w[h] * _gauss(grid, mu[h], var[h])
        pl = w[1 - h] * _gauss(grid, mu[1 - h], var[1 - h])
        cross = grid[np.argmin(np.abs(ph - pl))]
        ax.axvline(cross, color="crimson", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"{labels[k]}  P(on) frac={100 * (P[:, k] >= .5).mean():.0f}%",
                     fontsize=10)
        ax.set_xlabel("score")
    for ax in axes[K:]:
        ax.axis("off")
    fig.suptitle("Per-label 2-component GMM: background vs 'signature on' "
                 "(green); red = P(on)=0.5", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT}/confidence_gmm_fits.png", bbox_inches="tight")
    plt.close()

    # ---- (3 fig) FDR calls vs GMM-confident calls -------------------------
    cats = labels + ["unknown"]
    fdr_counts = pd.Series(df["celltype"]).value_counts()
    gmm_counts = pd.Series(call_gmm).value_counts()
    fdr_v = [int(fdr_counts.get(c, 0)) for c in cats]
    gmm_v = [int(gmm_counts.get(c, 0)) for c in cats]
    xpos = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    ax.bar(xpos - 0.2, fdr_v, width=0.4, label="mirrored-FDR call", color="#4c72b0")
    ax.bar(xpos + 0.2, gmm_v, width=0.4, label="GMM confidence call", color="#dd8452")
    ax.set_xticks(xpos); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylabel("# cells")
    ax.set_title(f"Assignment comparison "
                 f"(GMM rule: P_top>={P_MIN}, margin>={P_MARGIN})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/confidence_fdr_vs_gmm.png", bbox_inches="tight")
    plt.close()

    # ---- write augmented table + console summary --------------------------
    df_out = df.copy()
    df_out["robustz_margin"] = margin_z
    df_out["gmm_p_top"] = p1
    df_out["gmm_p_second"] = p2
    df_out["gmm_margin"] = pmargin
    df_out["call_gmm"] = call_gmm
    df_out["gmm_reason"] = reason
    df_out.to_csv(f"{OUT}/cell_scores_confidence.csv", index=False)

    print("\n=== ambiguity (robust-z margin) ===")
    for d in (0.25, 0.5, 1.0):
        a = int((margin_z < d).sum())
        print(f"  margin < {d:>4.2f} sd-equiv: {a:>7,} ({100 * a / n:5.1f}%)")

    print("\n=== top colliding label pairs (robust-z margin < 0.5) ===")
    pr = pd.Series([tuple(sorted((lab[i], lab[j])))
                    for i, j in zip(top_idx[near], sec_idx[near])]
                   ).value_counts().head(8)
    print(pr.to_string())

    print(f"\n=== GMM confidence rule (P_top>={P_MIN}, margin>={P_MARGIN}) ===")
    print(pd.Series(call_gmm).value_counts().to_string())
    print("\nreason for unknown:")
    print(pd.Series(reason).value_counts().to_string())

    print("\n=== FDR call  vs  GMM call (cross-tab) ===")
    ct = pd.crosstab(df["celltype"], pd.Series(call_gmm, name="gmm"))
    print(ct.to_string())

    print(f"\nsaved -> {OUT}:\n  confidence_ambiguity_sweep.png\n"
          f"  confidence_confusability.png\n  confidence_gmm_fits.png\n"
          f"  confidence_fdr_vs_gmm.png\n  cell_scores_confidence.csv")


if __name__ == "__main__":
    main()
