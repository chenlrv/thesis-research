"""2-component Gaussian mixture per score (from score_genes_slice3/cell_scores.csv).

For each cell type's score: fit a background bump + a positive bump. Outputs the
positive-component WEIGHT (estimated prevalence), a per-cell posterior P(positive),
and the score where P crosses 0.5. Unlike Otsu, the mixture models the class
imbalance, so rare types shouldn't be over-called. Per-type only; no cross-type
assignment yet.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

CSV = "D:/thesis-research/score_genes_slice3/cell_scores.csv"
OUT_DIR = "D:/thesis-research/score_genes_slice3"
P_CALL, P_HIGH = 0.5, 0.9


def fit_mixture(v):
    gm = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-4,
                         random_state=0, n_init=5)
    gm.fit(v.reshape(-1, 1))
    mu = gm.means_.ravel()
    sd = np.sqrt(gm.covariances_.ravel())
    w = gm.weights_.ravel()
    pos = int(np.argmax(mu))                     # positive = higher-mean component
    P = gm.predict_proba(v.reshape(-1, 1))[:, pos]
    grid = np.linspace(v.min(), v.max(), 4000)
    Pg = gm.predict_proba(grid.reshape(-1, 1))[:, pos]
    above = np.where(Pg > P_CALL)[0]
    thr = grid[above[0]] if len(above) else np.nan
    return P, thr, (mu, sd, w, pos)


def main():
    df = pd.read_csv(CSV)
    score_cols = [c for c in df.columns if c.startswith("score_")]
    labels = [c.replace("score_", "").replace("_", " ") for c in score_cols]
    x, y = df["x"].to_numpy(), -df["y"].to_numpy()

    res = {}
    print(f"{'type':18s} {'bg_mu':>7s} {'pos_mu':>7s} {'prevalence':>11s} "
          f"{'thr@.5':>7s} {'%P>.5':>7s} {'%P>.9':>7s}")
    for c, lab in zip(score_cols, labels):
        v = df[c].to_numpy()
        P, thr, (mu, sd, w, pos) = fit_mixture(v)
        res[c] = dict(P=P, thr=thr, mu=mu, sd=sd, w=w, pos=pos)
        print(f"{lab:18s} {mu[1-pos]:7.2f} {mu[pos]:7.2f} {100*w[pos]:10.1f}% "
              f"{thr:7.2f} {100*(P>P_CALL).mean():6.1f}% {100*(P>P_HIGH).mean():6.1f}%")

    # save posteriors
    out = df[["x", "y"]].copy()
    for c, lab in zip(score_cols, labels):
        out["P_" + lab.replace(" ", "_")] = res[c]["P"]
    out.to_csv(f"{OUT_DIR}/mixture_posteriors.csv", index=False)

    # histograms with the two fitted components + P>0.5 region
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=100)
    for ax, c, lab in zip(axes.ravel(), score_cols, labels):
        v = df[c].to_numpy(); r = res[c]; P = r["P"]
        ax.hist(v, bins=70, density=True, color="0.85")
        grid = np.linspace(v.min(), v.max(), 500)
        for k, name, col in [(r["pos"], "positive", "#2ca02c"),
                             (1 - r["pos"], "background", "#d62728")]:
            ax.plot(grid, r["w"][k] * norm.pdf(grid, r["mu"][k], r["sd"][k]),
                    col, lw=2, label=name)
        if np.isfinite(r["thr"]):
            ax.axvline(r["thr"], c="k", ls="--", lw=1.2, label=f"P=.5 @ {r['thr']:.2f}")
        ax.set_yscale("log")
        ax.set_title(f"{lab}  prev={100*r['w'][r['pos']]:.1f}%  (P>.5: {100*(P>.5).mean():.1f}%)")
        ax.legend(fontsize=7)
    fig.suptitle("slice_3 mixture per score (green=positive bump, red=background)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/mixture_components.png",
                                    dpi=100, bbox_inches="tight"); plt.close()

    # spatial maps: P>0.5 (light) and P>0.9 (dark)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=95)
    for ax, c, lab in zip(axes.ravel(), score_cols, labels):
        P = res[c]["P"]
        mid = (P > P_CALL) & (P <= P_HIGH); high = P > P_HIGH
        ax.scatter(x, y, s=1.0, c="lightgrey", linewidths=0, rasterized=True)
        ax.scatter(x[mid], y[mid], s=2.2, c="#74c476", linewidths=0, rasterized=True)
        ax.scatter(x[high], y[high], s=2.6, c="#00441b", linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{lab}  P>.5:{100*(P>P_CALL).mean():.1f}%  P>.9:{100*(P>P_HIGH).mean():.1f}%")
    fig.suptitle("slice_3 mixture-positive cells (light=P>.5, dark=P>.9)", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/mixture_spatial.png",
                                    dpi=95, bbox_inches="tight"); plt.close()

    print(f"\nsaved mixture_posteriors.csv, mixture_components.png, mixture_spatial.png "
          f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
