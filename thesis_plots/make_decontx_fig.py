"""Regenerate the ambient-correction (decontX) robustness figure.

For each slice that has a decontX result, stratify non-tumor cells into terciles of
decontX-estimated contamination and recompute the reporter correlations
F1 = GFP~tdTomato and F2 = GFP~Cx3cr1 within each tercile (union-positive mask,
matching agents/probe_validation/probe_validation_slice1.py). If the anomalous
negative correlations are flat across terciles, ambient load does not explain them.

decontX outputs currently exist for slices 1 (tumor) and 3 (control).

Run: conda run -n thesis_research python thesis_plots/make_decontx_fig.py
"""
import os
import glob
import anndata as ad
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse

DEC = "D:/thesis-research/resources/cache/decontx/slice_{}_decontx.h5ad"
OUT = "D:/thesis-research/thesis_plots/decontx_ambient_control.png"
TUMOR_COL = "pred_tumor_XGBoost"
ORANGE, RED = "#E69F00", "#D55E00"
TRUE = {1, "1", "1.0", True, "True", "true", "TRUE"}


def which_slices():
    out = []
    for s in range(1, 7):
        if os.path.exists(DEC.format(s)):
            out.append(s)
    return out


def col(a, gene):
    panel = {g.lower(): g for g in a.var_names}
    xv = a[:, panel[gene.lower()]].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(float)


def pearson_union(x, y):
    m = (x > 0) | (y > 0)
    if m.sum() < 3:
        return np.nan, 0
    xx, yy = x[m], y[m]
    xm, ym = xx - xx.mean(), yy - yy.mean()
    sx, sy = np.sqrt((xm * xm).sum()), np.sqrt((ym * ym).sum())
    if sx == 0 or sy == 0:
        return np.nan, int(m.sum())
    return float((xm * ym).sum() / (sx * sy)), int(m.sum())


def analyze(s):
    a = ad.read_h5ad(DEC.format(s))
    tv = a.obs[TUMOR_COL]
    tumor = tv.astype(object).isin(TRUE).to_numpy()
    nt = ~tumor
    cont = a.obs["decontx_contamination"].to_numpy(float)[nt]
    gfp, tdt, cx3 = col(a, "GFP")[nt], col(a, "tdTomato")[nt], col(a, "Cx3cr1")[nt]
    q1, q2 = np.quantile(cont, [1 / 3, 2 / 3])
    strata = {"Low": cont <= q1, "Mid": (cont > q1) & (cont <= q2), "High": cont > q2}
    res = {"median_contam": float(np.median(cont)), "n_nt": int(nt.sum()), "strata": {}}
    for name, mask in strata.items():
        f1, n1 = pearson_union(gfp[mask], tdt[mask])
        f2, n2 = pearson_union(gfp[mask], cx3[mask])
        res["strata"][name] = dict(mean_contam=float(cont[mask].mean()),
                                   F1=f1, F2=f2, n=int(mask.sum()))
    return res


def main():
    slices = which_slices()
    print("decontX available for slices:", slices)
    data = {s: analyze(s) for s in slices}
    for s, r in data.items():
        print(f"\nSlice {s}: non-tumor n={r['n_nt']:,}  median contam={r['median_contam']:.3f}")
        for k, v in r["strata"].items():
            print(f"  {k:4s} (contam {v['mean_contam']:.3f}, n={v['n']:,}): "
                  f"GFP~tdTomato r={v['F1']:+.3f}  GFP~Cx3cr1 r={v['F2']:+.3f}")

    labels = ["Low", "Mid", "High"]
    n = len(slices)
    ncols = 3 if n > 3 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), dpi=150,
                             squeeze=False)
    axes = axes.ravel()
    for extra in axes[n:]:
        extra.axis("off")
    kind = {1: "tumor", 2: "tumor", 3: "control", 4: "control", 5: "tumor", 6: "tumor"}
    # shared y-limit so panels are directly comparable
    allv = [data[s]["strata"][k][f] for s in slices for k in labels for f in ("F1", "F2")]
    ylo = min(allv) - 0.12
    for ax, s in zip(axes, slices):
        r = data[s]
        xs = np.arange(3)
        w = 0.38
        f1 = [r["strata"][k]["F1"] for k in labels]
        f2 = [r["strata"][k]["F2"] for k in labels]
        b1 = ax.bar(xs - w / 2, f1, w, color=ORANGE, label="GFP – tdTomato")
        b2 = ax.bar(xs + w / 2, f2, w, color=RED, label="GFP – Cx3cr1")
        ax.axhline(0, color="#444444", lw=0.9)
        tick = [f"{k}\n(~{r['strata'][k]['mean_contam']*100:.0f}%)" for k in labels]
        ax.set_xticks(xs)
        ax.set_xticklabels(tick, fontsize=9)
        ax.set_xlabel("decontX-estimated contamination tercile")
        ax.set_ylabel("Pearson r")
        ax.set_title(f"Slice {s} ({kind.get(s,'')})  —  median contam {r['median_contam']*100:.0f}%",
                     fontsize=10)
        ax.set_ylim(ylo, 0.14)
        for bars, vals in [(b1, f1), (b2, f2)]:
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v - 0.012),
                            ha="center", va="top", fontsize=8, color="#222222")
        ax.legend(fontsize=8.5, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Ambient correction (decontX) does not rescue the reporter anomalies",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
