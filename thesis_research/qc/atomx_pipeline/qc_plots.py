import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

_COLOR = '#F8766D'


def _generate_violin_plots(adata, keys):
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[_COLOR])
    for key in keys:
        sc.pl.violin(
            adata,
            keys=key,
            groupby="Identity",
            stripplot=False,
            color=_COLOR,
        )


def _generate_scatter_plots(adata):
    x1 = adata.obs["nCount_RNA"].to_numpy(dtype=float)
    y1 = adata.obs["percent.NegPrb"].to_numpy(dtype=float)
    m1 = np.isfinite(x1) & np.isfinite(y1)
    r1 = _corrcoef(x1, y1)

    plt.figure(figsize=(3.0, 2.2), dpi=300)
    plt.scatter(x1[m1], y1[m1], s=2, c=_COLOR, alpha=1, linewidths=0)
    plt.xlabel("nCount_RNA")
    plt.ylabel("percent.NegPrb")
    plt.title(f"{r1:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.show()

    x2 = adata.obs["nCount_RNA"].to_numpy(dtype=float)
    y2 = adata.obs["nFeature_RNA"].to_numpy(dtype=float)
    m2 = np.isfinite(x2) & np.isfinite(y2)
    r2 = _corrcoef(x2, y2)

    plt.figure(figsize=(3.0, 2.2), dpi=300)
    plt.scatter(x2[m2], y2[m2], s=2, c=_COLOR, alpha=1, linewidths=0)
    plt.xlabel("nCount_RNA")
    plt.ylabel("nFeature_RNA")
    plt.title(f"{r2:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.show()

    x3 = adata.obs["Area"].to_numpy(dtype=float)
    y3 = adata.obs["nCount_RNA"].to_numpy(dtype=float)
    m3 = np.isfinite(x3) & np.isfinite(y3)
    r3 = _corrcoef(x3, y3)

    plt.figure(figsize=(3.0, 2.2), dpi=300)
    plt.scatter(x3[m3], y3[m3], s=2, c=_COLOR, alpha=1, linewidths=0)
    plt.xlabel("Area")
    plt.ylabel("nCount_RNA")
    plt.title(f"{r3:.2f}", fontweight="bold")
    plt.tight_layout()
    plt.show()


def _corrcoef(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(x[m], y[m])[0, 1]


def _generate_histograms(adata, keys):
    for key in keys:
        x = adata.obs[key].to_numpy(dtype=float)
        x = x[np.isfinite(x)]

        plt.figure()
        plt.hist(x, bins=100)
        plt.xlabel(key)
        plt.ylabel("Number of cells")
        plt.title(f"Histogram: {key}")
        plt.show()