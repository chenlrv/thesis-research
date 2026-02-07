import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_spatial_gene_positive(
    adata,
    plot_title: str,
    gene: str = "Lyve1",
    x_key: str = "CenterX_global_px",
    y_key: str = "CenterY_global_px",
    s: float = 0.5,
    alpha: float = 1.0,
    dpi: int = 500,
    figsize=(12, 12),
    invert_y: bool = True,
    savepath=None,
    use_raw: bool = False,  # set True if Lyve1 is in adata.raw
):
    # --- fetch coordinates ---
    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()

    # --- pick expression matrix + var names ---
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is None.")
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    if gene not in var_names:
        raise KeyError(f"Gene '{gene}' not found in var_names.")

    gi = int(np.where(var_names == gene)[0][0])

    # --- get gene expression vector ---
    if sp.issparse(X):
        expr = X[:, gi].toarray().ravel()
    else:
        expr = np.asarray(X[:, gi]).ravel()

    # --- positive if any expression > 0 ---
    is_pos = expr > 0

    # --- colors: blue for pos, gray for neg ---
    point_colors = np.where(is_pos, "#1f77b4", "#bdbdbd")  # blue, gray

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(
        x,
        y,
        c=point_colors,
        s=s,
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    n_pos = int(is_pos.sum())
    n_total = int(len(is_pos))
    ax.set_title(f"{plot_title}\n{gene}+ cells: {n_pos}/{n_total} ({n_pos/n_total:.1%})")

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("Saved:", savepath)

    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_spatial_gene_blues_better(
    adata,
    plot_title,
    gene="Lyve1",
    x_key="CenterX_global_px",
    y_key="CenterY_global_px",
    s=0.5,
    alpha_zero=0.6,
    alpha_pos=1.0,
    figsize=(12, 12),
    dpi=500,
    invert_y=True,
    savepath=None,
    use_raw=False,
    log1p=True,
    vmax_quantile=99,
    zero_color="#e0e0e0",
):
    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()

    X = adata.raw.X if use_raw else adata.X
    var_names = adata.raw.var_names if use_raw else adata.var_names
    if gene not in var_names:
        raise KeyError(f"Gene '{gene}' not found.")

    gi = int(np.where(var_names == gene)[0][0])
    expr = X[:, gi].toarray().ravel() if sp.issparse(X) else np.asarray(X[:, gi]).ravel()
    if log1p:
        expr = np.log1p(expr)

    is_pos = expr > 0
    pos = expr[is_pos]
    if pos.size == 0:
        raise ValueError(f"No cells have {gene} > 0.")

    vmin = np.percentile(pos, 5)
    vmax = np.percentile(pos, vmax_quantile)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # zeros first
    ax.scatter(
        x[~is_pos], y[~is_pos], c=zero_color, s=s, alpha=alpha_zero, linewidths=0, rasterized=True
    )

    # positives on top (colored by expression)
    sc = ax.scatter(
        x[is_pos],
        y[is_pos],
        c=expr[is_pos],
        s=s,
        alpha=alpha_pos,
        cmap="Blues",
        norm=norm,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()

    n_pos = int(is_pos.sum())
    ax.set_title(f"{plot_title}\n{gene}+ cells: {n_pos}/{len(expr)} ({n_pos/len(expr):.1%})")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{gene} expression ({'log1p' if log1p else 'linear'})")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("Saved:", savepath)
    plt.show()


# adata = anndata.read_h5ad('D:\\thesis-research\\resources\\merged_adata.h5ad')

# plot_spatial_gene_blues_better(adata, "Lyve1 Expression")
