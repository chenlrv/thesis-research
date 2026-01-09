import matplotlib.pyplot as plt
import pandas as pd


def plot_spatial_clusters(
        adata,
        plot_title: str,
        color_col="celltype",
        x_key="CenterX_global_px",
        y_key="CenterY_global_px",
        s=0.5,
        alpha=1.0,
        dpi=500,
        figsize=(12, 12),
        invert_y=True,
        savepath=None
):
    adata.obs[color_col] = adata.obs[color_col].astype("category")
    cats = list(adata.obs[color_col].cat.categories)

    palette_key = f"{color_col}_colors"
    if palette_key not in adata.uns:
        raise KeyError(
            f"'{palette_key}' not found in adata.uns. "
            f"Run sc.pl.umap(adata, color='{color_col}', show=False) once."
        )

    colors = list(adata.uns[palette_key])
    cat2color = dict(zip(cats, colors))

    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()
    labels = adata.obs[color_col].astype(str).to_numpy()
    point_colors = pd.Series(labels).map(cat2color).to_numpy()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, c=point_colors, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"Spatial {plot_title} plot colored by {color_col}")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("Saved:", savepath)
    plt.show()
