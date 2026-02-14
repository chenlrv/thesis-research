import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from anndata import AnnData


def plot_spatial(adata: AnnData, color_column: str = "class_name") -> None:
    for sample_id in adata.obs["sample_id"].cat.categories:
        adata_sample = adata[adata.obs["sample_id"] == sample_id].copy()
        plot_spatial_by_celltype(
            adata_sample, f"Spatial Cell Type Plot (per cell) - Sample {sample_id}", color_col=color_column
        )


def plot_spatial_by_celltype(
    adata,
    plot_title="",
    color_col="subclass_name",      # e.g. "class_name" / "subclass_name" / "mmc_label_fallback"
    x_key="CenterX_global_px",
    y_key="CenterY_global_px",
    s=0.5,
    alpha=1.0,
    dpi=300,
    figsize=(12, 12),
    invert_y=False,
    savepath=None,
):
    # ensure categorical (stable ordering)
    adata.obs[color_col] = adata.obs[color_col].astype("category")
    cats = list(adata.obs[color_col].cat.categories)

    # create or reuse colors stored in adata.uns
    palette_key = f"{color_col}_colors"
    if palette_key not in adata.uns or len(adata.uns[palette_key]) < len(cats):
        # build a palette large enough
        colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
        if len(colors) < len(cats):
            colors = (colors * (len(cats) // len(colors) + 1))[:len(cats)]
        # store as hex strings
        adata.uns[palette_key] = [plt.matplotlib.colors.to_hex(c) for c in colors[:len(cats)]]

    cat2color = dict(zip(cats, adata.uns[palette_key][:len(cats)]))

    # points
    x = adata.obs[x_key].to_numpy()
    y = adata.obs[y_key].to_numpy()
    labels = adata.obs[color_col].astype(str)
    point_colors = labels.map(cat2color).to_numpy()

    # plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(x, y, c=point_colors, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(plot_title if plot_title else f"Spatial plot colored by {color_col}")

    # legend on the right
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=cat2color[c], markeredgecolor="none", label=str(c))
        for c in cats
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, frameon=False)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
        print("Saved:", savepath)
    plt.show()


# ---- USAGE ----
adata = ad.read_h5ad('D:\\thesis-research\\resources\\cache\\adata_full.h5ad')
mmc_df = pd.read_csv('D:\\thesis-research\\resources\\map_my_cells\\cell_type_mapping.csv', skiprows=4)

mmc_df = mmc_df.copy()
mmc_df = mmc_df.set_index('cell_id')
adata.obs = adata.obs.join(mmc_df, how="left")

plot_spatial(adata)

# plot_spatial_by_celltype(adata, plot_title="MMC fallback (per cell)", color_col="mmc_label_fallback")
