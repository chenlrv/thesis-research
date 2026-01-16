import matplotlib.pyplot as plt
import pandas as pd


def generate_fov_arrangement_plot(fov_positions_path: str) -> None:
    """Generate a plot showing the arrangement of FOVs based on their global positions.

    Args:
        fov_path (str): Path to the CSV file containing FOV positions.
    """
    fov_df = pd.read_csv(fov_positions_path)

    FOV_COL = "fov"
    X_COL = "x_global_px"
    Y_COL = "y_global_px"

    fov_df[FOV_COL] = (
        fov_df[FOV_COL].astype(str)
        .str.replace(r"^FOV[_ ]*", "", regex=True)
    )
    fov_df[FOV_COL] = pd.to_numeric(fov_df[FOV_COL], errors="coerce").astype("Int64")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    x = fov_df[X_COL].to_numpy(float)
    y = fov_df[Y_COL].to_numpy(float)
    labels = fov_df[FOV_COL].to_numpy()

    ax.scatter(x, y, s=100, marker="s", c="#cfe8f3", edgecolors="none")

    for xi, yi, label in zip(x, y, labels):
        if pd.isna(label):
            continue
        ax.text(xi, yi, str(int(label)), ha="center", va="center", fontsize=3, color="black")

    ax.set_title("FOV Arrangement", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def generate_cells_with_area_plot(adata) -> None:
    """Generate a scatter plot of cells colored by their area.

    Args:
        adata (AnnData): Annotated data matrix containing cell information.
    """
    import numpy as np

    XCOL = "CenterX_global_px"
    YCOL = "CenterY_global_px"
    ACOL = "Area"

    x = adata.obs[XCOL].to_numpy(float)
    y = adata.obs[YCOL].to_numpy(float)
    a = adata.obs[ACOL].to_numpy(float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(a)

    # Clip color scale to make it readable (1st–99th percentile)
    vmin, vmax = np.percentile(a[m], [1, 99])

    plt.figure(figsize=(6, 6), dpi=400)
    sc = plt.scatter(x[m], y[m], c=a[m], s=0.15, linewidths=0, vmin=vmin, vmax=vmax, cmap="viridis_r")
    plt.xlabel("Global X Position (px)")
    plt.ylabel("Global Y Position (px)")
    plt.title("Cell area (1–99% percentile)", {"fontsize": 16, "fontweight": "bold"})
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=ACOL)
    plt.tight_layout()
    plt.show()
