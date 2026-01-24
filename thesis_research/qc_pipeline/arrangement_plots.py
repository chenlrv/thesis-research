from pathlib import Path

from anndata import AnnData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from thesis_research.utils.columns import X_GLOBAL_PX, Y_GLOBAL_PX, AREA, FOV, SAMPLE_ID, CENTER_Y_GLOBAL_PX, \
    CENTER_X_GLOBAL_PX
from thesis_research.utils.entity_type import EntityType, get_path, get_output_dir


def plot_positions(adata: AnnData, run_id: str) -> None:
    print("Generating position plots...")
    sample_id = adata.uns[SAMPLE_ID]
    output_dir = get_output_dir(sample_id, run_id)

    plot_fov_positions(sample_id, output_dir)
    plot_cells_positions_with_area(adata, output_dir)


def plot_fov_positions(
    sample_id: str, output_dir: Path, faulty_fovs=None, faulty_color="#d95f5f"
) -> None:
    """Generate a plot showing the FOVs based on their global positions."""
    fov_df = pd.read_csv(get_path(EntityType.FOV_POSITIONS_FILE, sample_id))

    fov_df[FOV] = pd.to_numeric(fov_df[FOV], errors="coerce").astype("Int64")
    fov_df = fov_df[fov_df[FOV].notna()]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    x = fov_df[X_GLOBAL_PX].to_numpy(float)
    y = fov_df[Y_GLOBAL_PX].to_numpy(float)
    fov_labels = fov_df[FOV].to_numpy()

    faulty_set = set(int(f) for f in faulty_fovs) if faulty_fovs is not None else set()
    is_faulty = fov_df[FOV].isin(faulty_set).to_numpy()

    ax.scatter(x[~is_faulty], y[~is_faulty], s=100, marker="s", c="#cfe8f3", edgecolors="none")
    ax.scatter(
        x[is_faulty],
        y[is_faulty],
        s=110,
        marker="s",
        c=faulty_color,
        edgecolors="black",
        linewidths=0.6,
    )

    for xi, yi, label in zip(x, y, fov_labels):
        if pd.isna(label):
            continue
        ax.text(xi, yi, str(label), ha="center", va="center", fontsize=3, color="black")

    ax.set_title("FOV Arrangement", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "fov_positions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cells_positions_with_area(adata: AnnData, output_dir: Path) -> None:
    """Generate a scatter plot of cells colored by their area.
    """

    x = adata.obs[CENTER_X_GLOBAL_PX].to_numpy(float)
    y = adata.obs[CENTER_Y_GLOBAL_PX].to_numpy(float)
    a = adata.obs[AREA].to_numpy(float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(a)

    # Clip color scale to make it readable (1st–99th percentile)
    vmin, vmax = np.percentile(a[m], [1, 99])

    fig = plt.figure(figsize=(6, 6), dpi=400)
    sc = plt.scatter(x[m], y[m], c=a[m], s=0.15, linewidths=0, vmin=vmin, vmax=vmax)
    plt.xlabel("Global X Position (px)")
    plt.ylabel("Global Y Position (px)")
    plt.title("Cell area (1–99% percentile)", {"fontsize": 16, "fontweight": "bold"})
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=AREA)
    plt.tight_layout()
    plt.savefig(output_dir / "cell_positions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
