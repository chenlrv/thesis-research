from pathlib import Path

from anndata import AnnData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from thesis_research.qc_pipeline.sample_slice import SampleSlice
from thesis_research.utils.columns import (
    X_GLOBAL_PX,
    Y_GLOBAL_PX,
    AREA,
    FOV,
    SAMPLE_ID,
    CENTER_Y_GLOBAL_PX,
    CENTER_X_GLOBAL_PX,
    SLICE_ID,
)
from thesis_research.utils.entity_type import EntityType, get_path, get_output_dir


def generate_position_plots(adata: AnnData, slices: list[SampleSlice], run_id: str) -> None:
    print("Generating position plots...")
    sample_id = adata.uns[SAMPLE_ID]
    output_dir = get_output_dir(sample_id, run_id)

    plot_fov_positions(sample_id, output_dir)
    plot_fov_positions_sliced(sample_id, slices, output_dir)
    plot_cells_positions_with_area(adata, output_dir)


def plot_fov_positions(
    sample_id: str,
    output_dir: Path,
    faulty_fovs=None,
    faulty_color="#d95f5f",
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

    ax.set_title(f"{sample_id} FOV Positions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "fov_positions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fov_positions_sliced(
    sample_id: str,
    slices: list[SampleSlice],
    output_dir: Path,
) -> None:
    """Generate a plot showing the FOVs slices based on their global positions."""
    fov_df = pd.read_csv(get_path(EntityType.FOV_POSITIONS_FILE, sample_id))
    fov_df[FOV] = pd.to_numeric(fov_df[FOV], errors="coerce").astype("Int64")
    fov_df = fov_df[fov_df[FOV].notna()]
    fov_df = _add_slice_column(fov_df, slices)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    x = fov_df[X_GLOBAL_PX].to_numpy(float)
    y = fov_df[Y_GLOBAL_PX].to_numpy(float)
    fov_labels = fov_df[FOV].to_numpy()
    slice_index = fov_df[SLICE_ID].to_numpy(int)

    rng = np.random.default_rng()
    slice_to_color = {si: rng.random(3) for si in np.unique(slice_index)}

    for i in np.unique(slice_index):
        m = slice_index == i
        if not np.any(m):
            continue
        ax.scatter(x[m], y[m], s=100, marker="s", c=[slice_to_color[i]], edgecolors="none")

    for xi, yi, label in zip(x, y, fov_labels):
        if pd.isna(label):
            continue
        ax.text(xi, yi, str(label), ha="center", va="center", fontsize=3, color="black")

    ax.set_title(f"{sample_id} FOV Positions Sliced", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "fov_positions_sliced.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cells_positions_with_area(adata: AnnData, output_dir: Path) -> None:
    """Generate a scatter plot of cells colored by their area."""
    sample_id = adata.uns[SAMPLE_ID]

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
    plt.title(f"{sample_id} Cell area (1–99% percentile)", {"fontsize": 16, "fontweight": "bold"})
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(sc, label=AREA)
    plt.tight_layout()
    plt.savefig(output_dir / "cell_positions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_counts_over_space(
    adata,
    x_col="CenterX_global_px",
    y_col="CenterY_global_px",
    count_col="nCount_RNA",
    vmin=10,
    vmax=5000,
    legendvals=(10, 100, 1000, 5000),
    s=1,                 # dot size (matplotlib "s" is area in points^2)
    cmap="inferno",
):
    """
    Spatial scatter of cells colored by log2(total counts), with counts clamped to [vmin, vmax].
    """

    x = adata.obs[x_col].to_numpy(float)
    y = adata.obs[y_col].to_numpy(float)
    counts = adata.obs[count_col].to_numpy(float)

    # clamp counts then log2-transform
    c = np.clip(counts, vmin, vmax)
    logc = np.log2(c)

    # normalize to [0, 1] using the same logic as the R code
    lo = np.log2(vmin)
    hi = np.log2(vmax)
    norm = (logc - lo) / (hi - lo)

    # discrete 101-color palette like viridis_pal(...)(101)
    palette = plt.get_cmap(cmap)(np.linspace(0, 1, 101))
    idx = np.clip(np.rint(norm * 100).astype(int), 0, 100)
    colors = palette[idx]

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=s, c=colors, marker="o", linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # legend colors computed with the same mapping
    legendvals = np.asarray(legendvals, dtype=float)
    lv = np.clip(legendvals, vmin, vmax)
    lv_norm = (np.log2(lv) - lo) / (hi - lo)
    lv_idx = np.clip(np.rint(lv_norm * 100).astype(int), 0, 100)
    lv_colors = palette[lv_idx]

    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=0, color="none")]
    labels = ["Total counts:"]
    handles += [plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=col) for col in lv_colors]
    labels += [str(int(v)) for v in legendvals]

    ax.legend(handles, labels, loc="lower right", frameon=True)

    plt.show()
    return fig, ax


def plot_flagged_cells(xy, flag, s=1):
    """
    xy   : (n_cells, 2) array-like of spatial coordinates
    flag : boolean array (True = flagged)
    """

    xy = np.asarray(xy)
    flag = np.asarray(flag)

    colors = np.where(flag, "red", "lightgrey")

    fig, ax = plt.subplots()
    ax.scatter(xy[:, 1], xy[:, 0],  # reverse columns like xy[, 2:1]
               s=s,
               c=colors,
               marker="o",
               linewidths=0)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()

def _add_slice_column(fov_df: pd.DataFrame, slices: list[SampleSlice]) -> pd.DataFrame:
    """Add a 'slice' column to the FOV DataFrame based on the provided slices."""
    fov_df = fov_df.copy()
    slice_mapping = {}
    for i, sample_slice in enumerate(slices):
        for fov in sample_slice.fov_ids:
            slice_mapping[fov] = i

    fov_df[SLICE_ID] = fov_df[FOV].map(slice_mapping)
    return fov_df
