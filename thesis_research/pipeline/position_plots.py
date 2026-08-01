from pathlib import Path

from anndata import AnnData
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Render FOV maps at a fixed data scale so a tile is always the same size on the
# page and the layout reflects true pixel distances, regardless of tissue width.
PX_PER_INCH = 13000.0  # data px per figure inch (constant across every FOV plot)
FOV_TILE_FRAC = 0.92  # shrink each tile slightly so neighbouring FOVs stay distinct
MIN_FIG_IN = 3.5
MAX_FIG_IN = 18.0

from thesis_research.utils.columns import (
    X_GLOBAL_PX,
    Y_GLOBAL_PX,
    AREA,
    FOV,
    SAMPLE_ID,
    CENTER_Y_GLOBAL_PX,
    CENTER_X_GLOBAL_PX,
    SLICE_ID,
    AREA_MICROMETERS,
    SLICE_TYPE,
)
from thesis_research.utils.entity_type import (
    SampleEntityType,
    get_sample_resource_path,
    get_output_dir,
)


def generate_position_plots(adata: AnnData, run_id: str) -> None:
    print("🕒 Generating position plots...")
    sample_id = adata.uns[SAMPLE_ID]
    output_dir = get_output_dir(run_id, sample_id)

    plot_fov_positions(sample_id, output_dir)

    slice_to_color = _get_slice_to_color(list(adata.obs[SLICE_ID].cat.categories))

    plot_fov_positions_all_slices(adata, sample_id, output_dir, slice_to_color)

    for slice_id in adata.obs[SLICE_ID].cat.categories:
        plot_fov_positions_for_slice(run_id, adata, sample_id, slice_id, slice_to_color)

    plot_cells_positions_with_area(adata, output_dir)


def plot_fov_positions(
    sample_id: str,
    output_dir: Path,
    faulty_fovs=None,
    faulty_color="#d95f5f",
) -> None:
    """Generate a plot showing the FOVs based on their global positions."""
    fov_df = pd.read_csv(get_sample_resource_path(SampleEntityType.FOV_POSITIONS_FILE, sample_id))

    fov_df[FOV] = pd.to_numeric(fov_df[FOV], errors="coerce").astype("Int64")
    fov_df = fov_df[fov_df[FOV].notna()]

    x = fov_df[X_GLOBAL_PX].to_numpy(float)
    y = fov_df[Y_GLOBAL_PX].to_numpy(float)
    fov_labels = fov_df[FOV].to_numpy()
    pitch_x, pitch_y = _infer_fov_pitch(x, y)

    fig, ax = plt.subplots(figsize=_fig_size_for_extent(x, y, pitch_x, pitch_y), dpi=200)

    faulty_set = set(int(f) for f in faulty_fovs) if faulty_fovs is not None else set()
    is_faulty = fov_df[FOV].isin(faulty_set).to_numpy()

    _draw_fov_tiles(ax, x[~is_faulty], y[~is_faulty], "#cfe8f3", pitch_x, pitch_y)
    _draw_fov_tiles(
        ax, x[is_faulty], y[is_faulty], faulty_color, pitch_x, pitch_y,
        edgecolor="black", linewidth=0.6,
    )

    for xi, yi, label in zip(x, y, fov_labels):
        if pd.isna(label):
            continue
        ax.text(xi, yi, str(label), ha="center", va="center", fontsize=3, color="black")

    ax.set_title(f"{sample_id} FOV Positions", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    _frame_fov_axes(ax, x, y, pitch_x, pitch_y)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "fov_positions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fov_positions_all_slices(
    adata: AnnData,
    sample_id: str,
    output_dir: Path,
    slice_to_color: dict[int, tuple],
) -> None:
    """Generate a plot showing the FOVs slices based on their global positions."""
    fov_df = pd.read_csv(get_sample_resource_path(SampleEntityType.FOV_POSITIONS_FILE, sample_id))
    fov_df[FOV] = pd.to_numeric(fov_df[FOV], errors="coerce").astype("Int64")
    fov_df = fov_df[fov_df[FOV].notna()]

    fov_to_slice = (
        adata.obs[[FOV, SLICE_ID]]
        .dropna()
        .drop_duplicates(subset=[FOV])
        .set_index(FOV)[SLICE_ID]
        .to_dict()
    )
    fov_df = _add_slice_column(fov_df, fov_to_slice)

    x = fov_df[X_GLOBAL_PX].to_numpy(float)
    y = fov_df[Y_GLOBAL_PX].to_numpy(float)
    fov_labels = fov_df[FOV].to_numpy()
    pitch_x, pitch_y = _infer_fov_pitch(x, y)
    slice_index = pd.to_numeric(fov_df[SLICE_ID], errors="coerce")
    slice_ids = sorted(slice_index.dropna().unique().astype(int).tolist())

    fig, ax = plt.subplots(figsize=_fig_size_for_extent(x, y, pitch_x, pitch_y), dpi=200)

    tumor_slice_ids = (
        adata.obs.loc[adata.obs[SLICE_TYPE] == "tumor", SLICE_ID].dropna().unique().tolist()
    )

    for slice_id in slice_ids:
        m = (slice_index == slice_id).to_numpy()
        if not np.any(m):
            continue
        _draw_fov_tiles(ax, x[m], y[m], slice_to_color[slice_id], pitch_x, pitch_y)

        if slice_id in tumor_slice_ids:
            cx, cy = np.median(x[m]), np.median(y[m])
            ax.scatter(
                [cx],
                [cy],
                s=220,
                marker="*",
                c="black",
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )

    for xi, yi, label in zip(x, y, fov_labels):
        if pd.isna(label):
            continue
        ax.text(xi, yi, str(label), ha="center", va="center", fontsize=3, color="black")

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=slice_to_color[slice_id],
            markeredgecolor="none",
            markersize=10,
            label=slice_id,
        )
        for slice_id in slice_ids
    ]
    ax.legend(handles=handles, title="Slice ID", loc="upper right", frameon=True)

    ax.set_title(f"{sample_id} FOV Positions Sliced", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    _frame_fov_axes(ax, x, y, pitch_x, pitch_y)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / "fov_positions_sliced.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fov_positions_for_slice(
    run_id: str, adata: AnnData, sample_id: str, slice_id: str, slice_to_color: dict[int, tuple]
) -> None:
    output_dir = get_output_dir(run_id, sample_id)

    fov_df = pd.read_csv(get_sample_resource_path(SampleEntityType.FOV_POSITIONS_FILE, sample_id))
    fov_df[FOV] = pd.to_numeric(fov_df[FOV], errors="coerce").astype("Int64")
    fov_df = fov_df[fov_df[FOV].notna()]
    adata = adata[adata.obs[SLICE_ID] == slice_id].copy()
    fov_to_slice = (
        adata.obs[[FOV, SLICE_ID]]
        .dropna()
        .drop_duplicates(subset=[FOV])
        .set_index(FOV)[SLICE_ID]
        .to_dict()
    )
    fov_df = _add_slice_column(fov_df, fov_to_slice)

    # Pitch is inferred from the full grid so every slice renders at the same tile size.
    pitch_x, pitch_y = _infer_fov_pitch(
        fov_df[X_GLOBAL_PX].to_numpy(float), fov_df[Y_GLOBAL_PX].to_numpy(float)
    )

    fov_df = fov_df[fov_df[SLICE_ID] == slice_id]
    x = fov_df[X_GLOBAL_PX].to_numpy(float)
    y = fov_df[Y_GLOBAL_PX].to_numpy(float)
    labels = fov_df[FOV].astype("Int64").to_numpy()

    n_fovs = int(adata.obs[FOV].nunique())
    n_cells_slice = adata.n_obs
    area_sum = pd.to_numeric(adata.obs[AREA_MICROMETERS]).sum()

    fig, ax = plt.subplots(
        figsize=_fig_size_for_extent(x, y, pitch_x, pitch_y), dpi=300, constrained_layout=False
    )
    _draw_fov_tiles(ax, x, y, slice_to_color[int(slice_id)], pitch_x, pitch_y)
    for xi, yi, lab in zip(x, y, labels):
        if pd.notna(lab):
            ax.text(xi, yi, str(int(lab)), ha="center", va="center", fontsize=2, color="black")

    area_txt = f"{area_sum:,.0f}"
    ax.set_title(
        f"Slice {slice_id} | FOVs {n_fovs} | Cells {n_cells_slice:,} | Area {area_txt} um",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Global X (px)")
    ax.set_ylabel("Global Y (px)")
    _frame_fov_axes(ax, x, y, pitch_x, pitch_y)
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"{sample_id} — FOV Positions", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"fov_positions_slice_{slice_id}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
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
    s=1,  # dot size (matplotlib "s" is area in points^2)
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
    handles += [
        plt.Line2D([], [], linestyle="", marker="o", markersize=6, color=col) for col in lv_colors
    ]
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
    ax.scatter(
        xy[:, 1],
        xy[:, 0],  # reverse columns like xy[, 2:1]
        s=s,
        c=colors,
        marker="o",
        linewidths=0,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.show()


def _add_slice_column(fov_df: pd.DataFrame, fov_to_slice: dict[str, str]) -> pd.DataFrame:
    """Add a 'slice' column to the FOV DataFrame based on the provided slices."""
    fov_df = fov_df.copy()
    fov_df[FOV] = pd.to_numeric(fov_df[FOV], errors="coerce").astype("Int64")
    fov_to_slice = {int(k): int(v) for k, v in fov_to_slice.items()}

    fov_df[SLICE_ID] = fov_df[FOV].map(fov_to_slice).astype("Int64")
    return fov_df


def _infer_fov_pitch(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Infer the FOV tile size (px) from the spacing of the FOV grid.

    Uses the median nearest-neighbour distance between FOV reference points,
    which equals the true FOV pitch: every interior FOV abuts a neighbour one
    pitch away, and the median is robust to the per-row stage jitter that makes
    raw coordinate diffs unreliable. FOVs are square, so the same pitch is used
    for both axes.
    """
    from scipy.spatial import cKDTree

    pts = np.column_stack([x, y])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 2:
        return float("nan"), float("nan")

    dist, _ = cKDTree(pts).query(pts, k=2)
    pitch = float(np.median(dist[:, 1]))
    return pitch, pitch


def _fig_size_for_extent(
    x: np.ndarray, y: np.ndarray, pitch_x: float, pitch_y: float
) -> tuple[float, float]:
    """Figure size (inches) for the data extent at a fixed px-per-inch scale.

    Scaling both axes by the same factor keeps tiles square and the same size on
    every plot; bounds only cap pathologically large/small canvases.
    """
    w_px = (np.nanmax(x) - np.nanmin(x)) + pitch_x
    h_px = (np.nanmax(y) - np.nanmin(y)) + pitch_y
    w_in, h_in = w_px / PX_PER_INCH, h_px / PX_PER_INCH

    big = max(w_in, h_in)
    if big > MAX_FIG_IN:
        w_in, h_in = w_in * MAX_FIG_IN / big, h_in * MAX_FIG_IN / big
    small = min(w_in, h_in)
    if small < MIN_FIG_IN:
        w_in, h_in = w_in * MIN_FIG_IN / small, h_in * MIN_FIG_IN / small
    return w_in, h_in


def _draw_fov_tiles(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    facecolor,
    pitch_x: float,
    pitch_y: float,
    edgecolor="none",
    linewidth: float = 0.0,
    zorder: int = 1,
) -> None:
    """Draw FOVs as data-coordinate rectangles so they scale with real distance."""
    w, h = pitch_x * FOV_TILE_FRAC, pitch_y * FOV_TILE_FRAC
    rects = [Rectangle((xi - w / 2, yi - h / 2), w, h) for xi, yi in zip(x, y)]
    coll = PatchCollection(
        rects,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
    )
    ax.add_collection(coll)


def _frame_fov_axes(
    ax, x: np.ndarray, y: np.ndarray, pitch_x: float, pitch_y: float, margin_frac: float = 0.04
) -> None:
    """Set equal-aspect limits around the FOV tiles (patches don't autoscale)."""
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    mx = pitch_x * (0.5 + margin_frac) + (xmax - xmin) * margin_frac
    my = pitch_y * (0.5 + margin_frac) + (ymax - ymin) * margin_frac
    ax.set_xlim(xmin - mx, xmax + mx)
    ax.set_ylim(ymin - my, ymax + my)
    ax.set_aspect("equal", adjustable="box")


def _get_slice_to_color(slice_ids: list[str]) -> dict[int, tuple]:
    slice_ids = sorted({int(s) for s in slice_ids if pd.notna(s)})
    cmap = plt.get_cmap("tab20")

    palette = {1: cmap(19), 2: cmap(13), 3: cmap(3), 4: cmap(5), 5: cmap(7), 6: cmap(9)}
    return {sid: palette[sid] if sid in palette else cmap((sid - 1) % cmap.N) for sid in slice_ids}
