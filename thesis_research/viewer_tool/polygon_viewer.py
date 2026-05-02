from __future__ import annotations

from pathlib import Path

import anndata as ad
import napari
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


_GREY = np.array([0.5, 0.5, 0.5, 0.7], dtype=np.float32)


def show_cell_polygons(
    polygons_path: str | Path,
    adata: ad.AnnData,
    color_col: str = "cell_type",
    fovs: list[int] | None = None,
    use_global: bool = True,
    max_shapes: int = 10_000,
    max_raster_px: int = 10_000,
) -> napari.Viewer:
    """
    Render cell segmentation polygons in napari, colored by adata annotation.

    For small FOV counts (< max_shapes cells) uses napari Shapes for true
    vector rendering. For larger datasets rasterizes to an image first.

    Parameters
    ----------
    polygons_path:
        Path to the CosMx polygons CSV.
    adata:
        AnnData with obs column 'cell_id' and uns['{color_col}_colors'] palette.
    color_col:
        Column in adata.obs to color cells by.
    fovs:
        If given, only load polygons for these FOV IDs.
    use_global:
        Use x_global_px / y_global_px.  Set False for local px (single FOV).
    max_shapes:
        Switch to rasterized mode above this many unique cells.
    max_raster_px:
        Longest side of the rasterized image in pixels.
    """
    polys_df = _load_polygons(polygons_path, fovs, use_global)
    cat2rgba, cell2color = _build_color_map(adata, color_col)

    n_cells = polys_df["cell"].nunique()
    print(f"Loaded {n_cells} cells.")

    v = napari.Viewer(title="Cell Polygon Viewer")

    if n_cells <= max_shapes:
        _add_shapes_layer(v, polys_df, cell2color, color_col)
    else:
        print(f"{n_cells} cells exceeds max_shapes={max_shapes}. Rasterizing...")
        _add_raster_layer(v, polys_df, cell2color, max_raster_px)

    _print_legend(cat2rgba, color_col)
    napari.run()
    return v


# ── data loading ──────────────────────────────────────────────────────────────

def _load_polygons(
    path: str | Path, fovs: list[int] | None, use_global: bool
) -> pd.DataFrame:
    x_col = "x_global_px" if use_global else "x_local_px"
    y_col = "y_global_px" if use_global else "y_local_px"
    df = pd.read_csv(path, usecols=["fov", "cell", x_col, y_col])
    if fovs is not None:
        df = df[df["fov"].isin(fovs)].copy()
    return df.rename(columns={x_col: "x", y_col: "y"})


def _build_color_map(
    adata: ad.AnnData, color_col: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    obs = adata.obs.copy()
    obs[color_col] = obs[color_col].astype("category")
    cats = list(obs[color_col].cat.categories)

    palette_key = f"{color_col}_colors"
    if palette_key not in adata.uns:
        raise KeyError(
            f"'{palette_key}' not found in adata.uns. "
            f"Run sc.pl.umap(adata, color='{color_col}', show=False) once."
        )

    palette = list(adata.uns[palette_key])
    cat2rgba: dict[str, np.ndarray] = {
        cat: np.array(to_rgba(col), dtype=np.float32)
        for cat, col in zip(cats, palette)
    }
    cell2color: dict[str, np.ndarray] = {
        str(cell_id): cat2rgba.get(str(cat), _GREY)
        for cell_id, cat in zip(obs["cell_id"], obs[color_col])
    }
    return cat2rgba, cell2color


# ── napari layers ─────────────────────────────────────────────────────────────

def _add_shapes_layer(
    v: napari.Viewer,
    df: pd.DataFrame,
    cell2color: dict[str, np.ndarray],
    name: str,
) -> None:
    polygon_list, face_colors = [], []
    for cell_id, group in df.groupby("cell", sort=False):
        polygon_list.append(group[["y", "x"]].to_numpy(dtype=np.float32))
        face_colors.append(cell2color.get(str(cell_id), _GREY))

    v.add_shapes(
        polygon_list,
        shape_type="polygon",
        face_color=np.stack(face_colors),
        edge_color="white",
        edge_width=0.5,
        opacity=0.75,
        name=name,
    )


def _add_raster_layer(
    v: napari.Viewer,
    df: pd.DataFrame,
    cell2color: dict[str, np.ndarray],
    max_px: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    x_min = float(df["x"].min())
    y_min = float(df["y"].min())
    x_max = float(df["x"].max())
    y_max = float(df["y"].max())

    W = x_max - x_min
    H = y_max - y_min
    scale = min(max_px / W, max_px / H, 1.0)

    dpi = 150
    fig, ax = plt.subplots(figsize=(W * scale / dpi, H * scale / dpi), dpi=dpi)
    # Invert y so row-0 = y_min, matching image/napari convention
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    print("Building polygon patches...")
    patches, facecolors = [], []
    for cell_id, group in df.groupby("cell", sort=False):
        xy = group[["x", "y"]].to_numpy(dtype=np.float64)
        xy[:, 0] -= x_min
        xy[:, 1] -= y_min
        patches.append(MplPolygon(xy, closed=True))
        c = cell2color.get(str(cell_id), _GREY)
        facecolors.append(c[:3].tolist())

    print("Rasterizing...")
    ax.add_collection(PatchCollection(patches, facecolors=facecolors, edgecolors="none"))
    fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Place image at correct world coordinates
    pixel_size = 1.0 / scale
    v.add_image(
        img,
        name="cells (rasterized)",
        translate=(y_min, x_min),
        scale=(pixel_size, pixel_size),
        rgb=True,
    )


# ── legend ────────────────────────────────────────────────────────────────────

def _print_legend(cat2rgba: dict[str, np.ndarray], color_col: str) -> None:
    print(f"\nColor legend ({color_col}):")
    for cat, rgba in cat2rgba.items():
        r, g, b = (int(v * 255) for v in rgba[:3])
        print(f"  \033[38;2;{r};{g};{b}m■\033[0m  {cat}")
