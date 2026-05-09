from __future__ import annotations

from pathlib import Path

import anndata as ad
import napari
import numpy as np
import pandas as pd
import zarr
from matplotlib.colors import to_rgba
from skimage.segmentation import find_boundaries


_GREY = np.array([0.5, 0.5, 0.5, 0.7], dtype=np.float32)


def show_zarr_slice(
    zarr_path: str | Path,
    cell_index_path: str | Path,
    adata: ad.AnnData,
    color_col: str = "cluster",
    opacity: float = 0.85,
) -> napari.Viewer:
    """
    Open a multi-scale zarr label store in napari, colored by adata annotation.

    Parameters
    ----------
    zarr_path:
        Path to the labels.zarr/ directory produced by prepare_viewer_zarr().
    cell_index_path:
        Path to cell_index.parquet produced by prepare_viewer_zarr().
    adata:
        AnnData with obs['cell_id'] and uns['{color_col}_colors'] palette.
    color_col:
        Column in adata.obs used for cell coloring.
    opacity:
        Labels layer opacity (0–1).
    """
    zarr_path = Path(zarr_path)
    cell_index_path = Path(cell_index_path)

    # ── Load zarr pyramid ───────────────────────────────────────────────────
    store = zarr.open_group(str(zarr_path), mode="r")
    n_levels = sum(1 for k in store.keys() if k.isdigit())
    # napari expects multiscale list ordered finest → coarsest; flip vertically
    pyramid = [np.asarray(store[str(i)])[::-1] for i in range(n_levels)]
    base_shape = pyramid[0].shape
    print(f"Zarr: {n_levels} levels, base {base_shape[0]}×{base_shape[1]} px")

    # ── Load cell index ─────────────────────────────────────────────────────
    idx_df = pd.read_parquet(cell_index_path)  # columns: cell_idx (int32), cell_id (str)

    # ── Build annotation lookup ─────────────────────────────────────────────
    obs = adata.obs.copy()
    obs[color_col] = obs[color_col].astype("category")
    cats = list(obs[color_col].cat.categories)

    palette_key = f"{color_col}_colors"
    if palette_key not in adata.uns:
        raise KeyError(
            f"'{palette_key}' not found in adata.uns. "
            f"Run sc.pl.umap(adata, color='{color_col}', show=False) first."
        )

    cat2rgba: dict[str, tuple] = {
        cat: tuple(float(v) for v in to_rgba(col))
        for cat, col in zip(cats, adata.uns[palette_key])
    }

    # cell_id → annotation category
    cell_id_series = obs["cell_id"].astype(str) if "cell_id" in obs.columns else obs.index.astype(str)
    cell2cat: dict[str, str] = dict(zip(cell_id_series, obs[color_col].astype(str)))

    # ── Build integer label → RGBA dict ────────────────────────────────────
    grey_tuple = tuple(float(v) for v in _GREY)
    transparent_tuple = (0.0, 0.0, 0.0, 0.0)

    cell_ids = idx_df["cell_id"].astype(str).values
    annotations = [cell2cat.get(cid) for cid in cell_ids]
    colors = [
        cat2rgba.get(ann, grey_tuple) if ann is not None else grey_tuple
        for ann in annotations
    ]
    cell_idxs = idx_df["cell_idx"].to_numpy(dtype=np.int32)

    color_dict: dict[int, tuple] = {0: transparent_tuple}
    color_dict.update(zip(cell_idxs.tolist(), colors))

    n_matched = sum(1 for ann in annotations if ann is not None)
    print(f"Matched {n_matched:,}/{len(cell_ids):,} cells to adata annotations")

    # ── Console legend ──────────────────────────────────────────────────────
    print(f"\nColor legend ({color_col}):")
    for cat, rgba in cat2rgba.items():
        r, g, b = (int(v * 255) for v in rgba[:3])
        print(f"  \033[38;2;{r};{g};{b}m■\033[0m  {cat}")
    print()

    # ── Open napari ─────────────────────────────────────────────────────────
    v = napari.Viewer(title=f"Cell Viewer — {color_col}")
    layer = v.add_labels(
        pyramid,
        multiscale=True,
        name=color_col,
        opacity=opacity,
    )
    layer.color = color_dict

    napari.run()
    return v


def show_zarr_binary(
    zarr_path: str | Path,
    cell_index_path: str | Path,
    adata: ad.AnnData,
    bool_col: str = "pred_tumor_XGBoost",
    opacity: float = 0.85,
    borders: bool = True,
) -> napari.Viewer:
    """
    Open a multi-scale zarr label store in napari with a binary coloring:
    cells where bool_col is True/1 → red; all other cells → gray.

    Parameters
    ----------
    zarr_path:
        Path to labels.zarr/ produced by prepare_viewer_zarr().
    cell_index_path:
        Path to cell_index.parquet produced by prepare_viewer_zarr().
    adata:
        AnnData whose obs index are cell barcodes and obs[bool_col] is
        a boolean/int column (1=tumor, 0=not tumor).
    bool_col:
        Column in adata.obs to use for the binary split.
    opacity:
        Labels layer opacity (0–1).
    """
    zarr_path = Path(zarr_path)
    cell_index_path = Path(cell_index_path)

    # ── Load zarr metadata ──────────────────────────────────────────────────
    store = zarr.open_group(str(zarr_path), mode="r")
    n_levels = sum(1 for k in store.keys() if k.isdigit())
    base_shape = store["0"].shape
    print(f"Zarr: {n_levels} levels, base {base_shape[0]}×{base_shape[1]} px")

    # ── Load cell index ─────────────────────────────────────────────────────
    idx_df = pd.read_parquet(cell_index_path)
    cell_ids  = idx_df["cell_id"].astype(str).values
    cell_idxs = idx_df["cell_idx"].to_numpy(dtype=np.int32)

    # ── Build cell_id → is_tumor lookup ────────────────────────────────────
    cell_id_col = (
        adata.obs["cell_id"].astype(str)
        if "cell_id" in adata.obs.columns
        else adata.obs.index.astype(str)
    )
    cell2tumor: dict[str, bool] = dict(
        zip(cell_id_col, adata.obs[bool_col].astype(bool))
    )

    annotations = [cell2tumor.get(cid) for cid in cell_ids]
    n_tumor = sum(1 for a in annotations if a)
    print(f"Tumor cells (red):  {n_tumor:,}")
    print(f"Other cells (gray): {len(cell_ids) - n_tumor:,}")

    # ── Build integer LUT: 0=background, 1=gray, 2=red ─────────────────────
    # Remap every zarr level to a 3-value array so the color dict stays tiny
    # (napari's label color dict is unreliable when mapping 100k+ entries to
    # only 2 colors; a 3-entry dict always works correctly).
    max_label = int(cell_idxs.max()) if len(cell_idxs) > 0 else 0
    lut = np.ones(max_label + 1, dtype=np.uint8)  # 1 = gray by default
    lut[0] = 0  # background stays transparent
    for cidx, ann in zip(cell_idxs.tolist(), annotations):
        if ann:
            lut[cidx] = 2  # tumor → red

    # Bake colors directly into RGBA arrays — bypasses napari color dict entirely
    grey_px = np.array([128, 128, 128, 128], dtype=np.uint8)
    red_px  = np.array([255,   0,   0, 230], dtype=np.uint8)

    print("Remapping zarr levels…")
    border_px = np.array([20, 20, 20, 200], dtype=np.uint8)  # dark border

    rgba_pyramid = []
    for i in range(n_levels):
        level = np.asarray(store[str(i)])           # int32, unique cell indices
        binary = lut[np.clip(level, 0, max_label)]  # 0=bg, 1=gray, 2=red
        H, W = binary.shape
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[binary == 1] = grey_px
        rgba[binary == 2] = red_px
        if borders:
            rgba[find_boundaries(level, mode="inner")] = border_px
        rgba_pyramid.append(rgba[::-1])  # flip vertically

    # ── Open napari ─────────────────────────────────────────────────────────
    v = napari.Viewer(title=f"Tumor Map — {bool_col}")
    layer = v.add_image(
        rgba_pyramid,
        multiscale=True,
        name=bool_col,
        rgb=True,
    )

    napari.run()
    return v
