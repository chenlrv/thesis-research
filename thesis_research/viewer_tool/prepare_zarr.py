from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from skimage.draw import polygon as draw_polygon


def prepare_viewer_zarr(
    polygons_path: str | Path,
    out_dir: str | Path,
    fovs: list[int] | None = None,
    max_pixels: int = 10_000,
    n_levels: int = 5,
    chunk_size: int = 512,
) -> None:
    """
    Convert a CosMx polygon CSV into a chunked multi-scale zarr label store.

    Run this once per tissue slice. Output layout inside out_dir:
        labels.zarr/0   ← base resolution (int32, shape H×W)
        labels.zarr/1   ← 2× downsampled
        ...
        cell_index.parquet  ← cell_idx (int32) | cell_id (str)

    Parameters
    ----------
    polygons_path:
        Path to the CosMx *-polygons.csv file.
    out_dir:
        Directory where labels.zarr/ and cell_index.parquet are written.
    fovs:
        If given, only include polygons whose FOV is in this list.  Use this to
        restrict processing to a single tissue slice (pass the FOV IDs for that
        slice, read from *_fov_slices.csv).
    max_pixels:
        Longest side of the base raster canvas in pixels.  Adjust upward for
        more detail (uses more RAM during rasterisation: 4 bytes × H × W).
    n_levels:
        Number of pyramid levels (each 2× coarser than the previous).
    chunk_size:
        Zarr chunk size in pixels (chunk_size × chunk_size tiles).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = out_dir / "labels.zarr"
    index_path = out_dir / "cell_index.parquet"

    # ── 1. Load polygon CSV ─────────────────────────────────────────────────
    print("Loading polygon CSV …")
    t0 = time.time()
    df = pd.read_csv(
        polygons_path,
        usecols=["fov", "cell", "x_global_px", "y_global_px"],
        dtype={"fov": np.int32, "cell": str,
               "x_global_px": np.float32, "y_global_px": np.float32},
    )
    if fovs is not None:
        fov_set = set(fovs)
        df = df[df["fov"].isin(fov_set)].copy()
        print(f"  Filtered to {len(fov_set)} FOVs")
    df.drop(columns=["fov"], inplace=True)
    print(f"  {len(df):,} rows in {time.time() - t0:.1f}s")

    # ── 2. Assign integer index to each unique cell ─────────────────────────
    unique_cells = df["cell"].unique()
    n_cells = len(unique_cells)
    print(f"  {n_cells:,} unique cells")

    cell2idx = pd.Series(
        np.arange(1, n_cells + 1, dtype=np.int32), index=unique_cells
    )
    pd.DataFrame(
        {"cell_idx": cell2idx.values, "cell_id": cell2idx.index}
    ).to_parquet(index_path, index=False)
    print(f"  Saved cell index → {index_path}")

    # ── 3. Compute canvas dimensions ────────────────────────────────────────
    x_min = float(df["x_global_px"].min())
    x_max = float(df["x_global_px"].max())
    y_min = float(df["y_global_px"].min())
    y_max = float(df["y_global_px"].max())

    W_native = x_max - x_min
    H_native = y_max - y_min
    scale = min(max_pixels / W_native, max_pixels / H_native, 1.0)
    W = int(np.ceil(W_native * scale)) + 2
    H = int(np.ceil(H_native * scale)) + 2

    print(f"  Native extent: {W_native:.0f} × {H_native:.0f} px")
    print(f"  Canvas: {H} × {W} px  (scale = {scale:.5f})")
    print(f"  Est. base array: {H * W * 4 / 1e6:.0f} MB")

    # ── 4. Rasterise polygons ───────────────────────────────────────────────
    print("Rasterising polygons …")
    t1 = time.time()
    labels = np.zeros((H, W), dtype=np.int32)

    # Pre-compute scaled coordinates
    df["cell_idx"] = cell2idx[df["cell"].values].values
    df["r"] = ((df["y_global_px"] - y_min) * scale).astype(np.float32)
    df["c"] = ((df["x_global_px"] - x_min) * scale).astype(np.float32)

    grouped = df.groupby("cell_idx", sort=False)
    n_groups = len(grouped)
    for i, (cell_idx, grp) in enumerate(grouped):
        if i % 20_000 == 0:
            print(f"  {i:,}/{n_groups:,}", end="\r", flush=True)
        rr, cc = draw_polygon(
            grp["r"].to_numpy(dtype=np.float64),
            grp["c"].to_numpy(dtype=np.float64),
            shape=(H, W),
        )
        if rr.size:
            labels[rr, cc] = cell_idx

    print(f"\n  Rasterised {n_groups:,} cells in {time.time() - t1:.1f}s")

    # ── 5. Write multi-scale zarr ───────────────────────────────────────────
    print("Writing zarr pyramid …")
    store = zarr.open_group(str(zarr_path), mode="w")

    level_arr = labels
    for level in range(n_levels):
        h, w = level_arr.shape
        store.create_array(
            str(level),
            data=level_arr,
            chunks=(chunk_size, chunk_size),
            overwrite=True,
        )
        print(f"  Level {level}: {h} × {w} px")
        # 2× nearest-neighbour downsample (correct for integer labels)
        level_arr = level_arr[::2, ::2]

    # Write minimal OME-Zarr multiscales metadata so napari can auto-detect levels
    store.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}],
            "datasets": [
                {"path": str(i), "coordinateTransformations": [
                    {"type": "scale", "scale": [2.0 ** i, 2.0 ** i]}
                ]}
                for i in range(n_levels)
            ],
        }
    ]

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s  →  {zarr_path}")
