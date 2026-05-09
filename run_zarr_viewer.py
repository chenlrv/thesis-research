"""
High-resolution zoomable cell viewer backed by a pre-computed zarr pyramid.

Usage
-----
Step 1 — prepare (run once per tissue slice, ~5-15 min):
    python run_zarr_viewer.py --prepare

Step 2 — open the viewer (instant each time after prepare):
    python run_zarr_viewer.py

Configure which sample and slice to view with SAMPLE and SLICE_ID below.

Available slices
----------------
L321:  slice 1 (tumor, mouse 2)  |  slice 2 (tumor, mouse 3)  |  slice 3 (healthy, mouse 1)
L34:   slice 4 (healthy, mouse 1) | slice 5 (tumor, mouse 3)   | slice 6 (tumor, mouse 2)
"""

from pathlib import Path
import argparse
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE   = "L321"   # "L321" or "L34"
SLICE_ID = 1        # tissue slice number (see header above for valid IDs per sample)

ADATA     = r"D:\thesis-research\resources\clustering\adata_full_with_pca_clustered.h5ad"
COLOR_COL = "cluster"   # column in adata.obs to color cells by

# ── Tumor viewer config (used with --tumor) ────────────────────────────────
TUMOR_DIR = Path(r"D:\thesis-research\resources\cache\with_tumor_prediction")
TUMOR_COL = "pred_tumor_XGBoost"

# Raster resolution: longest canvas side in pixels.
# Higher = more detail at max zoom, more RAM during --prepare (4 bytes × H × W).
#   10 000 px → ~400 MB  |  15 000 px → ~900 MB
MAX_PIXELS = 25_000

# ── Resource paths (edit only if your layout differs) ─────────────────────────
COSMX_DIR  = Path(r"D:\thesis-research\resources\cosmx")
VIEWER_DIR = Path(r"D:\thesis-research\resources\viewer")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fov_slices_path(sample: str) -> Path:
    return COSMX_DIR / sample / f"{sample}_fov_slices.csv"


def _polygons_path(sample: str) -> Path:
    return COSMX_DIR / sample / f"{sample}-polygons.csv"


def _slice_out_dir(sample: str, slice_id: int) -> Path:
    return VIEWER_DIR / sample / f"slice_{slice_id}"


def _load_fovs_for_slice(sample: str, slice_id: int) -> list[int]:
    """Read {sample}_fov_slices.csv and return all FOV IDs for the given slice."""
    csv_path = _fov_slices_path(sample)
    if not csv_path.exists():
        raise FileNotFoundError(f"FOV-slice mapping not found: {csv_path}")
    df = pd.read_csv(csv_path)
    rows = df[df["slice"] == slice_id]
    if rows.empty:
        raise ValueError(
            f"Slice {slice_id} not found in {csv_path}. "
            f"Available slices: {sorted(df['slice'].unique().tolist())}"
        )
    fovs: list[int] = []
    for _, row in rows.iterrows():
        fovs.extend(range(int(row["start"]), int(row["end"]) + 1))
    return fovs


# ── Main actions ───────────────────────────────────────────────────────────────

def prepare(sample: str, slice_id: int) -> None:
    from thesis_research.viewer_tool.prepare_zarr import prepare_viewer_zarr

    polygons = _polygons_path(sample)
    if not polygons.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygons}")

    fovs = _load_fovs_for_slice(sample, slice_id)
    print(f"Slice {slice_id} of {sample}: {len(fovs)} FOVs "
          f"({fovs[0]}–{fovs[-1]} and possibly non-contiguous)")

    out_dir = _slice_out_dir(sample, slice_id)
    prepare_viewer_zarr(
        polygons_path=polygons,
        out_dir=out_dir,
        fovs=fovs,
        max_pixels=MAX_PIXELS,
    )


def view_tumor(sample: str, slice_id: int, borders: bool = True) -> None:
    import anndata as ad
    from thesis_research.viewer_tool.zarr_viewer import show_zarr_binary

    out_dir  = _slice_out_dir(sample, slice_id)
    zarr_p   = out_dir / "labels.zarr"
    index_p  = out_dir / "cell_index.parquet"

    if not zarr_p.exists():
        print(
            f"Zarr store not found: {zarr_p}\n"
            "Run --prepare first:  python run_zarr_viewer.py --prepare"
        )
        return

    tumor_adata_p = TUMOR_DIR / f"slice_{slice_id}_adata.h5ad"
    if not tumor_adata_p.exists():
        print(f"Tumor prediction adata not found: {tumor_adata_p}")
        return

    print(f"Loading tumor predictions …")
    adata = ad.read_h5ad(tumor_adata_p)

    show_zarr_binary(
        zarr_path=zarr_p,
        cell_index_path=index_p,
        adata=adata,
        bool_col=TUMOR_COL,
        borders=borders,
    )


def view(sample: str, slice_id: int) -> None:
    import anndata as ad
    from thesis_research.viewer_tool.zarr_viewer import show_zarr_slice

    out_dir  = _slice_out_dir(sample, slice_id)
    zarr_p   = out_dir / "labels.zarr"
    index_p  = out_dir / "cell_index.parquet"

    if not zarr_p.exists():
        print(
            f"Zarr store not found: {zarr_p}\n"
            "Run --prepare first:  python run_zarr_viewer.py --prepare"
        )
        return

    print(f"Loading adata …")
    adata = ad.read_h5ad(ADATA)

    show_zarr_slice(
        zarr_path=zarr_p,
        cell_index_path=index_p,
        adata=adata,
        color_col=COLOR_COL,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Rasterise polygon CSV for the selected slice into a zarr pyramid.",
    )
    parser.add_argument(
        "--tumor",
        action="store_true",
        help="Open tumor viewer: red = XGBoost tumor cells, gray = everything else.",
    )
    parser.add_argument(
        "--no-borders",
        action="store_true",
        help="Hide cell borders in the tumor viewer (faster, cleaner at low zoom).",
    )
    parser.add_argument(
        "--sample", default=SAMPLE,
        help=f"Sample ID (default: {SAMPLE})",
    )
    parser.add_argument(
        "--slice", type=int, default=SLICE_ID, dest="slice_id",
        help=f"Tissue slice number (default: {SLICE_ID})",
    )
    args = parser.parse_args()

    if args.prepare:
        prepare(args.sample, args.slice_id)
    elif args.tumor:
        view_tumor(args.sample, args.slice_id, borders=not args.no_borders)
    else:
        view(args.sample, args.slice_id)
