"""Thesis Figure 1 — SingleR tumor calls across all six slices.

Reads the SingleR annotation CSVs and the per-slice AnnData files (in backed
mode — only obs is read), applies the three SingleR filters (score floor,
margin, best-class consistency), and plots the resulting tumor-candidate cells
spatially across all six slices in a 2x3 grid (L321 row on top, L34 row below).

Tumor candidates that appear in the control slices (3 and 4) and in clearly
non-tumor regions of the tumor slices are the false positives that motivate
the supervised refinement step (Stage 2 in the thesis text).

Saves: thesis_plots/figure_1_singler_calls.png  (300 dpi)
"""
import pathlib

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLICE_CACHE = BASE_DIR / "resources" / "cache"
ANNOT_DIR = BASE_DIR / "outputs" / "cell_annotation"
OUT_DIR = BASE_DIR / "thesis_plots"

SCORE_TUMOR_FLOOR = 0.2
DELTA_MARGIN = 0.08

# slice_id -> (slide_id, mouse_id, type)
SLICE_META = {
    1: ("L321", 2, "Tumor"),
    2: ("L321", 3, "Tumor"),
    3: ("L321", 1, "Control"),
    4: ("L34",  1, "Control"),
    5: ("L34",  3, "Tumor"),
    6: ("L34",  2, "Tumor"),
}

# Colours
TUMOR_RED = "#D62728"
BG_GRAY = "#D9D9D9"
TUMOR_BORDER = "#B71C1C"     # dark red for Tumor-slice panel border
CONTROL_BORDER = "#1565C0"   # dark blue for Control-slice panel border
TUMOR_TITLE = "#7F1D1D"
CONTROL_TITLE = "#0D3B66"

REQUIRED_CSV_COLS = (
    "cell_barcode", "predicted_cell_type",
    "score_tumor", "score_brain_struct", "score_brain_immune",
)


def _validate_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    return df


def _load_slice(slice_id: int):
    """Return (x, y, obs_names, n_total, is_candidate) for a slice.

    Reads the AnnData in backed mode so the gene expression matrix is never
    materialised — we only need coordinates from .obs. Applies the SingleR
    filter to produce the candidate mask aligned with the .obs index.
    """
    slide_id, _, _ = SLICE_META[slice_id]
    h5_path = SLICE_CACHE / f"slice_{slice_id}_adata.h5ad"
    csv_path = (
        ANNOT_DIR / slide_id / "05" / str(slice_id)
        / f"slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    adata = ad.read_h5ad(h5_path, backed="r")
    obs = adata.obs
    x = obs[CENTER_X_GLOBAL_PX].to_numpy()
    y = obs[CENTER_Y_GLOBAL_PX].to_numpy()
    obs_names = np.array(obs.index)
    n_total = len(obs_names)
    adata.file.close()

    df = _validate_csv(csv_path)
    df = df[df["predicted_cell_type"] == "Tumor"].copy()
    df["next_best_score"] = df[["score_brain_struct", "score_brain_immune"]].max(axis=1)
    df["delta_score"] = (df["score_tumor"] - df["next_best_score"]).abs()
    df = df[
        (df["score_tumor"] > SCORE_TUMOR_FLOOR)
        & (df["delta_score"] > DELTA_MARGIN)
        & (df["score_tumor"] > df["next_best_score"])
    ]
    candidate_ids = set(df["cell_barcode"].astype(str))
    is_candidate = np.isin(obs_names, np.array(list(candidate_ids)))
    return x, y, n_total, is_candidate


def _style_panel_border(ax, slice_type: str):
    color = TUMOR_BORDER if slice_type == "Tumor" else CONTROL_BORDER
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)


def make_figure(out_path: pathlib.Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Initial tumor candidates from reference-based annotation (SingleR)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    layout = {
        (0, 0): 1, (0, 1): 2, (0, 2): 3,
        (1, 0): 4, (1, 1): 5, (1, 2): 6,
    }
    summary = []
    for (row, col), slice_id in layout.items():
        slide_id, mouse_id, slice_type = SLICE_META[slice_id]
        ax = axes[row, col]

        x, y, n_total, is_cand = _load_slice(slice_id)
        n_cand = int(is_cand.sum())
        pct = 100 * n_cand / n_total if n_total else 0.0
        summary.append({
            "slice": slice_id, "slide": slide_id, "mouse": mouse_id,
            "type": slice_type, "n_total": n_total, "n_candidates": n_cand,
            "pct_candidates": round(pct, 2),
        })

        ax.scatter(
            x[~is_cand], y[~is_cand],
            c=BG_GRAY, s=0.6, alpha=0.45, linewidths=0, rasterized=True,
        )
        ax.scatter(
            x[is_cand], y[is_cand],
            c=TUMOR_RED, s=1.6, alpha=0.95, linewidths=0, rasterized=True,
        )

        title_color = TUMOR_TITLE if slice_type == "Tumor" else CONTROL_TITLE
        ax.set_title(
            f"Slice {slice_id} — {slide_id}, mouse {mouse_id} ({slice_type})\n"
            f"{n_cand:,} candidates / {n_total:,} cells  ({pct:.1f}%)",
            fontsize=11, color=title_color, fontweight="semibold",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        _style_panel_border(ax, slice_type)

    # Shared legend at the bottom — semantically clear ordering
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   markerfacecolor="white", markeredgecolor=TUMOR_BORDER,
                   markeredgewidth=2.5, label="Tumor-bearing slice"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   markerfacecolor="white", markeredgecolor=CONTROL_BORDER,
                   markeredgewidth=2.5, label="Sham-injected control slice"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=TUMOR_RED, markeredgecolor="none",
                   label="SingleR tumor candidate"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                   markerfacecolor=BG_GRAY, markeredgecolor="none",
                   label="other cells"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.035, 1, 0.955])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved: {out_path}\n")

    # Per-slice summary table for the caption / supplementary
    df = pd.DataFrame(summary).set_index("slice")
    print("=== Per-slice candidate counts ===")
    print(df.to_string())
    csv_out = out_path.with_suffix(".csv")
    df.to_csv(csv_out)
    print(f"\nsummary table: {csv_out}")

    plt.show()


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    make_figure(OUT_DIR / "figure_1_singler_calls.png")
