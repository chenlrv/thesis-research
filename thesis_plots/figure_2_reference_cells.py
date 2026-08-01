"""Thesis Figure 2 — Reference cells used for supervised refinement.

For each slide (L321, L34), shows the two reference classes that train the
supervised tumor classifier:

  * Tumor anchors (red) — high-confidence "Tumor" calls from the primary
    tumor slice (score_tumor >= 0.4, margin > 0.08). These are the positive
    class.

  * Look-alikes (blue) — cells called "Tumor" by SingleR in the control
    slice (score_tumor > 0.2, margin > 0.08). Because the control slice is
    sham-injected and contains no metastatic cells, any "Tumor" call here
    is by experimental design a false positive — a clean negative class
    capturing the failure modes of the reference-based step.

This figure is the methodological-contribution figure: it shows the
"experimental-design-as-ground-truth" idea visually.

Saves: thesis_plots/figure_2_reference_cells.png  (300 dpi)
"""
import pathlib

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import (
    _get_healthy_ref_ids,
    _get_tumor_ref_ids,
)
from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
OUT_DIR = BASE_DIR / "thesis_plots"

# Per-slide source slices
SLIDE_SOURCES = {
    "L321": {"tumor_source": 1, "control_source": 3},
    "L34":  {"tumor_source": 5, "control_source": 4},
}

# Colours
TUMOR_RED = "#D62728"
LOOKALIKE_BLUE = "#1565C0"
BG_GRAY = "#D9D9D9"


def _load_slide(slide_id: str):
    """Load the per-slide AnnData in backed mode (no expression matrix read)."""
    path = SLIDE_CACHE / f"sample_{slide_id}_adata.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    x = obs[CENTER_X_GLOBAL_PX].to_numpy()
    y = obs[CENTER_Y_GLOBAL_PX].to_numpy()
    obs_names = np.array(obs.index)
    n_total = len(obs_names)
    adata.file.close()
    return x, y, obs_names, n_total


def make_figure(out_path: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    fig.suptitle(
        "Reference cells for supervised refinement of tumor calls",
        fontsize=16, fontweight="bold", y=0.99,
    )

    summary = []
    for ax, slide_id in zip(axes, ["L321", "L34"]):
        sources = SLIDE_SOURCES[slide_id]
        x, y, obs_names, n_total = _load_slide(slide_id)

        tumor_ids = _get_tumor_ref_ids(slide_id)
        lookalike_ids = _get_healthy_ref_ids(slide_id)

        is_tumor_ref = np.isin(obs_names, np.array(list(tumor_ids)))
        is_lookalike = np.isin(obs_names, np.array(list(lookalike_ids)))
        is_other = ~is_tumor_ref & ~is_lookalike

        n_tumor = int(is_tumor_ref.sum())
        n_look = int(is_lookalike.sum())

        # Background — all non-reference cells in the slide
        ax.scatter(
            x[is_other], y[is_other],
            c=BG_GRAY, s=0.5, alpha=0.4, linewidths=0, rasterized=True,
        )
        # Tumor anchors (positive class) — drawn before look-alikes since they're more numerous
        ax.scatter(
            x[is_tumor_ref], y[is_tumor_ref],
            c=TUMOR_RED, s=2.0, alpha=0.95, linewidths=0, rasterized=True,
        )
        # Look-alikes (negative class) — drawn on top to emphasise the methodological idea
        ax.scatter(
            x[is_lookalike], y[is_lookalike],
            c=LOOKALIKE_BLUE, s=2.2, alpha=0.95, linewidths=0, rasterized=True,
        )

        ax.set_title(
            f"Slide {slide_id}\n"
            f"{n_tumor:,} tumor anchors (from slice {sources['tumor_source']}) + "
            f"{n_look:,} look-alikes (from slice {sources['control_source']})",
            fontsize=12, fontweight="semibold",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        summary.append({
            "slide": slide_id,
            "tumor_source_slice": sources["tumor_source"],
            "control_source_slice": sources["control_source"],
            "n_tumor_anchors": n_tumor,
            "n_lookalikes": n_look,
            "n_total_cells": n_total,
        })

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=TUMOR_RED, markeredgecolor="none",
                   label="High-confidence tumor anchor (positive class)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=LOOKALIKE_BLUE, markeredgecolor="none",
                   label="Look-alike: SingleR-called tumor in control slice (negative class)"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=BG_GRAY, markeredgecolor="none",
                   label="Other cells"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"saved: {out_path}\n")

    df = pd.DataFrame(summary).set_index("slide")
    print("=== Reference cell counts ===")
    print(df.to_string())
    csv_out = out_path.with_suffix(".csv")
    df.to_csv(csv_out)
    print(f"\nsummary table: {csv_out}")

    plt.show()


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    make_figure(OUT_DIR / "figure_2_reference_cells.png")
