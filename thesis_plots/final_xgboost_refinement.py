"""Final XGBoost refinement of the SingleR tumor candidates — six-slice grid.

Redraws the six-slice overview of the final tumor calls as a portrait grid of
two slices per row (3 rows x 2 columns) instead of the compact 2x3 landscape
grid, so each slice is substantially larger and the individual cells stay
legible. The figure carries no title: the thesis caption states the subject.

Per panel:
  * Red   : SingleR tumor candidate retained by the final XGBoost model
  * Blue  : SingleR tumor candidate rejected as a look-alike host cell
  * Gray  : all other segmented cells (never in the candidate pool)

The panel border is red for tumor-bearing and blue for sham-injected control
slices, matching the other thesis figures.

Two sources of the tumor calls are available via --source:

  refit   (default) Refits the Stage-3 XGBoost on the joint L321+L34 reference
          pool using the *current* pipeline settings -- the xgboost 3.2.0
          library defaults now set in classifiers.py -- and scores the
          candidates of each slice. This is the up-to-date algorithm.
  cache   Reads the `pred_tumor_XGBoost` column cached in
          resources/cache/with_tumor_prediction/ (written 2026-05-21 with the
          older tuned hyperparameters: n_estimators=300, max_depth=4,
          learning_rate=0.05, subsample/colsample_bytree=0.8 and an empirical
          scale_pos_weight). Kept only for comparison with the earlier figure.

The candidate pool itself is identical under both, since the SingleR
score_tumor / delta_score thresholds have not changed.

Output (300 dpi): final_xgboost_refinement[_cached].png and the matching .csv
of per-slice retained/rejected counts.
"""
import argparse
import pathlib
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from xgboost import XGBClassifier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from figure_4_spatial_refinement import _to_dense, build_reference_pool  # noqa: E402
from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import (  # noqa: E402
    _get_tumor_candidates_ids,
)
from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX  # noqa: E402

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
PRED_CACHE = SLIDE_CACHE / "with_tumor_prediction"
OUT_DIR = BASE_DIR / "thesis_plots"

PRED_COL = "pred_tumor_XGBoost"
PROB_THRESH = 0.5
RANDOM_STATE = 42

# slice_id -> (slide_id, type)
SLICES_TO_SHOW = {
    1: ("L321", "Tumor"),
    2: ("L321", "Tumor"),
    3: ("L321", "Control"),
    4: ("L34",  "Control"),
    5: ("L34",  "Tumor"),
    6: ("L34",  "Tumor"),
}

KEPT_RED = "#D62728"
REJECT_BLUE = "#1565C0"
BG_GRAY = "#D9D9D9"
TUMOR_BORDER = "#B71C1C"
CONTROL_BORDER = "#1565C0"

# Panel geometry. Two slices per row; each slice keeps its true aspect ratio,
# and a row is made as tall as the taller of the two panels it holds.
FIG_WIDTH = 15.0
N_COLS = 2
PANEL_PAD = 0.7           # inches of horizontal margin taken by labels/spines
TITLE_HEIGHT = 0.55       # inches per row taken by the two-line panel title
LEGEND_HEIGHT = 0.7       # inches reserved at the bottom for the legend


def _load_candidate_ids(slice_id):
    slide_id, _ = SLICES_TO_SHOW[slice_id]
    csv_path = (
        BASE_DIR / "outputs" / "cell_annotation" / slide_id / "05" / str(slice_id)
        / f"slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return _get_tumor_candidates_ids(slide_id, pd.read_csv(csv_path))


def _style_panel_border(ax, slice_type):
    color = TUMOR_BORDER if slice_type == "Tumor" else CONTROL_BORDER
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)


def fit_current_xgboost():
    """Stage-3 XGBoost as configured in the current pipeline (library defaults)."""
    X_ref, y_ref, joint_var_names = build_reference_pool()
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_ref, y_ref)
    print("  XGBoost (xgboost library defaults) fit")
    return model, joint_var_names


def load_slice_cached(slice_id):
    """Coordinates and masks from the 2026-05-21 cached predictions."""
    path = PRED_CACHE / f"slice_{slice_id}_adata.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    # Only obs is needed, so the matrix is never pulled into memory.
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    x = obs[CENTER_X_GLOBAL_PX].to_numpy(dtype=float)
    y = obs[CENTER_Y_GLOBAL_PX].to_numpy(dtype=float)
    is_kept = obs[PRED_COL].to_numpy(dtype=bool)
    is_cand = np.asarray(obs.index.isin(_load_candidate_ids(slice_id)))
    adata.file.close()
    return x, y, is_kept, is_cand & ~is_kept


def load_slice_refit(slice_id, model, joint_var_names):
    """Coordinates and masks from scoring the candidates with the current model."""
    path = SLIDE_CACHE / f"slice_{slice_id}_adata.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    slice_adata = ad.read_h5ad(path)

    x = slice_adata.obs[CENTER_X_GLOBAL_PX].to_numpy(dtype=float)
    y = slice_adata.obs[CENTER_Y_GLOBAL_PX].to_numpy(dtype=float)
    is_cand = np.asarray(slice_adata.obs_names.isin(_load_candidate_ids(slice_id)))
    is_kept = np.zeros(slice_adata.n_obs, dtype=bool)

    if is_cand.sum():
        # Subsetting rows before normalizing is numerically identical
        # (normalize_total scales each cell by its own total) and keeps peak
        # memory down; gene subsetting follows normalization, as the per-cell
        # totals are defined over the full panel.
        cand_adata = slice_adata[is_cand].copy()
        del slice_adata
        sc.pp.normalize_total(cand_adata, target_sum=1e4)
        sc.pp.log1p(cand_adata)
        cand_adata = cand_adata[:, joint_var_names].copy()
        prob_tumor = model.predict_proba(_to_dense(cand_adata.X))[:, 1]
        is_kept[np.where(is_cand)[0]] = prob_tumor > PROB_THRESH

    return x, y, is_kept, is_cand & ~is_kept


def make_figure(out_path, source):
    slice_ids = list(SLICES_TO_SHOW.keys())

    model = joint_var_names = None
    if source == "refit":
        model, joint_var_names = fit_current_xgboost()

    panels, aspects = {}, []
    for slice_id in slice_ids:
        slide_id, slice_type = SLICES_TO_SHOW[slice_id]
        if source == "refit":
            x, y, is_kept, is_reject = load_slice_refit(slice_id, model, joint_var_names)
        else:
            x, y, is_kept, is_reject = load_slice_cached(slice_id)
        panels[slice_id] = (x, y, is_kept, is_reject)
        dx = np.ptp(x)
        dy = np.ptp(y)
        aspects.append(dy / dx if dx else 1.0)
        print(f"slice {slice_id} ({slide_id}, {slice_type}): "
              f"{int(is_kept.sum()):,} retained / {int(is_reject.sum()):,} rejected "
              f"of {len(x):,} cells")

    n_rows = int(np.ceil(len(slice_ids) / N_COLS))
    panel_width = (FIG_WIDTH - PANEL_PAD) / N_COLS
    # A row is sized by its tallest slice so neither panel is squashed.
    row_heights = [
        panel_width * max(aspects[r * N_COLS:(r + 1) * N_COLS])
        for r in range(n_rows)
    ]
    fig_height = sum(row_heights) + LEGEND_HEIGHT + TITLE_HEIGHT * n_rows

    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(FIG_WIDTH, fig_height),
        gridspec_kw={"height_ratios": row_heights},
    )
    # No figure title: the caption carries it in the thesis.

    summary = []
    for ax, slice_id in zip(np.asarray(axes).ravel(), slice_ids):
        slide_id, slice_type = SLICES_TO_SHOW[slice_id]
        x, y, is_kept, is_reject = panels[slice_id]
        is_other = ~(is_kept | is_reject)

        n_kept = int(is_kept.sum())
        n_rej = int(is_reject.sum())
        n_cand = n_kept + n_rej
        pct_kept = 100 * n_kept / n_cand if n_cand else 0.0

        ax.scatter(x[is_other], y[is_other],
                   c=BG_GRAY, s=1.0, alpha=0.5, linewidths=0, rasterized=True)
        ax.scatter(x[is_reject], y[is_reject],
                   c=REJECT_BLUE, s=2.4, alpha=0.85, linewidths=0, rasterized=True)
        ax.scatter(x[is_kept], y[is_kept],
                   c=KEPT_RED, s=2.6, alpha=0.95, linewidths=0, rasterized=True)

        ax.set_title(
            f"Slice {slice_id} — {slide_id} ({slice_type})\n"
            f"{n_kept:,} retained / {n_rej:,} rejected ({pct_kept:.1f}% retained)",
            fontsize=12, fontweight="semibold", color="#333333", pad=6,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        _style_panel_border(ax, slice_type)

        summary.append({
            "slice": slice_id, "slide": slide_id, "type": slice_type,
            "model": "XGBoost", "source": source,
            "n_candidates": n_cand,
            "n_retained_tumor": n_kept,
            "n_rejected_lookalike": n_rej,
            "pct_retained": round(pct_kept, 2),
        })

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=KEPT_RED, markeredgecolor="none",
                   label="Refined tumor cells"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=REJECT_BLUE, markeredgecolor="none",
                   label="Rejected look-alike candidates"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=BG_GRAY, markeredgecolor="none",
                   label="Other cells"),
    ]
    legend_frac = LEGEND_HEIGHT / fig_height
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.004))

    plt.tight_layout(rect=[0.01, legend_frac, 0.99, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {out_path}")

    df = pd.DataFrame(summary)
    csv_out = out_path.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"summary: {csv_out}")
    print(f"total retained tumor cells: {int(df['n_retained_tumor'].sum()):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("refit", "cache"), default="refit")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    name = ("final_xgboost_refinement.png" if args.source == "refit"
            else "final_xgboost_refinement_cached.png")
    make_figure(OUT_DIR / name, args.source)
