"""Thesis Figure 4 — Spatial refinement of tumor candidates across classifiers.

For each of the six slices (4 tumor, 2 control), shows the SingleR candidate
pool refined by each of the five classifiers from Figure 3. Each panel:

  * Red    : candidate retained as tumor by this classifier
  * Blue   : candidate rejected as look-alike by this classifier
  * Gray   : other cells (not in the SingleR candidate pool)

Panel borders are colour-coded by slice type (red = tumor-bearing, blue =
sham-injected control), matching Figure 1.

The visual story has two parts:
  1. In tumor slices, the linear models reject many cells inside the dense
     tumor mass; XGBoost preserves the mass and rejects mainly peripheral
     scatter — the cells whose SingleR call was always weakest.
  2. In control slices, all classifiers should reject the small candidate
     set (those are the look-alikes the pipeline was designed to remove).
     Differences here probe how aggressive each classifier is at the
     candidate-rejection task.

Classifiers are trained jointly on the L321+L34 reference pool and applied
to the candidate cells in each slice (cells that were never in training).
The reference-pool CV metrics (Figure 3) are nearly identical across
classifiers; the differences shown here are the operationally meaningful
divergence that drives model selection.

Output (300 dpi):
  * figure_4_spatial_refinement_slice{1..6}.png — one full-page figure per
    slice: three models on the top row, the remaining two plus the legend on
    the bottom row. These are the figures that go in the thesis; a single 6x5
    grid makes the individual slices too small to read.
  * figure_4_spatial_refinement.png — the compact 6x5 overview, kept for
    reference / supplementary use.
  * figure_4_spatial_refinement.csv — per-slice, per-model retention counts.

Caption template for the per-slice figures (rephrased so the title states the
subject and the opening sentence states what is drawn, rather than restating
the title):

  Figure 4.N | Classifier decisions on the SingleR tumor candidates of
  slice N (L321, tumor-bearing).
  Each panel maps every segmented cell of the slice in global tissue
  coordinates, with one panel per evaluated model. Red marks a SingleR tumor
  candidate that the model retained as tumor, blue a candidate it rejected as
  a look-alike, and gray the remaining cells, which were never in the
  candidate pool. The inset gives the retained and rejected counts and the
  retention rate; the panel border is red for tumor-bearing and blue for
  sham-injected control slices.
"""
import pathlib

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import (
    _get_healthy_ref_ids,
    _get_tumor_candidates_ids,
    _get_tumor_ref_ids,
)
from thesis_research.utils.columns import CENTER_X_GLOBAL_PX, CENTER_Y_GLOBAL_PX

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
OUT_DIR = BASE_DIR / "thesis_plots"

PROB_THRESH = 0.5
N_PCS = 50
KNN_K = 15
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

MODEL_ORDER = ["LogReg", "LogReg + PCA", "LogReg + KNN (PCA)", "Random Forest", "XGBoost"]
MODEL_COLORS = {
    "LogReg":              "#4C72B0",
    "LogReg + PCA":        "#55A868",
    "LogReg + KNN (PCA)":  "#C44E52",
    "Random Forest":       "#DD8452",
    "XGBoost":             "#8172B2",
}

KEPT_RED = "#D62728"
REJECT_BLUE = "#1565C0"
BG_GRAY = "#D9D9D9"
TUMOR_BORDER = "#B71C1C"
CONTROL_BORDER = "#1565C0"

# Per-slice figures: the five models fill a 2x3 grid row-major, so the top row
# carries three models and the bottom row the remaining two plus the legend in
# the sixth cell. Panels are ~2.2 in wide at a 6.5 in text width, against
# ~1.3 in in the 6x5 overview grid.
PER_SLICE_GRID = (2, 3)
PANEL_WIDTH_IN = 5.0
# Panel height follows the tissue aspect ratio so the slice fills its axes
# instead of floating in whitespace; clamped so no slice makes a runaway page.
PANEL_HEIGHT_RANGE = (2.2, 5.0)


def _load_candidate_ids(slice_id):
    """Load candidate IDs for any slice (not just primary tumor slices)."""
    slide_id, _ = SLICES_TO_SHOW[slice_id]
    csv_path = (
        BASE_DIR / "outputs" / "cell_annotation" / slide_id / "05" / str(slice_id)
        / f"slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    annot_df = pd.read_csv(csv_path)
    return _get_tumor_candidates_ids(slide_id, annot_df)


def _style_panel_border(ax, slice_type):
    color = TUMOR_BORDER if slice_type == "Tumor" else CONTROL_BORDER
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)


def _to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def build_reference_pool():
    slide_ids = ["L321", "L34"]
    healthy_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ids = {sid: _get_tumor_ref_ids(sid) for sid in slide_ids}
    all_healthy = set().union(*healthy_ids.values())
    all_tumor = set().union(*tumor_ids.values())
    all_ref = all_healthy | all_tumor

    # Subset each slide to its reference cells before normalizing and before
    # concatenating. Normalizing all 846k cells needs ~600 MB of temporaries per
    # slide and exhausts a 16 GB machine; normalize_total scales each cell by its
    # own total, so restricting to the reference cells first is numerically
    # identical.
    adatas = {}
    for sid in slide_ids:
        path = SLIDE_CACHE / f"sample_{sid}_adata.h5ad"
        adata = ad.read_h5ad(path)
        adata_ref = adata[np.asarray(adata.obs_names.isin(all_ref))].copy()
        del adata
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        adatas[sid] = adata_ref

    adata_joint = ad.concat(adatas, join="inner", label="slide_id")
    del adatas
    healthy_mask = adata_joint.obs_names.isin(all_healthy)
    tumor_mask = adata_joint.obs_names.isin(all_tumor)
    ref_mask = np.asarray(healthy_mask | tumor_mask)

    X_ref = _to_dense(adata_joint.X[ref_mask])
    # Tumor is the positive class, matching figure_3_model_comparison: the models
    # score P(tumor) and a candidate is retained when P(tumor) > 0.5.
    y_ref = np.asarray(tumor_mask[ref_mask]).astype(int)
    joint_var_names = adata_joint.var_names.tolist()
    print(f"reference pool: {len(y_ref):,} cells "
          f"({int((y_ref == 1).sum()):,} tumor / {int((y_ref == 0).sum()):,} healthy)")
    return X_ref, y_ref, joint_var_names


def fit_classifiers(X_ref, y_ref):
    scaler_knn = StandardScaler()
    X_scaled = np.clip(scaler_knn.fit_transform(X_ref), -10, 10)
    n_comps_eff = min(N_PCS, X_ref.shape[0] - 1, X_ref.shape[1] - 1)
    pca_knn = PCA(n_components=n_comps_eff, random_state=RANDOM_STATE)
    X_pca = pca_knn.fit_transform(X_scaled)

    # Library defaults except max_iter (convergence), 300 trees, and the
    # 50 components / 15 neighbours standard in single-cell analysis. Class
    # balancing is deliberately not applied -- see figure_3_model_comparison.py.
    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    logreg_pca_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCS, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    # xgboost 3.2.0 library defaults -- see xgb_default_sensitivity.py.
    xgb_model = XGBClassifier(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    rf_model = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
    )
    logreg_knn_clf = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)

    print("\nFitting classifiers on the joint reference pool...")
    logreg_pipe.fit(X_ref, y_ref);             print("  LogReg fit")
    logreg_pca_pipe.fit(X_ref, y_ref);         print("  LogReg + PCA fit")
    rf_model.fit(X_ref, y_ref);                print("  Random Forest fit")
    xgb_model.fit(X_ref, y_ref);               print("  XGBoost fit")
    logreg_knn_clf.fit(X_pca, y_ref);          print("  LogReg + KNN (PCA) fit")
    nn_full = NearestNeighbors(n_neighbors=min(KNN_K, X_pca.shape[0]))
    nn_full.fit(X_pca)

    return {
        "logreg_pipe":     logreg_pipe,
        "logreg_pca_pipe": logreg_pca_pipe,
        "rf_model":        rf_model,
        "xgb_model":       xgb_model,
        "logreg_knn_clf":  logreg_knn_clf,
        "nn_full":         nn_full,
        "scaler_knn":      scaler_knn,
        "pca_knn":         pca_knn,
        "y_ref":           y_ref,
    }


def score_candidates(X_cand, art):
    """Returns dict of model_name -> is_tumor (bool array, same length as X_cand)."""
    prob_lr     = art["logreg_pipe"].predict_proba(X_cand)[:, 1]
    prob_lr_pca = art["logreg_pca_pipe"].predict_proba(X_cand)[:, 1]
    prob_rf     = art["rf_model"].predict_proba(X_cand)[:, 1]
    prob_xgb    = art["xgb_model"].predict_proba(X_cand)[:, 1]

    X_cand_pca = art["pca_knn"].transform(
        np.clip(art["scaler_knn"].transform(X_cand), -10, 10)
    )
    prob_lr_knn = art["logreg_knn_clf"].predict_proba(X_cand_pca)[:, 1]
    knn_frac = art["y_ref"][
        art["nn_full"].kneighbors(X_cand_pca, return_distance=False)
    ].mean(axis=1)

    # Probabilities are now P(tumor), so every rule inverts. The KNN rule was
    # "healthy if either signal says healthy"; its negation is "tumor only if
    # both signals say tumor", so the OR becomes an AND. This is the same
    # decision boundary as before, written from the tumor side.
    return {
        "LogReg":             prob_lr     > PROB_THRESH,
        "LogReg + PCA":       prob_lr_pca > PROB_THRESH,
        "LogReg + KNN (PCA)": (prob_lr_knn > PROB_THRESH) & (knn_frac > PROB_THRESH),
        "Random Forest":      prob_rf     > PROB_THRESH,
        "XGBoost":            prob_xgb    > PROB_THRESH,
    }


def process_slice(slice_id, joint_var_names, artifacts):
    slide_id, _ = SLICES_TO_SHOW[slice_id]
    path = SLIDE_CACHE / f"slice_{slice_id}_adata.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    slice_adata = ad.read_h5ad(path)

    # Coordinates are needed for every cell (the panel background); expression is
    # needed only for the candidates. Subsetting rows before normalizing keeps
    # peak memory down and is numerically identical, since normalize_total scales
    # each cell by its own total. Gene subsetting still happens *after*
    # normalization, as the per-cell totals are defined over the full panel.
    x = slice_adata.obs[CENTER_X_GLOBAL_PX].to_numpy()
    y = slice_adata.obs[CENTER_Y_GLOBAL_PX].to_numpy()
    candidate_ids = _load_candidate_ids(slice_id)
    is_cand = np.asarray(slice_adata.obs_names.isin(candidate_ids))
    n_obs_total = slice_adata.n_obs

    if is_cand.sum() == 0:
        empty = {m: np.zeros(n_obs_total, dtype=bool) for m in MODEL_ORDER}
        return x, y, is_cand, empty

    cand_adata = slice_adata[is_cand].copy()
    del slice_adata
    sc.pp.normalize_total(cand_adata, target_sum=1e4)
    sc.pp.log1p(cand_adata)
    cand_adata = cand_adata[:, joint_var_names].copy()

    X_cand = _to_dense(cand_adata.X)
    raw_results = score_candidates(X_cand, artifacts)

    cand_idx = np.where(is_cand)[0]
    expanded = {}
    for name, is_tumor_cand in raw_results.items():
        full = np.zeros(n_obs_total, dtype=bool)
        full[cand_idx] = is_tumor_cand
        expanded[name] = full
    return x, y, is_cand, expanded


def _legend_handles(marker_scale=1.0):
    return [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12 * marker_scale,
                   markerfacecolor="white", markeredgecolor=TUMOR_BORDER,
                   markeredgewidth=2.0, label="Tumor-bearing slice"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12 * marker_scale,
                   markerfacecolor="white", markeredgecolor=CONTROL_BORDER,
                   markeredgewidth=2.0, label="Sham-injected control slice"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10 * marker_scale,
                   markerfacecolor=KEPT_RED, markeredgecolor="none",
                   label="Candidate retained as tumor"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10 * marker_scale,
                   markerfacecolor=REJECT_BLUE, markeredgecolor="none",
                   label="Candidate rejected as look-alike"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10 * marker_scale,
                   markerfacecolor=BG_GRAY, markeredgecolor="none",
                   label="Other cells (not in candidate pool)"),
    ]


def _draw_panel(ax, x, y, is_cand, is_tumor, slice_type, n_cand, sizes, stat_fontsize):
    """Draw one classifier panel; returns (n_kept, n_rejected, pct_retained)."""
    is_reject = is_cand & ~is_tumor
    n_kept = int(is_tumor.sum())
    n_rej = int(is_reject.sum())
    pct_kept = 100 * n_kept / n_cand if n_cand else 0.0

    s_bg, s_rej, s_kept = sizes
    ax.scatter(
        x[~is_cand], y[~is_cand],
        c=BG_GRAY, s=s_bg, alpha=0.35, linewidths=0, rasterized=True,
    )
    ax.scatter(
        x[is_reject], y[is_reject],
        c=REJECT_BLUE, s=s_rej, alpha=0.85, linewidths=0, rasterized=True,
    )
    ax.scatter(
        x[is_tumor], y[is_tumor],
        c=KEPT_RED, s=s_kept, alpha=0.95, linewidths=0, rasterized=True,
    )

    ax.text(
        0.02, 0.98,
        f"{n_kept:,} kept  /  {n_rej:,} rejected\n({pct_kept:.1f}% retained)",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=stat_fontsize,
        bbox=dict(facecolor="white", edgecolor="#BDBDBD", alpha=0.85,
                  pad=2.5, boxstyle="round,pad=0.3"),
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    _style_panel_border(ax, slice_type)
    return n_kept, n_rej, pct_kept


def make_slice_figure(slice_id, panels, out_path):
    """One full-page figure for a single slice: 5 classifier panels + legend cell."""
    slide_id, slice_type = SLICES_TO_SHOW[slice_id]
    x, y, is_cand, results = panels
    n_cand = int(is_cand.sum())

    x_span = float(np.ptp(x)) or 1.0
    y_span = float(np.ptp(y)) or 1.0
    panel_h = float(np.clip(PANEL_WIDTH_IN * y_span / x_span, *PANEL_HEIGHT_RANGE))

    n_rows, n_cols = PER_SLICE_GRID
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(PANEL_WIDTH_IN * n_cols, panel_h * n_rows),
    )
    # No figure title: the caption carries it in the thesis.

    for idx, mname in enumerate(MODEL_ORDER):
        ax = axes.flat[idx]
        _draw_panel(
            ax, x, y, is_cand, results[mname], slice_type, n_cand,
            sizes=(1.2, 5.0, 5.5), stat_fontsize=12,
        )
        ax.set_title(mname, fontsize=17, fontweight="semibold",
                     color=MODEL_COLORS[mname])

    # Sixth cell: the legend, captioned with the slice identity. The identity is
    # the legend *title* rather than a free-floating text box so matplotlib
    # stacks the two; on the flat slices a fixed y-position collides with the
    # entries. Font size follows the panel height for the same reason.
    legend_ax = axes.flat[len(MODEL_ORDER)]
    legend_ax.axis("off")
    slice_color = TUMOR_BORDER if slice_type == "Tumor" else CONTROL_BORDER
    legend_fs = float(np.clip(3.8 * panel_h, 9.5, 15.0))
    leg = legend_ax.legend(
        handles=_legend_handles(marker_scale=legend_fs / 10.5), loc="center",
        title=f"Slice {slice_id}\n{slide_id} ({slice_type})",
        frameon=False, fontsize=legend_fs,
        labelspacing=0.9, handletextpad=1.0,
    )
    leg.get_title().set_fontsize(legend_fs * 1.4)
    leg.get_title().set_fontweight("semibold")
    leg.get_title().set_color(slice_color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"  saved: {out_path}")


def make_overview_figure(all_panels, out_path):
    """The compact 6x5 grid — kept for reference alongside the per-slice figures."""
    slice_ids = list(SLICES_TO_SHOW.keys())
    n_rows, n_cols = len(slice_ids), len(MODEL_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    # No figure title: the caption carries it in the thesis.

    for row, slice_id in enumerate(slice_ids):
        slide_id, slice_type = SLICES_TO_SHOW[slice_id]
        x, y, is_cand, results = all_panels[slice_id]
        n_cand = int(is_cand.sum())

        for col, mname in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            _draw_panel(
                ax, x, y, is_cand, results[mname], slice_type, n_cand,
                sizes=(0.3, 1.1, 1.2), stat_fontsize=8.5,
            )
            if row == 0:
                ax.set_title(mname, fontsize=12.5, fontweight="semibold",
                             color=MODEL_COLORS[mname])
            # Row label on the leftmost column
            if col == 0:
                ax.set_ylabel(
                    f"Slice {slice_id}\n{slide_id} ({slice_type})",
                    fontsize=11, fontweight="semibold", rotation=0,
                    ha="right", va="center", labelpad=24,
                    color=TUMOR_BORDER if slice_type == "Tumor" else CONTROL_BORDER,
                )

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=5,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.003))

    plt.tight_layout(rect=[0.02, 0.025, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {out_path}")


def make_figure(out_path):
    X_ref, y_ref, joint_var_names = build_reference_pool()
    artifacts = fit_classifiers(X_ref, y_ref)

    all_panels = {}
    summary = []
    for slice_id in SLICES_TO_SHOW:
        slide_id, slice_type = SLICES_TO_SHOW[slice_id]
        print(f"\n--- Slice {slice_id} ({slide_id}, {slice_type}) ---")
        x, y, is_cand, results = process_slice(slice_id, joint_var_names, artifacts)
        n_cand = int(is_cand.sum())
        print(f"  {n_cand:,} candidates")
        all_panels[slice_id] = (x, y, is_cand, results)

        for mname in MODEL_ORDER:
            is_tumor = results[mname]
            n_kept = int(is_tumor.sum())
            n_rej = n_cand - n_kept
            summary.append({
                "slice": slice_id, "slide": slide_id, "type": slice_type,
                "model": mname,
                "n_candidates": n_cand,
                "n_retained_tumor": n_kept,
                "n_rejected_lookalike": n_rej,
                "pct_retained": round(100 * n_kept / n_cand if n_cand else 0.0, 2),
            })

        make_slice_figure(
            slice_id, all_panels[slice_id],
            out_path.with_name(f"{out_path.stem}_slice{slice_id}{out_path.suffix}"),
        )

    make_overview_figure(all_panels, out_path)

    df = pd.DataFrame(summary)
    csv_out = out_path.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"summary: {csv_out}")

    print("\n=== Mean retention by classifier — tumor slices ===")
    tumor_df = df[df["type"] == "Tumor"]
    by_model_T = (
        tumor_df.groupby("model")[["n_retained_tumor", "n_rejected_lookalike", "pct_retained"]]
        .mean()
        .round(1)
    )
    print(by_model_T.loc[MODEL_ORDER].to_string())

    print("\n=== Mean retention by classifier — control slices ===")
    ctrl_df = df[df["type"] == "Control"]
    by_model_C = (
        ctrl_df.groupby("model")[["n_retained_tumor", "n_rejected_lookalike", "pct_retained"]]
        .mean()
        .round(1)
    )
    print(by_model_C.loc[MODEL_ORDER].to_string())

    plt.show()


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    make_figure(OUT_DIR / "figure_4_spatial_refinement.png")
