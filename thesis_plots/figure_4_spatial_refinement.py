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

Saves: thesis_plots/figure_4_spatial_refinement.png  (300 dpi)
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
    adatas = {}
    for sid in slide_ids:
        path = SLIDE_CACHE / f"sample_{sid}_adata.h5ad"
        adata = ad.read_h5ad(path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adatas[sid] = adata

    healthy_ids = {sid: _get_healthy_ref_ids(sid) for sid in slide_ids}
    tumor_ids = {sid: _get_tumor_ref_ids(sid) for sid in slide_ids}
    all_healthy = set().union(*healthy_ids.values())
    all_tumor = set().union(*tumor_ids.values())

    adata_joint = ad.concat(adatas, join="inner", label="slide_id")
    healthy_mask = adata_joint.obs_names.isin(all_healthy)
    tumor_mask = adata_joint.obs_names.isin(all_tumor)
    ref_mask = np.asarray(healthy_mask | tumor_mask)

    X_ref = _to_dense(adata_joint.X[ref_mask])
    y_ref = np.asarray(healthy_mask[ref_mask]).astype(int)
    joint_var_names = adata_joint.var_names.tolist()
    print(f"reference pool: {len(y_ref):,} cells "
          f"({int((y_ref == 1).sum()):,} healthy / {int((y_ref == 0).sum()):,} tumor)")
    return X_ref, y_ref, joint_var_names


def fit_classifiers(X_ref, y_ref):
    n_pos = int((y_ref == 1).sum())
    n_neg = int((y_ref == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    scaler_knn = StandardScaler()
    X_scaled = np.clip(scaler_knn.fit_transform(X_ref), -10, 10)
    n_comps_eff = min(N_PCS, X_ref.shape[0] - 1, X_ref.shape[1] - 1)
    pca_knn = PCA(n_components=n_comps_eff, random_state=RANDOM_STATE)
    X_pca = pca_knn.fit_transform(X_scaled)

    logreg_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    logreg_pca_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_PCS, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                   max_iter=5000, solver="liblinear", random_state=RANDOM_STATE)),
    ])
    xgb_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0, eval_metric="logloss",
    )
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    logreg_knn_clf = LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)

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

    return {
        "LogReg":             prob_lr     < PROB_THRESH,
        "LogReg + PCA":       prob_lr_pca < PROB_THRESH,
        "LogReg + KNN (PCA)": ~((prob_lr_knn >= PROB_THRESH) | (knn_frac >= PROB_THRESH)),
        "Random Forest":      prob_rf     < PROB_THRESH,
        "XGBoost":            prob_xgb    < PROB_THRESH,
    }


def process_slice(slice_id, joint_var_names, artifacts):
    slide_id, _ = SLICES_TO_SHOW[slice_id]
    path = SLIDE_CACHE / f"slice_{slice_id}_adata.h5ad"
    if not path.exists():
        raise FileNotFoundError(path)
    slice_adata = ad.read_h5ad(path)
    sc.pp.normalize_total(slice_adata, target_sum=1e4)
    sc.pp.log1p(slice_adata)
    slice_adata = slice_adata[:, joint_var_names].copy()

    candidate_ids = _load_candidate_ids(slice_id)
    is_cand = np.asarray(slice_adata.obs_names.isin(candidate_ids))
    x = slice_adata.obs[CENTER_X_GLOBAL_PX].to_numpy()
    y = slice_adata.obs[CENTER_Y_GLOBAL_PX].to_numpy()

    if is_cand.sum() == 0:
        empty = {m: np.zeros(slice_adata.n_obs, dtype=bool) for m in MODEL_ORDER}
        return x, y, is_cand, empty

    X_cand = _to_dense(slice_adata.X[is_cand])
    raw_results = score_candidates(X_cand, artifacts)

    cand_idx = np.where(is_cand)[0]
    expanded = {}
    for name, is_tumor_cand in raw_results.items():
        full = np.zeros(slice_adata.n_obs, dtype=bool)
        full[cand_idx] = is_tumor_cand
        expanded[name] = full
    return x, y, is_cand, expanded


def make_figure(out_path):
    X_ref, y_ref, joint_var_names = build_reference_pool()
    artifacts = fit_classifiers(X_ref, y_ref)

    slice_ids = list(SLICES_TO_SHOW.keys())
    n_rows, n_cols = len(slice_ids), len(MODEL_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.6 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Spatial refinement of SingleR tumor candidates across classifiers",
        fontsize=17, fontweight="bold", y=0.995,
    )

    summary = []
    for row, slice_id in enumerate(slice_ids):
        slide_id, slice_type = SLICES_TO_SHOW[slice_id]
        print(f"\n--- Slice {slice_id} ({slide_id}, {slice_type}) ---")
        x, y, is_cand, results = process_slice(slice_id, joint_var_names, artifacts)
        n_cand = int(is_cand.sum())
        print(f"  {n_cand:,} candidates")

        for col, mname in enumerate(MODEL_ORDER):
            ax = axes[row, col]
            is_tumor = results[mname]
            is_reject = is_cand & ~is_tumor

            n_kept = int(is_tumor.sum())
            n_rej = int(is_reject.sum())
            pct_kept = 100 * n_kept / n_cand if n_cand else 0.0

            ax.scatter(
                x[~is_cand], y[~is_cand],
                c=BG_GRAY, s=0.3, alpha=0.35, linewidths=0, rasterized=True,
            )
            ax.scatter(
                x[is_reject], y[is_reject],
                c=REJECT_BLUE, s=1.1, alpha=0.85, linewidths=0, rasterized=True,
            )
            ax.scatter(
                x[is_tumor], y[is_tumor],
                c=KEPT_RED, s=1.2, alpha=0.95, linewidths=0, rasterized=True,
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

            ax.text(
                0.02, 0.98,
                f"{n_kept:,} kept  /  {n_rej:,} rejected\n({pct_kept:.1f}% retained)",
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="#BDBDBD", alpha=0.85,
                          pad=2.5, boxstyle="round,pad=0.3"),
            )

            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            _style_panel_border(ax, slice_type)

            summary.append({
                "slice": slice_id, "slide": slide_id, "type": slice_type,
                "model": mname,
                "n_candidates": n_cand,
                "n_retained_tumor": n_kept,
                "n_rejected_lookalike": n_rej,
                "pct_retained": round(pct_kept, 2),
            })

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   markerfacecolor="white", markeredgecolor=TUMOR_BORDER,
                   markeredgewidth=2.0, label="Tumor-bearing slice"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=12,
                   markerfacecolor="white", markeredgecolor=CONTROL_BORDER,
                   markeredgewidth=2.0, label="Sham-injected control slice"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=KEPT_RED, markeredgecolor="none",
                   label="Candidate retained as tumor"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=REJECT_BLUE, markeredgecolor="none",
                   label="Candidate rejected as look-alike"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=BG_GRAY, markeredgecolor="none",
                   label="Other cells (not in candidate pool)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.003))

    plt.tight_layout(rect=[0.02, 0.025, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {out_path}")

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
