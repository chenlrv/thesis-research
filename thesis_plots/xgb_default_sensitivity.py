"""XGBoost hyperparameter sensitivity: library defaults vs the thesis settings.

Answers two questions raised in review:

  1. Were the non-default XGBoost hyperparameters chosen by cross-validation?
     (They were not -- they were fixed by hand. This script supplies the
     missing evidence.)

  2. Does reverting to the library defaults change the result?

Design
------
The Stage-2/3 reference pool is built exactly as in figure_3_model_comparison
and figure_4_spatial_refinement: positive class (y = 1) is 'healthy /
look-alike' -- cells called Tumor by SingleR in the two *control* sections,
which are false positives by construction; negative class (y = 0) is the
confident tumor anchor set from the tumor sections.

Configurations compared, all under the same 5-fold stratified CV (seed 42):

  current            the settings reported in the thesis
  all_defaults       XGBClassifier() library defaults, incl. scale_pos_weight = 1
  defaults_spw       library defaults but keeping the class-imbalance weight
  default_<param>    the thesis settings with ONE parameter reverted to its
                     library default (leave-one-out ablation)

For each configuration we report out-of-fold discrimination on the reference
pool AND the downstream consequence: how many SingleR tumor candidates survive
the P(healthy) < 0.5 filter in each of the six sections, and how far those
calls differ from the calls the thesis currently reports.

Outputs (thesis_plots/xgb_hparam_sensitivity/):
    cv_metrics.csv        out-of-fold CV metrics per configuration
    slice_calls.csv       refined tumor calls per configuration per slice
    agreement.csv         per-cell agreement with the current configuration
"""
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from thesis_research.pipeline.cell_type_annotation.tumor_cells.identify_tumor_cells import (
    _get_healthy_ref_ids,
    _get_tumor_ref_ids,
    _get_tumor_candidates_ids,
)

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
OUT_DIR = BASE_DIR / "thesis_plots" / "xgb_hparam_sensitivity"

PROB_THRESH = 0.5
RANDOM_STATE = 42
N_SPLITS = 5
# xgboost + OpenMP has deadlocked on this machine with n_jobs = -1; keep it bounded.
N_JOBS = 4

SLICES = {1: ("L321", "Tumor"), 2: ("L321", "Tumor"), 3: ("L321", "Control"),
          4: ("L34", "Control"), 5: ("L34", "Tumor"), 6: ("L34", "Tumor")}

# Settings reported in the thesis, and the xgboost library defaults they moved from.
CURRENT = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
DEFAULTS = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0)


def build_configs(scale_pos_weight):
    """name -> kwargs for XGBClassifier."""
    cfgs = {
        "current": {**CURRENT, "scale_pos_weight": scale_pos_weight},
        "all_defaults": {**DEFAULTS, "scale_pos_weight": 1.0},
        "defaults_spw": {**DEFAULTS, "scale_pos_weight": scale_pos_weight},
    }
    # Leave-one-out: revert a single parameter to its library default.
    for param, default_value in DEFAULTS.items():
        if CURRENT[param] == default_value:
            continue  # reg_lambda is already at its default
        cfgs[f"default_{param}"] = {
            **CURRENT, param: default_value, "scale_pos_weight": scale_pos_weight,
        }
    # scale_pos_weight is itself a deviation (library default is 1).
    cfgs["default_scale_pos_weight"] = {**CURRENT, "scale_pos_weight": 1.0}
    return cfgs


def make_model(kwargs):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0, **kwargs,
    )


def _to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def build_reference_pool():
    slide_ids = ["L321", "L34"]
    adatas = {}
    for sid in slide_ids:
        path = SLIDE_CACHE / f"sample_{sid}_adata.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
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
    print(f"reference pool: {len(y_ref):,} cells "
          f"({int((y_ref == 1).sum()):,} healthy look-alike / "
          f"{int((y_ref == 0).sum()):,} tumor anchor)")
    return X_ref, y_ref, list(adata_joint.var_names)


def cv_metrics(X_ref, y_ref, configs):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows, oof = [], {}
    for name, kwargs in configs.items():
        print(f"  CV: {name}", flush=True)
        p = cross_val_predict(make_model(kwargs), X_ref, y_ref, cv=cv,
                              method="predict_proba")[:, 1]
        oof[name] = p
        pred = (p >= PROB_THRESH).astype(int)
        rows.append({
            "config": name,
            "accuracy": accuracy_score(y_ref, pred),
            "precision": precision_score(y_ref, pred, zero_division=0),
            "recall": recall_score(y_ref, pred, zero_division=0),
            "f1": f1_score(y_ref, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_ref, p),
            "ap": average_precision_score(y_ref, p),
            **{k: v for k, v in kwargs.items()},
        })
    return pd.DataFrame(rows).set_index("config"), oof


def load_candidates(slice_id):
    slide_id, _ = SLICES[slice_id]
    csv_path = (BASE_DIR / "outputs" / "cell_annotation" / slide_id / "05" / str(slice_id)
                / f"slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    return _get_tumor_candidates_ids(slide_id, pd.read_csv(csv_path))


def score_slices(fitted, joint_var_names):
    """Apply every fitted configuration to the tumor candidates of each slice."""
    call_rows, calls_per_slice = [], {}
    for slice_id, (slide_id, slice_type) in SLICES.items():
        path = SLIDE_CACHE / f"slice_{slice_id}_adata.h5ad"
        if not path.exists():
            raise FileNotFoundError(path)
        adata = ad.read_h5ad(path)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, joint_var_names].copy()

        cand_ids = load_candidates(slice_id)
        is_cand = np.asarray(adata.obs_names.isin(cand_ids))
        n_cand = int(is_cand.sum())
        print(f"  slice {slice_id} ({slice_type}): {n_cand:,} candidates", flush=True)
        if n_cand == 0:
            continue

        X_cand = _to_dense(adata.X[is_cand])
        calls_per_slice[slice_id] = {}
        for name, model in fitted.items():
            keep = model.predict_proba(X_cand)[:, 1] < PROB_THRESH
            calls_per_slice[slice_id][name] = keep
            call_rows.append({
                "slice": slice_id, "slide": slide_id, "slice_type": slice_type,
                "config": name, "n_candidates": n_cand,
                "n_refined_tumor": int(keep.sum()),
                "pct_retained": 100.0 * keep.sum() / n_cand,
            })
        del adata, X_cand
    return pd.DataFrame(call_rows), calls_per_slice


def agreement_table(calls_per_slice, config_names):
    rows = []
    for name in config_names:
        n_same = n_tot = n_only_cur = n_only_new = n_both = 0
        for per_cfg in calls_per_slice.values():
            cur, new = per_cfg["current"], per_cfg[name]
            n_same += int((cur == new).sum())
            n_tot += cur.size
            n_only_cur += int((cur & ~new).sum())
            n_only_new += int((~cur & new).sum())
            n_both += int((cur & new).sum())
        union = n_both + n_only_cur + n_only_new
        rows.append({
            "config": name,
            "pct_candidates_same_call": 100.0 * n_same / n_tot,
            "n_lost_vs_current": n_only_cur,
            "n_gained_vs_current": n_only_new,
            "jaccard_vs_current": (n_both / union) if union else np.nan,
        })
    return pd.DataFrame(rows).set_index("config")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_ref, y_ref, joint_var_names = build_reference_pool()
    spw = int((y_ref == 0).sum()) / max(int((y_ref == 1).sum()), 1)
    print(f"scale_pos_weight (empirical tumor/healthy ratio) = {spw:.4f}")

    configs = build_configs(spw)
    print(f"\n{len(configs)} configurations, {N_SPLITS}-fold stratified CV")
    metrics_df, _ = cv_metrics(X_ref, y_ref, configs)
    print("\n=== Out-of-fold CV on the reference pool ===")
    print(metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc", "ap"]]
          .round(4).to_string())
    metrics_df.to_csv(OUT_DIR / "cv_metrics.csv")

    print("\nFitting each configuration on the full reference pool...")
    fitted = {}
    for name, kwargs in configs.items():
        model = make_model(kwargs)
        model.fit(X_ref, y_ref)
        fitted[name] = model
        print(f"  fit: {name}", flush=True)

    print("\nScoring tumor candidates slice by slice...")
    calls_df, calls_per_slice = score_slices(fitted, joint_var_names)
    calls_df.to_csv(OUT_DIR / "slice_calls.csv", index=False)

    pivot = calls_df.pivot_table(index="config", columns="slice",
                                 values="n_refined_tumor", aggfunc="sum")
    pivot["total"] = pivot.sum(axis=1)
    print("\n=== Refined tumor calls per slice (P(healthy) < 0.5) ===")
    print(pivot.to_string())

    agree_df = agreement_table(calls_per_slice, list(configs.keys()))
    agree_df.to_csv(OUT_DIR / "agreement.csv")
    print("\n=== Per-candidate agreement with the current settings ===")
    print(agree_df.round(4).to_string())
    print(f"\nwrote: {OUT_DIR}")
