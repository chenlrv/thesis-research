"""Stage-1 SingleR threshold sensitivity, measured end-to-end.

The Stage-1 thresholds (score_tumor floor for candidates, score_tumor floor for
the tumor reference set, delta_score margin) have no library default to fall
back on, so they are justified empirically instead. Two properties are measured:

  * Purity. The two control sections contain no tumor, so every Tumor call there
    is a false positive by construction. This gives an observable false-positive
    rate as a function of each threshold.

  * Insensitivity. What matters is not the threshold itself but the *pipeline
    output*. Each sweep is therefore carried through Stage 3 and reported as the
    final refined tumor count, not as a Stage-1 candidate count.

Three sweeps:

  A  candidate score_tumor floor   0.10 - 0.30, delta fixed at 0.08
     Stage-3 classifier held fixed; only the candidate set changes.

  B  candidate delta_score margin  0.00 - 0.15, score_tumor fixed at 0.2
     Stage-3 classifier held fixed; only the candidate set changes.

  C  tumor-reference score_tumor floor 0.30 - 0.50
     The classifier is retrained at each floor, then applied to the standard
     candidate set. Reports anchor-set purity and out-of-fold AUROC as well.

Outputs (thesis_plots/stage1_sensitivity/):
    sweep_a_candidate_floor.csv
    sweep_b_delta_margin.csv
    sweep_c_reference_floor.csv
"""
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
OUT_DIR = BASE_DIR / "thesis_plots" / "stage1_sensitivity"

PROB_THRESH = 0.5      # P(healthy) < this is retained as refined tumor
RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = 4

SLICES = {1: ("L321", "Tumor"), 2: ("L321", "Tumor"), 3: ("L321", "Control"),
          4: ("L34", "Control"), 5: ("L34", "Tumor"), 6: ("L34", "Tumor")}

# Which slice supplies which reference set, per slide (as in identify_tumor_cells).
HEALTHY_REF_SLICE = {"L321": 3, "L34": 4}   # control sections -> false-positive look-alikes
TUMOR_REF_SLICE = {"L321": 1, "L34": 5}     # tumor sections   -> tumor anchors

# Thesis operating point.
CAND_FLOOR = 0.2
CAND_DELTA = 0.08
REF_FLOOR = 0.4

SWEEP_A = [0.10, 0.15, 0.20, 0.25, 0.30]
SWEEP_B = [0.00, 0.04, 0.06, 0.08, 0.10, 0.15]
SWEEP_C = [0.30, 0.35, 0.40, 0.45, 0.50]


def _to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def annot(slice_id):
    """SingleR output for one slice, with delta_score and next_best attached."""
    slide, _ = SLICES[slice_id]
    path = (BASE_DIR / "outputs" / "cell_annotation" / slide / "05" / str(slice_id)
            / f"slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df = pd.read_csv(path)
    df = df[df["predicted_cell_type"] == "Tumor"].copy()
    df["next_best"] = df[["score_brain_struct", "score_brain_immune"]].max(axis=1)
    df["delta_score"] = (df["score_tumor"] - df["next_best"]).abs()
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    return df


def select(df, floor, delta):
    """The Stage-1 filter, with its two thresholds exposed."""
    return df[(df["score_tumor"] > floor)
              & (df["delta_score"] > delta)
              & (df["score_tumor"] > df["next_best"])]


ANNOT = {s: annot(s) for s in SLICES}


_CACHE = None


def _build_reference_cache():
    """Load the two slides once and keep only cells any reference floor could use.

    Re-reading the full 846k-cell slides for each floor exhausts memory, so the
    superset (healthy look-alikes plus anchors at the loosest floor) is loaded
    once and each floor is then a mask over it.
    """
    healthy, anchors = set(), {}
    for slide in ["L321", "L34"]:
        h = select(ANNOT[HEALTHY_REF_SLICE[slide]], CAND_FLOOR, CAND_DELTA)
        healthy |= set(h["cell_barcode"])
        t = ANNOT[TUMOR_REF_SLICE[slide]]
        t = t[(t["score_tumor"] >= min(SWEEP_C)) & (t["delta_score"] > CAND_DELTA)
              & (t["score_tumor"] > t["next_best"])]
        for bc, s in zip(t["cell_barcode"], t["score_tumor"]):
            anchors[bc] = max(anchors.get(bc, -np.inf), s)
    all_ref = healthy | set(anchors)

    adatas = {}
    for slide in ["L321", "L34"]:
        adata = ad.read_h5ad(SLIDE_CACHE / f"sample_{slide}_adata.h5ad")
        sub = adata[np.asarray(adata.obs_names.isin(all_ref))].copy()
        del adata
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        adatas[slide] = sub

    joint = ad.concat(adatas, join="inner", label="slide_id")
    del adatas
    names = joint.obs_names.astype(str)
    X_all = _to_dense(joint.X)
    is_healthy = np.asarray(names.isin(healthy))
    anchor_score = np.array([anchors.get(n, np.nan) for n in names], dtype=float)
    print(f"reference cache: {X_all.shape[0]:,} cells "
          f"({int(is_healthy.sum()):,} healthy look-alike, "
          f"{int((~np.isnan(anchor_score)).sum()):,} candidate anchors)")
    return X_all, is_healthy, anchor_score, list(joint.var_names)


def reference_pool(ref_floor):
    """(X, y, var_names, n_anchors) for a floor. y = 1 is healthy look-alike."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _build_reference_cache()
    X_all, is_healthy, anchor_score, var_names = _CACHE
    is_anchor = ~np.isnan(anchor_score) & (anchor_score >= ref_floor) & ~is_healthy
    mask = is_healthy | is_anchor
    return X_all[mask], is_healthy[mask].astype(int), var_names, int(is_anchor.sum())


def fit_model(X, y):
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
    model.fit(X, y)
    return model


def oof_auroc(X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def score_union(models, var_names):
    """P(healthy) under every model, for every cell any sweep might select.

    Returns slice_id -> DataFrame(cell_barcode, score_tumor, delta_score, p_<model>).
    """
    loosest_floor, loosest_delta = min(SWEEP_A + [CAND_FLOOR]), min(SWEEP_B + [CAND_DELTA])
    out = {}
    for slice_id in SLICES:
        df = ANNOT[slice_id]
        # Union of every candidate set the sweeps can produce.
        union_a = select(df, loosest_floor, CAND_DELTA)
        union_b = select(df, CAND_FLOOR, loosest_delta)
        union = pd.concat([union_a, union_b]).drop_duplicates("cell_barcode")
        ids = set(union["cell_barcode"])
        print(f"  slice {slice_id}: scoring {len(ids):,} cells", flush=True)

        adata = ad.read_h5ad(SLIDE_CACHE / f"slice_{slice_id}_adata.h5ad")
        keep = np.asarray(adata.obs_names.isin(ids))
        sub = adata[keep].copy()
        del adata
        sc.pp.normalize_total(sub, target_sum=1e4)
        sc.pp.log1p(sub)
        sub = sub[:, var_names].copy()
        X = _to_dense(sub.X)

        res = pd.DataFrame({"cell_barcode": sub.obs_names.astype(str)})
        res = res.merge(union[["cell_barcode", "score_tumor", "delta_score", "next_best"]],
                        on="cell_barcode", how="left")
        for name, model in models.items():
            res[f"p_{name}"] = model.predict_proba(X)[:, 1]
        out[slice_id] = res
        del sub, X
    return out


def refined_counts(scored, prob_col, floor, delta):
    """Refined tumor calls under a Stage-1 filter, split by section type."""
    tumor_n = ctrl_n = cand_t = cand_c = 0
    for slice_id, res in scored.items():
        sel = res[(res["score_tumor"] > floor) & (res["delta_score"] > delta)
                  & (res["score_tumor"] > res["next_best"])]
        n_refined = int((sel[prob_col] < PROB_THRESH).sum())
        if SLICES[slice_id][1] == "Tumor":
            tumor_n += n_refined
            cand_t += len(sel)
        else:
            ctrl_n += n_refined
            cand_c += len(sel)
    return cand_t, tumor_n, cand_c, ctrl_n


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Train one classifier per tumor-reference floor (sweep C) ------------
    print("Training Stage-3 classifiers across tumor-reference floors...")
    models, ref_info = {}, []
    var_names = None
    for floor in SWEEP_C:
        X, y, vn, n_anchor = reference_pool(floor)
        if var_names is None:
            var_names = vn
        auc = oof_auroc(X, y)
        models[f"ref{floor}"] = fit_model(X, y)
        ref_info.append({"reference_floor": floor, "n_tumor_anchors": n_anchor,
                         "n_healthy_refs": int((y == 1).sum()), "oof_auroc": auc})
        print(f"  floor {floor}: {n_anchor:,} anchors, OOF AUROC {auc:.4f}", flush=True)
        del X, y

    # ---- Anchor purity, measured on control tissue --------------------------
    # Applying the anchor filter to the control sections, where any Tumor call is
    # a false positive, gives the contamination rate of the anchor set.
    for row in ref_info:
        floor = row["reference_floor"]
        fp = tot = 0
        for slice_id, (slide, typ) in SLICES.items():
            d = ANNOT[slice_id]
            sel = d[(d["score_tumor"] >= floor) & (d["delta_score"] > CAND_DELTA)
                    & (d["score_tumor"] > d["next_best"])]
            if typ == "Control":
                fp += len(sel)
            else:
                tot += len(sel)
        row["control_false_pos"] = fp
        row["tumor_slice_cells"] = tot
        row["anchor_purity_pct"] = 100.0 * tot / max(tot + fp, 1)

    # ---- Score every cell any sweep might touch -----------------------------
    print("\nScoring candidate cells slice by slice...")
    scored = score_union(models, var_names)

    baseline = f"ref{REF_FLOOR}"

    # ---- Sweep A: candidate score_tumor floor -------------------------------
    rows = []
    for floor in SWEEP_A:
        ct, nt, cc, nc = refined_counts(scored, f"p_{baseline}", floor, CAND_DELTA)
        rows.append({"candidate_floor": floor, "tumor_candidates": ct,
                     "refined_tumor": nt, "control_candidates": cc,
                     "control_refined": nc,
                     "pct_vs_thesis": np.nan})
    a = pd.DataFrame(rows)
    ref_val = a.loc[a["candidate_floor"] == CAND_FLOOR, "refined_tumor"].iloc[0]
    a["pct_vs_thesis"] = 100.0 * (a["refined_tumor"] - ref_val) / ref_val
    a.to_csv(OUT_DIR / "sweep_a_candidate_floor.csv", index=False)
    print("\n=== Sweep A - candidate score_tumor floor (delta = 0.08) ===")
    print(a.round(2).to_string(index=False))

    # ---- Sweep B: candidate delta_score margin ------------------------------
    rows = []
    for delta in SWEEP_B:
        ct, nt, cc, nc = refined_counts(scored, f"p_{baseline}", CAND_FLOOR, delta)
        rows.append({"delta_margin": delta, "tumor_candidates": ct,
                     "refined_tumor": nt, "control_candidates": cc,
                     "control_refined": nc, "pct_vs_thesis": np.nan})
    b = pd.DataFrame(rows)
    ref_val = b.loc[b["delta_margin"] == CAND_DELTA, "refined_tumor"].iloc[0]
    b["pct_vs_thesis"] = 100.0 * (b["refined_tumor"] - ref_val) / ref_val
    b.to_csv(OUT_DIR / "sweep_b_delta_margin.csv", index=False)
    print("\n=== Sweep B - candidate delta_score margin (floor = 0.2) ===")
    print(b.round(2).to_string(index=False))

    # ---- Sweep C: tumor-reference floor -------------------------------------
    rows = []
    for row in ref_info:
        floor = row["reference_floor"]
        ct, nt, cc, nc = refined_counts(scored, f"p_ref{floor}", CAND_FLOOR, CAND_DELTA)
        rows.append({**row, "refined_tumor": nt, "control_refined": nc})
    c = pd.DataFrame(rows)
    ref_val = c.loc[c["reference_floor"] == REF_FLOOR, "refined_tumor"].iloc[0]
    c["pct_vs_thesis"] = 100.0 * (c["refined_tumor"] - ref_val) / ref_val
    c.to_csv(OUT_DIR / "sweep_c_reference_floor.csv", index=False)
    print("\n=== Sweep C - tumor-reference score_tumor floor ===")
    print(c.round(4).to_string(index=False))

    print(f"\nwrote: {OUT_DIR}")
