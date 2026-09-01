"""Stage-1 threshold sensitivity with a held-out control-tissue evaluation.

Supersedes stage1_threshold_sensitivity.py, which had two defects:

  1. The healthy reference class is defined by the *same* filter as the tumor
     candidates, applied to the control sections. The earlier sweep held the
     classifier fixed while varying the candidate set, so at floors >= 0.2 every
     control cell being scored had also been a training negative and a zero
     false-positive count was close to guaranteed. Here the reference sets are
     rebuilt at every threshold, in lockstep.

  2. Control-tissue false positives were measured on training cells. Here they
     are measured cross-slide: a model trained on the L321 side (healthy
     look-alikes from slice 3, anchors from slice 1) is evaluated on the L34
     control section (slice 4), and vice versa. No cell contributes to both the
     training set and the error estimate.

The cross-slide split also yields a transfer AUROC, which is the only criterion
available that can discriminate between tumor-reference floors -- the
control-tissue criterion is saturated there (zero false positives throughout).

Three sweeps, as before: (A) candidate score_tumor floor, (B) delta_score
margin, (C) tumor-reference score_tumor floor. The margin is varied everywhere
it appears (candidates, healthy references and anchors), since it is one
parameter.

Outputs (thesis_plots/stage1_sensitivity_heldout/):
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
from xgboost import XGBClassifier

BASE_DIR = pathlib.Path(r"D:\thesis-research")
SLIDE_CACHE = BASE_DIR / "resources" / "cache"
OUT_DIR = BASE_DIR / "thesis_plots" / "stage1_sensitivity_heldout"

PROB_THRESH = 0.5
RANDOM_STATE = 42
N_JOBS = 4

SLICES = {1: ("L321", "Tumor"), 2: ("L321", "Tumor"), 3: ("L321", "Control"),
          4: ("L34", "Control"), 5: ("L34", "Tumor"), 6: ("L34", "Tumor")}
HEALTHY_REF_SLICE = {"L321": 3, "L34": 4}
TUMOR_REF_SLICE = {"L321": 1, "L34": 5}

CAND_FLOOR, CAND_DELTA, REF_FLOOR = 0.2, 0.08, 0.4
SWEEP_A = [0.10, 0.15, 0.20, 0.25, 0.30]
SWEEP_B = [0.00, 0.04, 0.06, 0.08, 0.10, 0.15]
SWEEP_C = [0.30, 0.35, 0.40, 0.45, 0.50]

MIN_FLOOR, MIN_DELTA, MIN_REF = min(SWEEP_A), min(SWEEP_B), min(SWEEP_C)


def _to_dense(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def annot(slice_id):
    slide, _ = SLICES[slice_id]
    path = (BASE_DIR / "outputs" / "cell_annotation" / slide / "05" / str(slice_id)
            / f"slice_{slice_id}_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv")
    df = pd.read_csv(path)
    df = df[df["predicted_cell_type"] == "Tumor"].copy()
    df["next_best"] = df[["score_brain_struct", "score_brain_immune"]].max(axis=1)
    df["delta_score"] = (df["score_tumor"] - df["next_best"]).abs()
    df["cell_barcode"] = df["cell_barcode"].astype(str)
    return df.set_index("cell_barcode")


ANNOT = {s: annot(s) for s in SLICES}


def passes(slice_id, floor, delta, ge=False):
    """Barcodes in a slice passing the Stage-1 filter."""
    d = ANNOT[slice_id]
    score_ok = (d["score_tumor"] >= floor) if ge else (d["score_tumor"] > floor)
    return d.index[score_ok & (d["delta_score"] > delta)
                   & (d["score_tumor"] > d["next_best"])]


# --------------------------------------------------------------------------- #
# Expression cache
# --------------------------------------------------------------------------- #
def load_slice(slice_id, barcodes, var_names=None):
    adata = ad.read_h5ad(SLIDE_CACHE / f"slice_{slice_id}_adata.h5ad")
    keep = np.asarray(adata.obs_names.isin(set(barcodes)))
    sub = adata[keep].copy()
    del adata
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    if var_names is not None:
        sub = sub[:, var_names].copy()
    names = np.asarray(sub.obs_names.astype(str))
    X = _to_dense(sub.X)
    vn = list(sub.var_names)
    del sub
    return names, X, vn


print("Loading training cells (supersets across all sweeps)...")
TRAIN = {}
VAR_NAMES = None
for sl in [1, 5]:                                    # tumor anchors
    bcs = passes(sl, MIN_REF, MIN_DELTA, ge=True)
    names, X, vn = load_slice(sl, bcs, VAR_NAMES)
    VAR_NAMES = VAR_NAMES or vn
    TRAIN[sl] = (names, X)
    print(f"  slice {sl}: {len(names):,} anchor-eligible cells")
for sl in [3, 4]:                                    # healthy look-alikes
    bcs = passes(sl, MIN_FLOOR, MIN_DELTA)
    names, X, vn = load_slice(sl, bcs, VAR_NAMES)
    TRAIN[sl] = (names, X)
    print(f"  slice {sl}: {len(names):,} look-alike-eligible cells")


def subset(slice_id, barcodes):
    names, X = TRAIN[slice_id]
    mask = np.isin(names, np.asarray(barcodes))
    return X[mask]


def build_xy(slides, floor, delta, ref_floor):
    """Training matrix from the given slides. y = 1 is healthy look-alike."""
    Xs, ys = [], []
    for slide in slides:
        h = passes(HEALTHY_REF_SLICE[slide], floor, delta)
        t = passes(TUMOR_REF_SLICE[slide], ref_floor, delta, ge=True)
        Xh = subset(HEALTHY_REF_SLICE[slide], h)
        Xt = subset(TUMOR_REF_SLICE[slide], t)
        Xs += [Xh, Xt]
        ys += [np.ones(Xh.shape[0], int), np.zeros(Xt.shape[0], int)]
    return np.vstack(Xs), np.concatenate(ys)


def fit(X, y):
    m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
    m.fit(X, y)
    return m


# --------------------------------------------------------------------------- #
# Train every model the sweeps need
# --------------------------------------------------------------------------- #
CONFIGS = {}
for f in SWEEP_A:
    CONFIGS[("A", f)] = (f, CAND_DELTA, REF_FLOOR)
for d in SWEEP_B:
    CONFIGS[("B", d)] = (CAND_FLOOR, d, REF_FLOOR)
for r in SWEEP_C:
    CONFIGS[("C", r)] = (CAND_FLOOR, CAND_DELTA, r)

print(f"\nTraining {len(CONFIGS)} full models + {2 * len(CONFIGS)} cross-slide models...")
FULL, FOLD, XSLIDE_AUC, NTRAIN = {}, {}, {}, {}
for key, (f, d, r) in CONFIGS.items():
    X, y = build_xy(["L321", "L34"], f, d, r)
    FULL[key] = fit(X, y)
    NTRAIN[key] = (int((y == 1).sum()), int((y == 0).sum()))
    aucs = []
    for train_slide, test_slide in [("L321", "L34"), ("L34", "L321")]:
        Xtr, ytr = build_xy([train_slide], f, d, r)
        m = fit(Xtr, ytr)
        FOLD[(key, test_slide)] = m
        Xte, yte = build_xy([test_slide], f, d, r)
        if len(np.unique(yte)) == 2:
            aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    XSLIDE_AUC[key] = float(np.mean(aucs)) if aucs else np.nan
    print(f"  {key}: {NTRAIN[key][0]:,} healthy / {NTRAIN[key][1]:,} anchors, "
          f"cross-slide AUROC {XSLIDE_AUC[key]:.4f}", flush=True)

# --------------------------------------------------------------------------- #
# Score candidates
# --------------------------------------------------------------------------- #
print("\nScoring candidates slice by slice...")
refined = {k: {"tumor": 0, "control_in_sample": 0} for k in CONFIGS}
heldout_fp = {k: 0 for k in CONFIGS}
cand_counts = {k: {"tumor": 0, "control": 0} for k in CONFIGS}

for slice_id, (slide, typ) in SLICES.items():
    union = set(passes(slice_id, MIN_FLOOR, CAND_DELTA)) | set(
        passes(slice_id, CAND_FLOOR, MIN_DELTA))
    names, X, _ = load_slice(slice_id, union, VAR_NAMES)
    print(f"  slice {slice_id} ({typ}): {len(names):,} cells", flush=True)
    meta = ANNOT[slice_id].loc[names]
    score = meta["score_tumor"].to_numpy()
    delta = meta["delta_score"].to_numpy()
    nxt = meta["next_best"].to_numpy()

    for key, (f, d, r) in CONFIGS.items():
        sel = (score > f) & (delta > d) & (score > nxt)
        if sel.sum() == 0:
            continue
        p_full = FULL[key].predict_proba(X[sel])[:, 1]
        n_ref = int((p_full < PROB_THRESH).sum())
        if typ == "Tumor":
            refined[key]["tumor"] += n_ref
            cand_counts[key]["tumor"] += int(sel.sum())
        else:
            refined[key]["control_in_sample"] += n_ref
            cand_counts[key]["control"] += int(sel.sum())
            # Held-out: score this control section with the model trained on the
            # *other* slide, so none of these cells were training negatives.
            p_out = FOLD[(key, slide)].predict_proba(X[sel])[:, 1]
            heldout_fp[key] += int((p_out < PROB_THRESH).sum())
    del X, names

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
OUT_DIR.mkdir(parents=True, exist_ok=True)


def table(sweep, values, label):
    rows = []
    for v in values:
        key = (sweep, v)
        rows.append({
            label: v,
            "n_healthy_train": NTRAIN[key][0], "n_anchors_train": NTRAIN[key][1],
            "tumor_candidates": cand_counts[key]["tumor"],
            "refined_tumor": refined[key]["tumor"],
            "control_candidates": cand_counts[key]["control"],
            "control_fp_in_sample": refined[key]["control_in_sample"],
            "control_fp_heldout": heldout_fp[key],
            "cross_slide_auroc": XSLIDE_AUC[key],
        })
    df = pd.DataFrame(rows)
    base = {"A": CAND_FLOOR, "B": CAND_DELTA, "C": REF_FLOOR}[sweep]
    ref = df.loc[df[label] == base, "refined_tumor"].iloc[0]
    df["pct_vs_thesis"] = 100.0 * (df["refined_tumor"] - ref) / ref
    return df


a = table("A", SWEEP_A, "candidate_floor")
b = table("B", SWEEP_B, "delta_margin")
c = table("C", SWEEP_C, "reference_floor")
a.to_csv(OUT_DIR / "sweep_a_candidate_floor.csv", index=False)
b.to_csv(OUT_DIR / "sweep_b_delta_margin.csv", index=False)
c.to_csv(OUT_DIR / "sweep_c_reference_floor.csv", index=False)

print("\n=== Sweep A - candidate score_tumor floor ===")
print(a.round(4).to_string(index=False))
print("\n=== Sweep B - delta_score margin ===")
print(b.round(4).to_string(index=False))
print("\n=== Sweep C - tumor-reference floor ===")
print(c.round(4).to_string(index=False))
print(f"\nwrote: {OUT_DIR}")
