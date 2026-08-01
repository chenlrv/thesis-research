"""
Annotation v2: score_genes priors + XGBoost correction.

Two-stage annotation, designed as the cell-type analogue of the tumor pipeline
in tumor_cells/refine_annotation_classifiers.py. The upstream method
(`sc.tl.score_genes` on canonical marker modules) plays the role SingleR plays
in the tumor pipeline; XGBoost is the downstream corrector.

Stage 1 — module-score priors (sc.tl.score_genes):
    For each class in ANCHOR_CLASSES, compute a per-cell score over a
    canonical marker module. Every cell gets a first-pass label = argmax
    over the four module scores.

Stage 2 — XGBoost correction on lineage-gated anchors:
    High-confidence anchors are the subset of cells whose argmax class
    passes (a) a per-class q90 module-score threshold, (b) a confidence
    margin between best and second-best score, and (c) a lineage gate
    (tdTomato status / GFP status / tumor-vs-control slice). Anchors are
    the training set for a multi-class XGBoost classifier. Anchor-defining
    genes (module markers + lineage tracers) are WITHHELD from features so
    the classifier learns the broader transcriptional correlates of each
    label rather than memorising the anchor rule.

Stage 3 — control-slice refinement (same motif as tumor pipeline):
    MDM is tumor-recruited, so any cell the round-1 XGBoost predicts as MDM
    in a control slice is biologically near-impossible. These hard negatives
    are folded back into a second round of training under a new 'other'
    class (no forced re-assignment to microglia/BAM/astrocyte). Round 2
    fits a five-class classifier that can natively assign 'other' at
    inference, giving an honest 'none of the above' bucket.

Outputs (written to adata.obs):
    score_<class>                       : sc.tl.score_genes module score
    first_pass_label                    : argmax of module scores
    anchor_label                        : high-confidence subset (XGBoost training set)
    score_<class>_round1                : XGBoost round-1 probability
    pred_xgb_round1                     : XGBoost round-1 argmax (4 classes)
    score_<class>_refined               : XGBoost round-2 probability
    pred_xgb_refined                    : XGBoost round-2 argmax (5 classes incl. 'other')
"""

import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

WITH_TUMOR_DIR = pathlib.Path(r"D:\thesis-research\resources\cache\with_tumor_prediction")
TUMOR_COL = "pred_tumor_XGBoost"

TUMOR_SLICES = {1, 2, 5, 6}
CONTROL_SLICES = {3, 4}
SLICE_TO_SLIDE = {1: "L321", 2: "L321", 3: "L321", 4: "L34", 5: "L34", 6: "L34"}

ANCHOR_CLASSES = ("microglia", "BAM", "MDM", "astrocyte")
OTHER_CLASS = "other"
REFINED_CLASSES = ANCHOR_CLASSES + (OTHER_CLASS,)

# Canonical marker modules — used by sc.tl.score_genes to produce per-cell
# module scores. These play the same role as SingleR's per-class scores in
# the tumor pipeline.
MARKER_MODULES = {
    "microglia": ["TMEM119", "Trem2", "Cx3cr1", "C1qa", "C1qb", "C1qc", "P2ry12"],
    "BAM":       ["Lyve1", "Mrc1", "Cd163", "Pf4", "Marco", "Selenop", "Cd209a"],
    "MDM":       ["Ccr2", "Cd14", "S100a4", "Plac8", "Crip1", "Ly6c2"],
    "astrocyte": ["GFAP", "S100b", "Sox9", "Glul", "Sparcl1", "Clu", "Apod"],
}

# Lineage tracer genes — used as gates on anchor assignment only, never in
# module scores or classifier features.
LINEAGE_GENES = ["tdTomato", "GFP"]

# All anchor-defining genes (module markers + lineage tracers) are withheld
# from classifier features to prevent the classifier from trivially memorising
# the anchor rule.
ANCHOR_GENES = (
    {g.lower() for genes in MARKER_MODULES.values() for g in genes}
    | {g.lower() for g in LINEAGE_GENES}
)


def _to_dense(x):
    if issparse(x):
        return x.toarray()
    return np.asarray(x)


def _gene_lookup(adata, gene):
    """Case-insensitive panel lookup. Returns actual var_name or None."""
    panel = {g.lower(): g for g in adata.var_names}
    return panel.get(gene.lower())


def _gene_expr(adata, gene):
    """Return per-cell expression vector for `gene`, or None if missing from the panel."""
    actual = _gene_lookup(adata, gene)
    if actual is None:
        return None
    idx = list(adata.var_names).index(actual)
    col = adata.X[:, idx]
    return col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()


def load_slice(slice_id):
    """Load a slice, drop predicted tumor cells, normalize + log1p, tag metadata."""
    path = WITH_TUMOR_DIR / f"slice_{slice_id}_adata.h5ad"
    adata = ad.read_h5ad(path)
    print(f"slice {slice_id}: loaded {adata.n_obs:,} cells")

    non_tumor = adata.obs[TUMOR_COL].fillna(0).astype(int) == 0
    adata = adata[non_tumor].copy()
    print(f"slice {slice_id}: kept {adata.n_obs:,} non-tumor cells")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.obs["slice_id"] = slice_id
    adata.obs["slide_id"] = SLICE_TO_SLIDE[slice_id]
    adata.obs["slice_type"] = "tumor" if slice_id in TUMOR_SLICES else "control"
    return adata


def load_all_slices():
    """Load all 6 slices and concatenate into one joint AnnData."""
    adatas = {}
    for sid in sorted(TUMOR_SLICES | CONTROL_SLICES):
        path = WITH_TUMOR_DIR / f"slice_{sid}_adata.h5ad"
        if not path.exists():
            print(f"slice {sid}: missing at {path}, skipping")
            continue
        adatas[sid] = load_slice(sid)
    if not adatas:
        raise FileNotFoundError(f"no slice files found in {WITH_TUMOR_DIR}")

    joint = ad.concat(adatas, join="inner", label="slice_concat_key", index_unique="-")
    print(f"\njoint adata: {joint.n_obs:,} cells x {joint.n_vars:,} genes")
    return joint


def score_marker_modules(adata):
    """Compute per-class module scores using sc.tl.score_genes.

    Adds adata.obs['score_<class>'] for every class in ANCHOR_CLASSES.
    Missing genes are skipped (with a warning), not an error.
    """
    panel_lookup = {g.lower(): g for g in adata.var_names}

    for cls, genes in MARKER_MODULES.items():
        present = [panel_lookup[g.lower()] for g in genes if g.lower() in panel_lookup]
        missing = [g for g in genes if g.lower() not in panel_lookup]
        if missing:
            print(f"{cls}: missing from panel -> {missing}")
        print(f"{cls}: score_genes on {len(present)} genes -> {present}")
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{cls}", use_raw=False)


def assign_first_pass_labels(
    adata,
    percentile: float = 0.90,
    margin_min: float = 0.10,
):
    """Compute first-pass labels and lineage-gated high-confidence anchors.

    Adds to adata.obs:
        first_pass_label : argmax over module scores (every cell gets a label,
                           analogous to SingleR's predicted_cell_type)
        score_top        : best module score
        score_margin     : best - second-best
        anchor_label     : high-confidence subset used as XGBoost training data
                           (argmax class passes per-class q<percentile> score
                            AND margin >= margin_min AND lineage gate).
                           Cells that fail are 'unlabeled'.
    """
    score_cols = [f"score_{c}" for c in ANCHOR_CLASSES]
    scores = np.column_stack([adata.obs[c].to_numpy(dtype=float) for c in score_cols])

    best_idx = scores.argmax(axis=1)
    best_class = np.array(ANCHOR_CLASSES)[best_idx]
    best_score = scores[np.arange(len(scores)), best_idx]
    second_score = np.partition(scores, -2, axis=1)[:, -2]
    margin = best_score - second_score

    adata.obs["first_pass_label"] = pd.Categorical(best_class, categories=list(ANCHOR_CLASSES))
    adata.obs["score_top"] = best_score
    adata.obs["score_margin"] = margin

    # Per-class confidence thresholds: top quantile of each class's score.
    thresholds = {
        c: float(np.nanquantile(adata.obs[f"score_{c}"], percentile))
        for c in ANCHOR_CLASSES
    }

    # Lineage gates — same as v2 lineage anchors.
    tdT = _gene_expr(adata, "tdTomato")
    gfp = _gene_expr(adata, "GFP")
    if tdT is None or gfp is None:
        raise ValueError("tdTomato and GFP must be present in the panel for lineage gates")
    is_tumor_slice = (adata.obs["slice_type"] == "tumor").to_numpy()

    gates = {
        "microglia": (tdT == 0),
        "BAM":       (tdT == 0),
        "MDM":       (tdT > 0) & is_tumor_slice,
        "astrocyte": (gfp == 0) & (tdT == 0),
    }

    labels = np.full(adata.n_obs, "unlabeled", dtype=object)
    for c in ANCHOR_CLASSES:
        mask = (
            (best_class == c)
            & (best_score >= thresholds[c])
            & (margin >= margin_min)
            & gates[c]
        )
        labels[mask] = c

    adata.obs["anchor_label"] = pd.Categorical(
        labels, categories=list(ANCHOR_CLASSES) + ["unlabeled"]
    )

    print("\nFirst-pass label counts (argmax over module scores):")
    for c in ANCHOR_CLASSES:
        k = int((best_class == c).sum())
        print(f"  {c:<10}: {k:>8,}")

    print(f"\nPer-class module-score thresholds (q{int(percentile*100)}):")
    for c in ANCHOR_CLASSES:
        print(f"  {c:<10}: {thresholds[c]:.3f}")
    print(f"  margin_min = {margin_min}")

    print("\nAnchor (XGBoost training set) counts:")
    for c in list(ANCHOR_CLASSES) + ["unlabeled"]:
        k = int((labels == c).sum())
        pct = 100 * k / len(labels)
        print(f"  {c:<10}: {k:>8,}  ({pct:5.1f}%)")

    print("\nAnchor counts per slice type:")
    for stype in ("tumor", "control"):
        smask = (adata.obs["slice_type"] == stype).to_numpy()
        for c in ANCHOR_CLASSES:
            k = int(((labels == c) & smask).sum())
            print(f"  {stype:<8} | {c:<10}: {k:>7,}")


def _feature_mask(adata):
    """Boolean mask over var_names. Excludes anchor-defining genes (no leakage)."""
    mask = np.array([g.lower() not in ANCHOR_GENES for g in adata.var_names])
    excluded = [g for g in adata.var_names if g.lower() in ANCHOR_GENES]
    print(f"\nfeature space: {mask.sum()} genes (excluded {len(excluded)}: {excluded})")
    return mask


def _make_xgb(random_state, num_class):
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def _balanced_sample_weight(y, num_class):
    counts = np.bincount(y, minlength=num_class)
    weights = len(y) / (num_class * np.maximum(counts, 1))
    return weights[y]


def train_anchor_classifier(adata, random_state=42):
    """Train multi-class XGBoost on anchor cells. Returns (clf, feat_mask)."""
    feat_mask = _feature_mask(adata)
    label_arr = adata.obs["anchor_label"].astype(str).to_numpy()
    is_anchor = np.isin(label_arr, ANCHOR_CLASSES)

    if issparse(adata.X):
        X_full = adata.X.tocsc()[:, feat_mask].tocsr()
    else:
        X_full = adata.X[:, feat_mask]

    X_train = _to_dense(X_full[is_anchor])
    y_train_str = label_arr[is_anchor]
    class_to_int = {c: i for i, c in enumerate(ANCHOR_CLASSES)}
    y_train = np.array([class_to_int[c] for c in y_train_str])

    print(f"\ntraining XGBoost on {len(y_train):,} anchor cells:")
    for c in ANCHOR_CLASSES:
        print(f"  {c:<10}: {int((y_train_str == c).sum()):>8,}")

    sample_weight = _balanced_sample_weight(y_train, len(ANCHOR_CLASSES))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    acc_scores, f1_scores = [], []
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), start=1):
        fold_clf = _make_xgb(random_state, len(ANCHOR_CLASSES))
        fold_clf.fit(X_train[tr], y_train[tr], sample_weight=sample_weight[tr])
        y_pred = fold_clf.predict(X_train[va])
        acc_scores.append(accuracy_score(y_train[va], y_pred))
        f1_scores.append(f1_score(y_train[va], y_pred, average="macro"))
        print(f"  fold {fold}: accuracy={acc_scores[-1]:.4f}  f1_macro={f1_scores[-1]:.4f}")
    print("\n5-fold CV on anchors:")
    print(f"  accuracy : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"  f1_macro : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    clf = _make_xgb(random_state, len(ANCHOR_CLASSES))
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    proba_all = clf.predict_proba(X_full)
    pred_int = proba_all.argmax(axis=1)
    pred_str = np.array(ANCHOR_CLASSES)[pred_int]

    for i, c in enumerate(ANCHOR_CLASSES):
        adata.obs[f"score_{c}_round1"] = proba_all[:, i]
    adata.obs["pred_xgb_round1"] = pd.Categorical(pred_str, categories=list(ANCHOR_CLASSES))

    return clf, feat_mask


def refine_with_control_slices(adata, feat_mask, random_state=42):
    """Round-2 retraining with control-slice look-alike MDMs as an 'other' class.

    Look-alikes = cells the round-1 classifier predicted as MDM in a control
    slice. Biologically near-impossible (no tumor -> no monocyte recruitment),
    so they form a clean hard-negative set. Round 2 adds them to training
    under a new 'other' label and refits a five-class classifier.
    """
    is_control = (adata.obs["slice_type"] == "control").to_numpy()
    pred_round1 = adata.obs["pred_xgb_round1"].astype(str).to_numpy()
    look_alike = is_control & (pred_round1 == "MDM")
    n_look = int(look_alike.sum())
    print(f"\nlook-alike MDMs (predicted MDM in control slice): {n_look:,}")
    if n_look == 0:
        print("  none — skipping refinement")
        return None

    label_arr = adata.obs["anchor_label"].astype(str).to_numpy()
    is_anchor = np.isin(label_arr, ANCHOR_CLASSES)
    anchor_idx = np.where(is_anchor)[0]
    look_idx = np.where(look_alike)[0]

    other_labels = np.full(len(look_idx), OTHER_CLASS, dtype=object)
    train_idx = np.concatenate([anchor_idx, look_idx])
    train_labels = np.concatenate([label_arr[anchor_idx], other_labels])

    class_to_int = {c: i for i, c in enumerate(REFINED_CLASSES)}
    y_train = np.array([class_to_int[c] for c in train_labels])

    if issparse(adata.X):
        X_full = adata.X.tocsc()[:, feat_mask].tocsr()
    else:
        X_full = adata.X[:, feat_mask]
    X_train = _to_dense(X_full[train_idx])

    print(f"\nround-2 training set: {len(y_train):,} cells "
          f"({len(anchor_idx):,} anchors + {len(look_idx):,} look-alikes labelled '{OTHER_CLASS}')")
    for c in REFINED_CLASSES:
        print(f"  {c:<10}: {int((train_labels == c).sum()):>8,}")

    sample_weight = _balanced_sample_weight(y_train, len(REFINED_CLASSES))
    clf2 = _make_xgb(random_state, len(REFINED_CLASSES))
    clf2.fit(X_train, y_train, sample_weight=sample_weight)

    proba_all2 = clf2.predict_proba(X_full)
    pred_int2 = proba_all2.argmax(axis=1)
    pred_str2 = np.array(REFINED_CLASSES)[pred_int2]
    for i, c in enumerate(REFINED_CLASSES):
        adata.obs[f"score_{c}_refined"] = proba_all2[:, i]
    adata.obs["pred_xgb_refined"] = pd.Categorical(pred_str2, categories=list(REFINED_CLASSES))

    print("\nMDM predictions per slice (round 1 -> refined):")
    for sid in sorted(TUMOR_SLICES | CONTROL_SLICES):
        smask = (adata.obs["slice_id"] == sid).to_numpy()
        if not smask.any():
            continue
        n_r1 = int(((pred_round1 == "MDM") & smask).sum())
        n_r2 = int(((pred_str2 == "MDM") & smask).sum())
        stype = "tumor  " if sid in TUMOR_SLICES else "control"
        print(f"  slice {sid} ({stype}): {n_r1:>6,}  ->  {n_r2:>6,}")

    print("\nRefined prediction counts across all cells:")
    for c in REFINED_CLASSES:
        print(f"  {c:<10}: {int((pred_str2 == c).sum()):>8,}")

    return clf2


if __name__ == "__main__":
    print("Starting annotation v2 — score_genes priors + XGBoost correction")

    adata = load_all_slices()
    score_marker_modules(adata)
    assign_first_pass_labels(adata)
    clf, feat_mask = train_anchor_classifier(adata)
    refine_with_control_slices(adata, feat_mask)

    out_path = WITH_TUMOR_DIR / "annotation_v2_joint.h5ad"
    adata.write_h5ad(out_path)
    print(f"\nwrote: {out_path}")
