"""Control-calibrated gate for myeloid subtypes (Microglia/MDM/BAM), slices 1,2,3.

WHY THIS REPLACES THE GMM GATE
------------------------------
A 2-component GMM is a *clustering*, not a presence/absence *detector*: it always
splits a score into a low/high component and calls the top chunk "+", even when
there is no real positive population. Slice 3 is a CONTROL (non-tumor) slice, so
it should contain essentially NO MDMs (MDM = infiltrating monocyte-derived
macrophages). Yet the GMM gate called ~1,600 MDM there -- a fabricated population
carved from the upper tail of noise. The control slice therefore proves the GMM
over-calls.

FIX: control-calibrated absolute thresholds.
  * Score every slice on ONE combined object (a common scale so a threshold
    transfers across slices).
  * Use the CONTROL slice as the negative reference. MDM should be ~0 there, so
    the MDM "+" bar = a high percentile of the control's MDM score -> the control
    yields ~1% MDM (near zero), and tumor slices only get MDM where cells truly
    exceed that background. Same idea (looser) for BAM. Microglia is the resident
    default (abundant in control), so its bar is a LOW control percentile.

Every "+" call requires positive enrichment (score >= 0); MDM/BAM add a high
control percentile on top for specificity.
Assignment (mutually exclusive; specific evidence overrides the microglia default):
  MDM  = mdm >= max(p97(control),0)  AND NOT bam_stable+   (most specific)
  BAM  = bam >= max(p97(control),0)                        (specific)
  Microglia = micro >= 0                                   (enriched above background)
  both MDM+ and BAM+ -> unknown ; neither + and not micro -> unknown

Output -> score_genes_slice{N}_merged/classify/myeloid_subtype_gate.png|csv
          score_genes_slice_all/myeloid_subtype_gate_counts.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, TUMOR_COL)
from myeloid_subtype_contrastive import broad_myeloid, sg, _dense  # noqa: E402

SLICES = [1, 2, 3]
CONTROL_SLICE = 3                 # non-tumor control -> defines the MDM/BAM null
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ALLOUT = "D:/thesis-research/score_genes_slice_all"

MODULES = {
    # positive ("should express"); pan-myeloid (Cx3cr1/C1q) and activation genes
    # (Trem2/Apoe/Spp1/Gpnmb/Cd9/Itgax/Lgals3) excluded per genes_instructions.txt.
    "micro":      ["TMEM119", "Adgrg1", "Hpgds", "Gpr183", "Selplg"],
    "bam":        ["Mrc1", "Cd163", "Pf4", "Maf", "Cd5l"],
    # disease-stable BAM veto: BAM-SPECIFIC only (Pf4/Maf). Mrc1/Cd163/Cd5l
    # excluded -- shared with / upregulated in activated microglia (line 18/40).
    "bam_stable": ["Pf4", "Maf"],
    "mdm":        ["Ccr2", "Plac8", "S100a8", "S100a9", "Vcan", "Sell", "Fn1",
                   "Crip1", "Clec12a", "Itgal", "Fpr1", "Ccr1"],
}
# Every "+" call requires POSITIVE enrichment above the expression-matched
# background (score >= FLOOR=0) -- a below-background score can never justify a
# call. On top of that floor, MDM/BAM use a high control percentile for extra
# specificity (anchored on the control null). Microglia gets the floor only
# (the control is microglia-rich, so it is not a null for microglia).
Q = {"bam": 0.97, "bam_stable": 0.97, "mdm": 0.97}
FLOOR = 0.0

AUDIT = ["TMEM119", "Adgrg1", "Mrc1", "Cd163", "Pf4", "Ccr2", "Plac8", "S100a8"]
SUBS = ["Microglia", "MDM", "BAM"]
COL = {"Microglia": "#17becf", "MDM": "#00a087", "BAM": "#d62728", "unknown": "#cccccc"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def load_myeloid(n):
    """Return the broad-Myeloid AnnData for slice n plus plotting coordinates."""
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _read_X(h5); var = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X); adata.var_names = pd.Index(var); adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    nt = adata[~tumor].copy(); cxnt, cynt = cx[~tumor], cy[~tumor]

    mask = broad_myeloid(nt, _dense(nt.X))
    mye = nt[mask].copy()
    mye.obs["slice"] = n
    mye.obs["cx"] = cxnt[mask]
    mye.obs["cy"] = cynt[mask]
    bg = {"cxnt": cxnt, "cynt": cynt, "tx": cx[tumor], "ty": cy[tumor]}
    print(f"slice {n}: non-tumor {nt.n_obs:,}, Myeloid {mye.n_obs:,}, tumor {int(tumor.sum()):,}")
    return mye, bg


def assign(scores, is_ctrl):
    """Control-calibrated hierarchical assignment. `scores` = dict module->array."""
    thr = {m: max(float(np.quantile(scores[m][is_ctrl], Q[m])), FLOOR) for m in Q}
    thr["micro"] = FLOOR                              # positive enrichment only
    micro_hit = scores["micro"] >= thr["micro"]
    bam_hit = scores["bam"] >= thr["bam"]
    bstab_hit = scores["bam_stable"] >= thr["bam_stable"]
    mdm_hit = (scores["mdm"] >= thr["mdm"]) & ~bstab_hit   # MDM must be Pf4/Maf-

    sub = np.full(len(micro_hit), "unknown", dtype=object)
    sub[micro_hit] = "Microglia"                    # resident default
    sub[bam_hit & ~mdm_hit] = "BAM"                 # specific evidence overrides
    sub[mdm_hit & ~bam_hit] = "MDM"                 # most specific
    sub[mdm_hit & bam_hit] = "unknown"              # ambiguous
    return sub, thr, {"micro": micro_hit, "bam": bam_hit, "mdm": mdm_hit}


def purity(mye, labels, n):
    rows = []
    m_slice = mye.obs["slice"].to_numpy() == n
    for s in SUBS + ["unknown"]:
        m = (labels == s) & m_slice
        r = {"slice": n, "subtype": s, "n": int(m.sum())}
        for g in AUDIT:
            r[g] = (round(float(_dense(mye[:, g].X)[m].mean()), 2)
                    if g in mye.var_names and m.sum() else np.nan)
        rows.append(r)
    return rows


def plot_slice(n, mye, labels, bg):
    out = f"D:/thesis-research/score_genes_slice{n}_merged/classify"
    os.makedirs(out, exist_ok=True)
    m_slice = mye.obs["slice"].to_numpy() == n
    lab = labels[m_slice]
    mxy = np.c_[mye.obs["cx"].to_numpy()[m_slice], mye.obs["cy"].to_numpy()[m_slice]]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=170)
    ax.scatter(bg["cxnt"], bg["cynt"], s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
    if len(bg["tx"]):
        ax.scatter(bg["tx"], bg["ty"], s=1.2, c="black", linewidths=0,
                   rasterized=True, label=f"tumor ({len(bg['tx']):,})")
    for s in ["unknown", "MDM", "Microglia", "BAM"]:
        m = lab == s
        if m.any():
            ax.scatter(mxy[m, 0], mxy[m, 1], s=4, c=COL[s], linewidths=0,
                       rasterized=True, label=f"{s} ({int(m.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    tag = " (CONTROL)" if n == CONTROL_SLICE else ""
    ax.set_title(f"slice {n}{tag} myeloid subtypes — control-calibrated gate",
                 fontweight="bold")
    ax.legend(loc="lower right", markerscale=3, fontsize=8)
    fig.savefig(f"{out}/myeloid_subtype_gate.png", bbox_inches="tight")
    plt.close(fig)


def main():
    assert CONTROL_SLICE in SLICES, "control slice must be among SLICES"
    os.makedirs(ALLOUT, exist_ok=True)

    myes, bgs = [], {}
    for n in SLICES:
        mye, bg = load_myeloid(n)
        myes.append(mye); bgs[n] = bg

    # combined -> ONE score scale so the control-derived threshold transfers
    comb = ad.concat(myes, join="inner", index_unique="-")
    scores = {m: sg(comb, MODULES[m], f"sc_{m}") for m in MODULES}
    slice_arr = comb.obs["slice"].to_numpy()
    is_ctrl = slice_arr == CONTROL_SLICE

    labels, thr, hits = assign(scores, is_ctrl)

    print(f"\ncontrol = slice {CONTROL_SLICE}; module thresholds:")
    for m in MODULES:
        lab = "abs>=0" if m == "micro" else f"p{int(Q[m]*100)}"
        frac = float((scores[m] >= thr[m])[is_ctrl].mean())
        print(f"  {m:11s} {lab:>6s} thr={thr[m]:+.3f}  control pass={frac:5.1%}")

    all_rows = []
    for n in SLICES:
        m_slice = slice_arr == n
        cnt = {s: int((labels[m_slice] == s).sum()) for s in SUBS + ["unknown"]}
        tag = " (CONTROL)" if n == CONTROL_SLICE else ""
        print(f"\nslice {n}{tag}: Myeloid {int(m_slice.sum()):,}  {cnt}")
        pur = pd.DataFrame(purity(comb, labels, n))
        print(pur.to_string(index=False))
        sdir = f"D:/thesis-research/score_genes_slice{n}_merged/classify"
        os.makedirs(sdir, exist_ok=True)
        pur.to_csv(f"{sdir}/myeloid_subtype_gate.csv", index=False)
        plot_slice(n, comb, labels, bgs[n])
        all_rows.append(pur)

    pd.concat(all_rows).to_csv(f"{ALLOUT}/myeloid_subtype_gate_counts.csv", index=False)
    print(f"\nsaved combined -> {ALLOUT}/myeloid_subtype_gate_counts.csv")

    # headline sanity check: MDM in the control must now be near zero
    ctrl_mdm = int((labels[is_ctrl] == "MDM").sum())
    print(f"\nCONTROL-SLICE SANITY: MDM in slice {CONTROL_SLICE} = {ctrl_mdm} "
          f"({ctrl_mdm / max(is_ctrl.sum(),1):.1%} of its myeloid)")


if __name__ == "__main__":
    main()
