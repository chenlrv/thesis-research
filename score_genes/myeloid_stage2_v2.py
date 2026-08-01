"""Stage-2 myeloid subtyping (Microglia / BAM / MDM) on the L321 slide.

Calibration: slices 1,2 (tumor) are calibrated against slice 3 (CONTROL, same
slide L321 -> no cross-slide batch effect).

Design (agreed):
  * Input = the Stage-1 v2 pan-myeloid cells (celltype_v2 == "Myeloid").
  * Standard probes only -- TMEM119/Lyve1 (custom, unreliable) dropped. Microglia
    is scored POSITIVELY on standard homeostatic genes, NOT a leftover default.
  * Score all slices on ONE combined object -> common scale so a control-derived
    threshold transfers.
  * Control-calibrated positive gates:
      - MDM  : control is the NULL (no infiltration) -> high control percentile.
      - BAM  : residents exist in control -> high control percentile (positive tail).
      - micro: control is microglia-rich -> LOW control percentile (admit most).
  * Anti-ambient coverage gate: winning subtype must detect >= k own markers,
    k = p10 of control cells passing that module's score gate (>=1 floor).
  * Assignment: BAM/MDM (specific) override microglia; BAM & MDM -> unresolved;
    only-micro -> Microglia; none -> unresolved.

Output -> score_genes_slice_all/myeloid_stage2_v2_counts.csv
          score_genes_slice{n}_v2/myeloid_stage2_v2.png
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix

SLICES = [1, 2, 3]
CONTROL_SLICE = 3
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
V2 = "D:/thesis-research/score_genes_slice{}_v2/cell_scores.csv"
ALLOUT = "D:/thesis-research/score_genes_slice_all"
TUMOR_COL = "pred_tumor_XGBoost"

MODULES = {
    "micro": ["Adgrg1", "Hpgds", "Gpr183"],
    "bam":   ["Mrc1", "Cd163", "Pf4", "Maf", "Cd5l"],
    # data-driven: keep MDM genes with tumor/control ratio >=1.3 in myeloid
    # (the S100a8/9/Vcan/Fn1/Itgal set carries the strongest infiltration signal;
    # Clec12a 0.89x and Sell 1.23x dropped as flat).
    "mdm":   ["Ccr2", "Plac8", "Cd14", "Ccr1", "Fpr1", "Crip1",
              "S100a8", "S100a9", "Vcan", "Fn1", "Itgal"],
}
BAM_STABLE = ["Pf4", "Maf"]                 # disease-stable BAM veto for MDM
# control percentile for each module's positive bar
Q = {"micro": 0.10, "bam": 0.90, "mdm": 0.97}
SUBS = ["Microglia", "BAM", "MDM"]
COL = {"Microglia": "#17becf", "BAM": "#d62728", "MDM": "#00a087",
       "unresolved": "#cccccc"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_X(h5):
    node = h5["X"]
    if isinstance(node, h5py.Group):
        enc = str(node.attrs.get("encoding-type", ""))
        shape = tuple(node.attrs["shape"])
        data, idx, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
        M = csc_matrix((data, idx, indptr), shape=shape) if "csc" in enc \
            else csr_matrix((data, idx, indptr), shape=shape)
        return M.tocsr()
    return csr_matrix(node[...])


def _read_var(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_num(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_bool(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def sg(adata, genes, name):
    g = [x for x in genes if x in adata.var_names]
    sc.tl.score_genes(adata, gene_list=g, score_name=name, ctrl_size=50,
                      n_bins=25, random_state=0)
    return adata.obs[name].to_numpy()


def load_myeloid(n):
    """Stage-1 v2 myeloid cells for slice n: normalized AnnData + raw coverage + coords."""
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _read_X(h5); var = list(_read_var(h5))
        cx = _read_num(h5, "CenterX_global_px")
        cy = _read_num(h5, "CenterY_global_px")
        tumor = _read_bool(h5, TUMOR_COL)
    keep = ~tumor
    Xk = X[keep].tocsr()
    del X
    df = pd.read_csv(V2.format(n))
    assert len(df) == keep.sum(), f"slice {n}: v2 rows {len(df)} != non-tumor {keep.sum()}"
    mye_mask = df["celltype_v2"].to_numpy() == "Myeloid"
    Xm = Xk[mye_mask]
    # raw coverage per module (detection count) BEFORE normalization
    vidx = {g: i for i, g in enumerate(var)}
    cov = {}
    for m, genes in MODULES.items():
        cols = [vidx[g] for g in genes if g in vidx]
        cov[m] = np.asarray((Xm[:, cols] > 0).sum(1)).ravel()
    adata = ad.AnnData(X=Xm.astype(np.float32))
    adata.var_names = pd.Index(var); adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    adata.obs["slice"] = n
    adata.obs["cx"] = cx[keep][mye_mask]
    adata.obs["cy"] = cy[keep][mye_mask]
    for m in MODULES:
        adata.obs[f"cov_{m}"] = cov[m]
    bg = {"cxnt": cx[keep], "cynt": cy[keep], "tx": cx[tumor], "ty": cy[tumor]}
    print(f"slice {n}: non-tumor {int(keep.sum()):,}, v2-Myeloid {int(mye_mask.sum()):,}, "
          f"tumor {int(tumor.sum()):,}")
    return adata, bg


def main():
    os.makedirs(ALLOUT, exist_ok=True)
    myes, bgs = [], {}
    for n in SLICES:
        a, bg = load_myeloid(n); myes.append(a); bgs[n] = bg
    comb = ad.concat(myes, join="inner", index_unique="-")
    is_ctrl = comb.obs["slice"].to_numpy() == CONTROL_SLICE

    scores = {m: sg(comb, MODULES[m], f"s_{m}") for m in MODULES}
    scores["bam_stable"] = sg(comb, BAM_STABLE, "s_bstab")
    cov = {m: comb.obs[f"cov_{m}"].to_numpy() for m in MODULES}

    # control-calibrated score bars
    bar = {m: float(np.quantile(scores[m][is_ctrl], Q[m])) for m in MODULES}
    # anti-ambient coverage k = p10 of control cells passing the module score gate
    kcov = {}
    for m in MODULES:
        cc = cov[m][is_ctrl & (scores[m] >= bar[m])]
        kcov[m] = int(max(1, np.floor(np.percentile(cc, 10)))) if cc.size else 1

    print("\nmodule   Q     score_bar   ctrl_pass   k_cov")
    for m in MODULES:
        cp = float((scores[m] >= bar[m])[is_ctrl].mean())
        print(f"  {m:6s} p{int(Q[m]*100):>2} {bar[m]:>+9.3f} {cp:>10.1%} {kcov[m]:>7d}")

    hit = {m: (scores[m] >= bar[m]) & (cov[m] >= kcov[m]) for m in MODULES}
    bstab = scores["bam_stable"] >= float(np.quantile(scores["bam_stable"][is_ctrl], 0.97))
    mdm_hit = hit["mdm"] & ~bstab                      # MDM must be Pf4/Maf-negative
    bam_hit = hit["bam"]
    micro_hit = hit["micro"]

    lab = np.full(comb.n_obs, "unresolved", dtype=object)
    lab[micro_hit] = "Microglia"                       # positive microglia call
    lab[bam_hit & ~mdm_hit] = "BAM"                    # specific overrides micro
    lab[mdm_hit & ~bam_hit] = "MDM"
    lab[bam_hit & mdm_hit] = "unresolved"              # conflict

    # --- VALIDATION: are the 3 axes distinct populations or one activation axis? ---
    from itertools import combinations

    from scipy.stats import spearmanr
    print("\n=== module-score Spearman corr (all myeloid) — high +corr => NOT distinct ===")
    for a, b in combinations(["micro", "bam", "mdm"], 2):
        r = spearmanr(scores[a], scores[b]).correlation
        print(f"  {a:5s} vs {b:5s}: {r:+.3f}")
    print("\n=== mean module score by assigned subtype (diag should dominate its row) ===")
    print(f"{'subtype':11s} {'n':>6s} {'s_micro':>8s} {'s_bam':>8s} {'s_mdm':>8s}")
    for s in SUBS + ["unresolved"]:
        m = lab == s
        if m.any():
            print(f"{s:11s} {int(m.sum()):>6d} {scores['micro'][m].mean():>+8.3f} "
                  f"{scores['bam'][m].mean():>+8.3f} {scores['mdm'][m].mean():>+8.3f}")

    slice_arr = comb.obs["slice"].to_numpy()
    rows = []
    for n in SLICES:
        m = slice_arr == n
        cnt = {s: int((lab[m] == s).sum()) for s in SUBS + ["unresolved"]}
        tot = int(m.sum())
        tag = " (CONTROL)" if n == CONTROL_SLICE else ""
        print(f"\nslice {n}{tag}: myeloid {tot:,}  " +
              "  ".join(f"{s}={cnt[s]} ({100*cnt[s]/max(tot,1):.0f}%)" for s in SUBS + ["unresolved"]))
        rows.append({"slice": n, "myeloid": tot, **cnt})
        plot_slice(n, comb, lab, bgs[n])
        # persist per-cell labels (cx/cy carried for an alignment cross-check downstream)
        pd.DataFrame({"cx": comb.obs["cx"].to_numpy()[m],
                      "cy": comb.obs["cy"].to_numpy()[m],
                      "subtype": lab[m]}).to_csv(
            f"D:/thesis-research/score_genes_slice{n}_v2/myeloid_stage2_labels.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{ALLOUT}/myeloid_stage2_v2_counts.csv", index=False)

    ctrl_mdm = int((lab[is_ctrl] == "MDM").sum())
    print(f"\nCONTROL SANITY: MDM in slice {CONTROL_SLICE} = {ctrl_mdm} "
          f"({ctrl_mdm/max(is_ctrl.sum(),1):.1%} of control myeloid) -- should be ~0")
    print(f"saved -> {ALLOUT}/myeloid_stage2_v2_counts.csv")


def plot_slice(n, comb, lab, bg):
    out = f"D:/thesis-research/score_genes_slice{n}_v2"
    m = comb.obs["slice"].to_numpy() == n
    cx = comb.obs["cx"].to_numpy()[m]; cy = comb.obs["cy"].to_numpy()[m]; ll = lab[m]
    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    ax.scatter(bg["cxnt"], bg["cynt"], s=0.5, c="#eee", linewidths=0, rasterized=True)
    if len(bg["tx"]):
        ax.scatter(bg["tx"], bg["ty"], s=1.2, c="black", linewidths=0, rasterized=True,
                   label=f"tumor ({len(bg['tx']):,})")
    for s in ["unresolved", "Microglia", "BAM", "MDM"]:
        mm = ll == s
        if mm.any():
            ax.scatter(cx[mm], cy[mm], s=5, c=COL[s], linewidths=0, rasterized=True,
                       label=f"{s} ({int(mm.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    tag = " (CONTROL)" if n == CONTROL_SLICE else ""
    ax.set_title(f"slice {n}{tag} myeloid subtypes — control(s{CONTROL_SLICE})-calibrated")
    ax.legend(loc="lower right", markerscale=3, fontsize=8)
    fig.savefig(f"{out}/myeloid_stage2_v2.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
