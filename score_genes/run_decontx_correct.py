"""decontX ambient correction, split at the R boundary so you can run decontX in
your own R env.

Workflow:
  1. python run_decontx_correct.py export
        -> writes resources/cache/decontx/<slice>_work/{counts.mtx, clusters.csv}
  2. run decontX in R (directly):
        Rscript score_genes/run_decontx.R  D:/thesis-research/resources/cache/decontx
        (or open run_decontx.R in RStudio, set WORKDIR_PARENT, and source() it)
        -> writes decontx_counts.mtx + decontx_contamination.csv into each *_work
  3. python run_decontx_correct.py assemble
        -> writes resources/cache/decontx/<slice>_decontx.h5ad

  python run_decontx_correct.py          # 'auto': export -> Rscript subprocess -> assemble

decontX runs on the FULL slice (tumor + non-tumor) so the tumor/neuronal ambient
pool is modelled; coarse Leiden clusters are passed as z. Every gene is kept --
contamination is subtracted from the counts, not removed as features.
Then re-run run_myeloid_stage2_cluster.py with USE_DECONTX = True.
"""
import glob
import os
import shutil
import subprocess
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix

# All six slices are processed with the same grouping, so that the population prior
# supplied to decontX is constructed identically everywhere (earlier runs mixed a
# coarse Leiden partition on slices 1 and 3 with a 25-cluster partition elsewhere).
SLICES = {
    f"slice_{i}": f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{i}_adata.h5ad"
    for i in (1, 2, 3, 4, 5, 6)
}
OUT_DIR = "D:/thesis-research/resources/cache/decontx"
R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_decontx.R")
RSCRIPT = ""   # set to the full path of Rscript.exe to override autodetection (auto mode)
TUMOR_COL = "pred_tumor_XGBoost"
COARSE_RES = 0.5
CONTROL_PREFIXES = ("neg", "negprb", "blank", "falsecode", "systemcontrol")
WRITE_CLUSTERS = True       # supply a coarse Leiden partition as decontX's z prior
# Whole-slice Leiden exceeds available memory on slices 5 and 6; the cluster
# labels for those slices are provided as deposited inputs (clusters.csv).


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


def _read_var_names(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_obs_num(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_obs_bool(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def coarse_clusters(adata):
    """Quick whole-slice Leiden to hand decontX a population structure (z)."""
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars - 1))
    a = a[:, a.var.highly_variable].copy()          # subset -> avoid dense scale OOM
    sc.tl.pca(a, n_comps=min(30, a.n_vars - 1), zero_center=False)
    sc.pp.neighbors(a, n_neighbors=15)
    try:
        sc.tl.leiden(a, resolution=COARSE_RES, flavor="igraph",
                     n_iterations=2, directed=False)
    except Exception as e:
        print(f"  leiden(igraph) failed ({e}); default flavor")
        sc.tl.leiden(a, resolution=COARSE_RES)
    return a.obs["leiden"].astype(int).to_numpy()


def find_rscript():
    """Locate Rscript.exe: explicit override -> PATH -> standard Windows installs."""
    if RSCRIPT:
        return RSCRIPT
    exe = shutil.which("Rscript")
    if exe:
        return exe
    for pat in (
        r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-*\bin\Rscript.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\R\R-*\bin\x64\Rscript.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\R\R-*\bin\Rscript.exe"),
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "Rscript not found. Use the manual workflow (export -> run R yourself -> "
        "assemble), or set RSCRIPT to the full path of Rscript.exe."
    )


def workdir_for(name):
    return os.path.join(OUT_DIR, name + "_work")


def _load(path):
    """Load a slice, drop control probes (cells preserved & ordered)."""
    with h5py.File(path, "r") as h5:
        X = _read_X(h5)
        var = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var)
    adata.var_names_make_unique()
    keep = ~adata.var_names.str.lower().str.startswith(CONTROL_PREFIXES)
    adata = adata[:, keep].copy()
    return adata, cx, cy, tumor


def export_slice(name, path):
    work = workdir_for(name)
    os.makedirs(work, exist_ok=True)
    adata, cx, cy, tumor = _load(path)

    # decontX cannot decompose an all-zero cell (contamination = 0/0). Drop any,
    # and persist the kept-cell mask (over all cells) so assemble stays aligned.
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    keep = totals > 0
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"{name}: dropping {n_drop} zero-count cell(s) "
              f"({100 * n_drop / adata.n_obs:.2f}%) before decontX")
        adata = adata[keep].copy()
    np.save(os.path.join(work, "kept_cells.npy"), keep)

    mmwrite(os.path.join(work, "counts.mtx"), adata.X.T.tocsr().astype(np.int32))
    cfile = os.path.join(work, "clusters.csv")
    if WRITE_CLUSTERS:
        print(f"{name}: {adata.shape}; coarse Leiden for decontX z ...")
        z = coarse_clusters(adata)
        pd.DataFrame({"cluster": z}).to_csv(cfile, index=False)
    else:
        if os.path.exists(cfile):
            os.remove(cfile)   # no z -> decontX self-clusters in R
        print(f"{name}: {adata.shape}; wrote counts.mtx only (decontX self-clusters)")
    print(f"  exported -> {work}")


def assemble_slice(name, path):
    work = workdir_for(name)
    dec_path = os.path.join(work, "decontx_counts.mtx")
    if not os.path.exists(dec_path):
        raise FileNotFoundError(
            f"{dec_path} missing -- run decontX (R) on {OUT_DIR} first")
    adata, cx, cy, tumor = _load(path)
    keep = np.load(os.path.join(work, "kept_cells.npy"))   # cells dropped at export
    cx, cy, tumor = cx[keep], cy[keep], tumor[keep]
    var_names = adata.var_names
    n_kept = int(keep.sum())

    dec = mmread(dec_path).T.tocsr()            # cells x genes
    dec.data = np.rint(dec.data)
    dec.eliminate_zeros()
    if dec.shape != (n_kept, len(var_names)):
        raise ValueError(f"{name}: decontx shape {dec.shape} != "
                         f"expected {(n_kept, len(var_names))}")
    cont = pd.read_csv(os.path.join(work, "decontx_contamination.csv"))["contamination"].to_numpy()
    if len(cont) != n_kept:
        raise ValueError(f"{name}: contamination length {len(cont)} != kept cells {n_kept}")

    out = ad.AnnData(X=dec)
    out.var_names = var_names
    out.obs["CenterX_global_px"] = cx
    out.obs["CenterY_global_px"] = cy
    out.obs[TUMOR_COL] = tumor
    out.obs["decontx_contamination"] = cont
    out_path = os.path.join(OUT_DIR, f"{name}_decontx.h5ad")
    out.write_h5ad(out_path)
    print(f"{name}: kept {n_kept} cells, median contamination = "
          f"{np.median(cont):.3f}  ->  {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if mode in ("export", "auto"):
        for name, path in SLICES.items():
            export_slice(name, path)

    if mode == "export":
        print(f"\nExported. Now run decontX in your R env:\n"
              f"  Rscript score_genes/run_decontx.R {OUT_DIR}\n"
              f"  (or open score_genes/run_decontx.R in RStudio and source it)\n"
              f"Then: python score_genes/run_decontx_correct.py assemble")
        return

    if mode == "auto":
        rscript = find_rscript()
        print(f"running decontX via {rscript} ...")
        subprocess.run([rscript, R_SCRIPT, OUT_DIR], check=True)

    if mode in ("assemble", "auto"):
        for name, path in SLICES.items():
            assemble_slice(name, path)


if __name__ == "__main__":
    main()
