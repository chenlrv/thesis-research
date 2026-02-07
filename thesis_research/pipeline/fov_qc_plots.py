import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from thesis_research.pipeline.position_plots import plot_fov_positions
from thesis_research.utils.columns import AdataObs, SAMPLE_ID
from thesis_research.utils.entity_type import get_output_dir


def run_fov_qc(adata: AnnData, run_id: str) -> AnnData:
    sample_id = adata.uns[SAMPLE_ID]
    output_dir = get_output_dir(run_id, sample_id)

    barcodemap = pd.read_csv("/resources/barcodemap_Mm_UCC.csv")
    barcodemap["gene"] = barcodemap["gene"].astype(str)
    barcodemap["barcode"] = barcodemap["barcode"].astype(str)

    xy = adata.obs[[AdataObs.CENTER_X_GLOBAL_PX, AdataObs.CENTER_Y_GLOBAL_PX]].to_numpy()
    fov = adata.obs[AdataObs.FOV].to_numpy()

    grid_id, grid_fov = make_grid(xy=xy, fov=fov, squares_per_fov=49, min_cells_per_square=10)

    keep_frac = np.mean(np.array(grid_id, dtype=object) != None)
    print("GRID: cells kept in valid squares:", keep_frac)

    grid_ids_unique = pd.unique(pd.Series(grid_id).dropna())
    print("GRID: #valid grid squares:", len(grid_ids_unique))
    print("GRID: #unique FOVs:", pd.Series(fov).nunique())
    print("GRID: grid_fov mapping size:", len(grid_fov))

    grid_ids, bitcounts, bit_names = grid_bitcounts_from_adata(
        adata=adata, grid_id=grid_id, barcodemap_df=barcodemap
    )

    print("\nBARCODES: barcode length counts (top):")
    print(barcodemap["barcode"].str.len().value_counts().head())

    chars = set("".join(barcodemap["barcode"].astype(str).tolist()))
    print("BARCODES: unique chars:", chars)

    overlap = len(set(barcodemap["gene"]) & set(adata.var_names.astype(str)))
    print("BARCODES: gene overlap with adata:", overlap, "out of", adata.n_vars)

    print("\nBITCOUNTS: shape:", bitcounts.shape, "(n_grids, n_bits)")
    print("BITCOUNTS: n_bits from names:", len(bit_names))

    rs = bitcounts.sum(axis=1)
    print("BITCOUNTS: row-sum mean/std/min/max:", rs.mean(), rs.std(), rs.min(), rs.max())
    print("BITCOUNTS: row-sum CV (std/mean):", (rs.std() / rs.mean()) if rs.mean() != 0 else np.nan)

    zero_rows = rs == 0
    print("BITCOUNTS: zero rows fraction:", zero_rows.mean(), "count:", zero_rows.sum())

    col_sums = bitcounts.sum(axis=0)
    print("BITCOUNTS: bits with zero total:", int((col_sums == 0).sum()), "out of", len(col_sums))

    grid_fov_vec = np.array([grid_fov[g] for g in grid_ids], dtype=object)

    comparators = get_nearest_neighbors_by_fov(
        x=bitcounts, grid_fov_vec=grid_fov_vec, n_neighbors=10
    )

    print("\nCOMPARATORS: shape:", comparators.shape)
    print("COMPARATORS: missing fraction (=-1):", (comparators == -1).mean())

    bad = 0
    total = 0
    for i in range(comparators.shape[0]):
        nbrs = comparators[i]
        nbrs = nbrs[nbrs >= 0]
        if nbrs.size == 0:
            continue
        bad += int(np.sum(grid_fov_vec[nbrs] == grid_fov_vec[i]))
        total += int(nbrs.size)
    print("COMPARATORS: same-FOV comparator count (should be 0):", bad, "out of", total)

    (
        totalcounts_grid,
        totalcountshat,
        totalcounts_resids,
        flagged_grids,
        flagged_fovs_fortotalcounts,
        frac_flagged_by_fov,
    ) = fov_qc_totalcounts(
        X=adata.X,
        grid_id=grid_id,
        grid_ids=grid_ids,
        grid_fov_vec=grid_fov_vec,
        comparators=comparators,
        max_totalcounts_loss=0.6,
    )

    print("Flagged FOVs for total-count loss:", len(flagged_fovs_fortotalcounts))
    print(flagged_fovs_fortotalcounts[:20])
    print("\nTop 10 FOVs by fraction flagged:")
    print(frac_flagged_by_fov.sort_values(ascending=False).head(10))

    resid = compute_bit_residuals(bitcounts, comparators)
    print("resid shape:", resid.shape, "min/max:", float(np.min(resid)), float(np.max(resid)))

    max_prop_loss = 0.6
    fovstats = summarize_fov_bias(resid, grid_fov_vec, bit_names, max_prop_loss=max_prop_loss)

    print("fovstats['flag'] shape:", fovstats["flag"].shape)
    print("total flagged (fov,bit) pairs:", int(fovstats["flag"].to_numpy().sum()))

    flags_per_fov_x_reportercycle, flagged_fovs_forbias = flags_per_fov_reportercycle_and_fovs(
        fovstats["flag"]
    )

    print("\nFlagged FOVs for barcode/reportercycle bias:", len(flagged_fovs_forbias))
    print("first 30:", flagged_fovs_forbias[:30])

    print("\nTop failing reporter cycles (count of FOVs with >=0.5 flagged bits):")
    print(
        ((flags_per_fov_x_reportercycle >= 0.5).sum(axis=0)).sort_values(ascending=False).head(15)
    )

    flagged_fovs = sorted(set(flagged_fovs_fortotalcounts) | set(flagged_fovs_forbias))

    plot_fov_positions(
        sample_id=sample_id,
        output_dir=output_dir,
        faulty_fovs=flagged_fovs,
    )

    new_adata = adata.copy()
    return new_adata[[~AdataObs.FOV in flagged_fovs]]


def make_grid(
    xy: np.ndarray,
    fov: np.ndarray,
    squares_per_fov: int = 49,
    min_cells_per_square: int = 10,
) -> tuple[np.ndarray, dict]:
    """
    Split each FOV into a sqrt(squares_per_fov) × sqrt(squares_per_fov) grid.

    Parameters
    ----------
    xy : (n_cells, 2) array
        Cell coordinates in pixels (X, Y) (global pixel positions)
    fov : (n_cells,) array
        FOV id per cell
    squares_per_fov : int
        49 means 7×7.
    min_cells_per_square : int
        Squares with fewer cells are marked invalid.

    Returns
    -------
    grid_id : (n_cells,) int array
        Grid square id for each cell; -1 = invalid / dropped square.
    grid_fov : dict
        Mapping from grid square ID -> fov ID
    """
    xy = np.asarray(xy)
    fov = np.asarray(fov).astype(str)

    ncuts = int(np.floor(np.sqrt(squares_per_fov)))  # 7 for 49
    grid_id = np.array([None] * len(fov), dtype=object)

    x = xy[:, 0]
    y = xy[:, 1]

    # Assign bins within each FOV separately
    for f in np.unique(fov):
        idx = fov == f
        if idx.sum() == 0:
            continue

        xb = pd.cut(x[idx], bins=ncuts, include_lowest=True)
        yb = pd.cut(y[idx], bins=ncuts, include_lowest=True)

        # Build grid IDs as strings
        grid_id[idx] = (
            pd.Series(fov[idx]) + "_" + xb.astype(str) + "_" + yb.astype(str)
        ).to_numpy()

    # Drop grid squares with too few cells
    vc = pd.Series(grid_id).value_counts(dropna=True)
    too_small = set(vc[vc < min_cells_per_square].index)
    grid_id = np.array(
        [g if (g is not None and g not in too_small) else None for g in grid_id], dtype=object
    )

    # Build mapping grid square -> FOV
    grid_fov = {}
    for g in pd.unique(pd.Series(grid_id).dropna()):
        grid_fov[g] = g.split("_", 1)[0]

    return grid_id, grid_fov


import numpy as np


def get_colorvals(barcodes):
    """
    Equivalent of R getcolorvals(): unique non-dot characters across all barcodes.
    """
    all_chars = set("".join(map(str, barcodes)))
    all_chars.discard(".")
    ordered = []
    seen = set()
    for bc in map(str, barcodes):
        for ch in bc:
            if ch == "." or ch in seen:
                continue
            seen.add(ch)
            ordered.append(ch)
    return ordered


def barcode_to_bitmatrix(barcodes: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Equivalent of R barcode2bitmatrix().

    Parameters
    ----------
    barcodes : array-like of str, length n_genes
        CosMx barcode strings (same length, includes '.')

    Returns
    -------
    bitmat : (n_genes, n_bits) np.ndarray float32
        1 if gene uses that bit (cycle+color), else 0
    bit_names : list[str]
        Column names like 'reportercycle12R'
    """
    barcodes = np.asarray(barcodes, dtype=str)
    if barcodes.size == 0:
        raise ValueError("barcodes is empty")

    L = len(barcodes[0])
    if any(len(bc) != L for bc in barcodes):
        raise ValueError("All barcodes must have the same length")

    colorvals = get_colorvals(barcodes)  # typically 4 values
    nreportercycles = L // 2
    if nreportercycles * 2 != L:
        raise ValueError("Barcode length must be even (2 chars per reporter cycle).")

    if len(colorvals) != 4:
        print("Warning: Unexpected number of colors in barcodes:", colorvals)
        # CosMx usually has 4 colors, but we'll not hard-fail; just warn by raising informative error.
        # You can relax this if your panels differ.
        pass

    # column names match R:
    bit_names = []
    for i in range(1, nreportercycles + 1):
        for c in colorvals:
            bit_names.append(f"reportercycle{i}{c}")

    nbits = nreportercycles * len(colorvals)
    bitmat = np.zeros((len(barcodes), nbits), dtype=np.float32)

    for i in range(1, nreportercycles + 1):
        pos = i * 2 - 1
        bc_here = np.array([bc[pos] for bc in barcodes])

        for c_i, c in enumerate(colorvals):
            col = (i - 1) * len(colorvals) + c_i
            bitmat[bc_here == c, col] = 1.0

    return bitmat, bit_names


from scipy import sparse


def grid_mean_gene_expression(X: sparse.csr_matrix, grid_id):
    """
    Aggregate a cell×gene count matrix into grid×gene mean expression.

    Parameters
    ----------
    X : csr_matrix, shape (n_cells, n_genes)
        Raw counts (adata.X).
    grid_id : array-like length n_cells
        Grid square label per cell; None means drop cell (like NA in R).

    Returns
    -------
    grid_ids : np.ndarray of str, shape (n_grids,)
        Unique valid grid square IDs.
    grid_gene_mean : csr_matrix, shape (n_grids, n_genes)
        Mean expression per gene for each grid square.
    """
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    grid_id = np.asarray(grid_id, dtype=object)
    keep = grid_id != None

    Xk = X[keep]
    gids = grid_id[keep]

    # Map each grid label -> row index 0..n_grids-1
    grid_ids = pd.unique(gids)
    gid_to_i = {g: i for i, g in enumerate(grid_ids)}

    gi = np.fromiter((gid_to_i[g] for g in gids), dtype=int, count=len(gids))
    ci = np.arange(len(gids), dtype=int)

    # gridmap: (n_grids x n_cells_kept), with 1 if cell belongs to grid
    gridmap = sparse.csr_matrix(
        (np.ones(len(ci), dtype=np.float32), (gi, ci)), shape=(len(grid_ids), len(gids))
    )

    # Sum counts per grid square
    grid_gene_sum = gridmap @ Xk  # stays sparse

    # Convert to mean by dividing each row by number of cells in that grid
    ncells = np.asarray(gridmap.sum(axis=1)).ravel()
    inv = sparse.diags(1.0 / np.maximum(ncells, 1.0))
    grid_gene_mean = (inv @ grid_gene_sum).tocsr()

    return np.asarray(grid_ids, dtype=object), grid_gene_mean


def build_gene_bit_matrix(var_names, barcodemap_df):
    """
    Build a gene×bit sparse matrix aligned to adata.var_names.

    Parameters
    ----------
    var_names : array-like of str
        adata.var_names
    barcodemap_df : pd.DataFrame
        Must have columns ['gene', 'barcode'].

    Returns
    -------
    gene_bit : csr_matrix, shape (n_genes_in_adata, n_bits)
        Rows aligned to var_names; genes not in barcodemap are all-zero rows.
    bit_names : list[str]
        Column names like 'reportercycle12R'
    """
    var_names = pd.Index(var_names.astype(str))
    bm = barcodemap_df.copy()
    bm["gene"] = bm["gene"].astype(str)
    bm["barcode"] = bm["barcode"].astype(str)

    # Keep only genes present in adata
    bm = bm[bm["gene"].isin(var_names)]
    if bm.shape[0] == 0:
        raise ValueError("No overlap between barcodemap genes and adata.var_names")

    # Build bit matrix for the overlapped genes (dense small-ish), then place into full matrix
    bitmat_dense, bit_names = barcode_to_bitmatrix(bm["barcode"].to_numpy())

    # Map bm genes -> row indices in adata
    rows = var_names.get_indexer(bm["gene"])
    # bitmat_dense is (n_overlap, n_bits): convert to sparse
    gene_bit_overlap = sparse.csr_matrix(bitmat_dense)

    # Place overlap rows into a full (n_genes, n_bits) sparse matrix
    gene_bit_full = sparse.lil_matrix((len(var_names), gene_bit_overlap.shape[1]), dtype=np.float32)
    gene_bit_full[rows, :] = gene_bit_overlap
    gene_bit_full = gene_bit_full.tocsr()

    return gene_bit_full, bit_names


def grid_bitcounts_from_adata(adata, grid_id, barcodemap_df):
    """
    Equivalent to R cellxgene2squarexbit() + the row-normalization step,
    using adata.X and adata.var_names.

    Returns
    -------
    grid_ids : np.ndarray (n_grids,)
    bitcounts : np.ndarray (n_grids, n_bits) float32
    bit_names : list[str]
    """
    # 1) grid×gene mean
    grid_ids, grid_gene_mean = grid_mean_gene_expression(adata.X, grid_id)

    # 2) gene×bit aligned to adata.var_names
    gene_bit, bit_names = build_gene_bit_matrix(adata.var_names.to_numpy(), barcodemap_df)

    # 3) grid×bit = (grid×gene) @ (gene×bit)
    grid_bit = (grid_gene_mean @ gene_bit).tocsr()

    # 4) row-normalize like R
    row_sums = np.asarray(grid_bit.sum(axis=1)).ravel().astype(np.float32)
    mean_sum = float(np.mean(row_sums[row_sums > 0])) if np.any(row_sums > 0) else 1.0

    inv = np.zeros_like(row_sums, dtype=np.float32)
    inv[row_sums > 0] = 1.0 / row_sums[row_sums > 0]
    Dinv = sparse.diags(inv)

    bitcounts = (Dinv @ grid_bit).multiply(mean_sum)  # keep sparse for a moment
    bitcounts = bitcounts.toarray().astype(np.float32)

    return grid_ids, bitcounts, bit_names


def get_nearest_neighbors_by_fov(x, grid_fov_vec, n_neighbors=10):
    """
    Python port of R getNearestNeighborsByFOV().

    Returns:
      comparators: (n_grids, n_neighbors) int array
                   each row lists comparator grid indices, from OTHER FOVs,
                   with max 1 comparator per other FOV. -1 means missing.
    """
    grid_fov_vec = np.asarray(grid_fov_vec, dtype=object)

    k0 = min(n_neighbors * 50, x.shape[0])  # same as R
    nn = NearestNeighbors(n_neighbors=k0, metric="euclidean")
    nn.fit(x)
    top = nn.kneighbors(x, return_distance=False)

    comparators = np.full((x.shape[0], n_neighbors), -1, dtype=int)

    for i in range(x.shape[0]):
        seen_fovs = {grid_fov_vec[i]}  # exclude own FOV
        picked = []
        for j in top[i]:
            f = grid_fov_vec[j]
            if f in seen_fovs:
                continue
            seen_fovs.add(f)  # ensures max 1 per other FOV
            picked.append(j)
            if len(picked) == n_neighbors:
                break
        if picked:
            comparators[i, : len(picked)] = picked

    return comparators


from scipy import sparse


def fov_qc_totalcounts(
    X,  # adata.X (csr_matrix)
    grid_id,  # per-cell grid label (None dropped)
    grid_ids,  # per-grid labels in same order as bitcounts/comparators
    grid_fov_vec,  # per-grid FOV (aligned to grid_ids)
    comparators,  # (n_grids, 10) comparator grid indices
    max_totalcounts_loss=0.6,
):
    """
    Port of the 'Search for FOVs with low outlier total counts' block in runFOVQC().

    Returns
    -------
    totalcounts_grid : pd.Series(index=grid_id_label)
        Mean total counts per cell for each grid square.
    totalcountshat : np.ndarray shape (n_grids,)
        Expected total counts per grid from comparator grids.
    totalcounts_resids : pd.Series(index=grid_id_label)
        log2(obs/expected) per grid.
    flagged_grids : pd.Series(index=grid_id_label)
        Boolean per grid: residual below threshold.
    flagged_fovs_fortotalcounts : list
        FOV IDs failing the >75% rule.
    frac_flagged_by_fov : pd.Series(index=fov)
        Fraction of grids flagged per FOV.
    """
    # per-cell total counts
    if sparse.issparse(X):
        totalcounts_cell = np.asarray(X.sum(axis=1)).ravel()
    else:
        totalcounts_cell = np.sum(X, axis=1)

    # per-grid mean of totalcounts (ignore dropped cells)
    grid_id = np.asarray(grid_id, dtype=object)
    keep = grid_id != None
    tc_kept = totalcounts_cell[keep]
    gid_kept = grid_id[keep]

    totalcounts_grid = pd.Series(tc_kept).groupby(gid_kept).mean()
    totalcounts_grid = totalcounts_grid.reindex(grid_ids)  # align to grid row order

    obs = totalcounts_grid.to_numpy(dtype=float)

    # expected per grid = mean of comparator grids (comparators are guaranteed valid indices)
    totalcountshat = np.empty_like(obs, dtype=float)
    for i in range(len(obs)):
        totalcountshat[i] = np.mean(obs[comparators[i]])

    # residual log2(obs/exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        resids = np.log2(obs / totalcountshat)

    totalcounts_resids = pd.Series(resids, index=grid_ids)

    # flagging
    threshold = np.log2(1.0 - max_totalcounts_loss)
    flagged_grids = totalcounts_resids < threshold

    # per-FOV fraction flagged
    frac_flagged_by_fov = (
        pd.Series(flagged_grids.to_numpy(dtype=float)).groupby(grid_fov_vec).mean()
    )

    flagged_fovs_fortotalcounts = frac_flagged_by_fov.index[frac_flagged_by_fov > 0.75].tolist()

    return (
        totalcounts_grid,
        totalcountshat,
        totalcounts_resids,
        flagged_grids,
        flagged_fovs_fortotalcounts,
        frac_flagged_by_fov,
    )


import numpy as np


def compute_bit_residuals(bitcounts: np.ndarray, comparators: np.ndarray) -> np.ndarray:
    """
    Equivalent of:
      yhat <- colMeans(bitcounts[comparators[i,], ])
      resid <- log2((bitcounts+1)/(yhat+1))
    Returns resid with shape (n_grids, n_bits)
    """
    n_grids, n_bits = bitcounts.shape
    yhat = np.empty_like(bitcounts, dtype=np.float32)

    for i in range(n_grids):
        yhat[i, :] = bitcounts[comparators[i]].mean(axis=0)

    resid = np.log2((bitcounts + 1.0) / (yhat + 1.0)).astype(np.float32)
    return resid


import pandas as pd
from scipy import stats


def summarize_fov_bias(
    resid: np.ndarray, grid_fov_vec: np.ndarray, bit_names: list, max_prop_loss: float = 0.6
):
    """
    Returns dict with:
      flag: DataFrame (fov x bit) boolean
      bias: DataFrame (fov x bit) float (mean resid per fov)
      p:    DataFrame (fov x bit) float (ttest of mean vs 0)
      propagree: DataFrame (fov x bit) float
    """
    grid_fov_vec = np.asarray(grid_fov_vec, dtype=object)
    fovs = pd.unique(grid_fov_vec)

    bias = pd.DataFrame(index=fovs, columns=bit_names, dtype=float)
    pval = pd.DataFrame(index=fovs, columns=bit_names, dtype=float)
    propagree = pd.DataFrame(index=fovs, columns=bit_names, dtype=float)

    # exactly like R:
    agree_thresh = np.log2(1.0 - max_prop_loss) / 3.0
    bias_thresh = np.log2(1.0 - max_prop_loss)

    # precompute indices per fov (speed)
    fov_to_idx = {f: np.where(grid_fov_vec == f)[0] for f in fovs}

    for b, bit in enumerate(bit_names):
        for f in fovs:
            idx = fov_to_idx[f]
            vals = resid[idx, b]
            if vals.size == 0:
                continue

            mu = float(np.mean(vals))
            bias.loc[f, bit] = mu

            tstat, p = stats.ttest_1samp(vals, popmean=0.0, nan_policy="omit")
            pval.loc[f, bit] = float(p) if np.isfinite(p) else np.nan

            propagree.loc[f, bit] = float(np.mean(vals < agree_thresh))

    flag = (bias < bias_thresh) & (pval < 0.01) & (propagree >= 0.5)
    return {"flag": flag, "bias": bias, "p": pval, "propagree": propagree}


def flags_per_fov_reportercycle_and_fovs(fov_flag_df: pd.DataFrame):
    """
    Equivalent to the R collapsing:
      reportercycle <- substr(bitname, 1, nchar(bitname)-1)
      flags_per_fov_x_reportercycle[fov, rc] = rowMeans(flag bits in rc)
      flagged_fovs_forbias = fovs with any rc >= 0.5
    """
    bit_names = list(fov_flag_df.columns)
    reportercycle = [bn[:-1] for bn in bit_names]  # drop last char (color)

    cycles = sorted(set(reportercycle))
    out = pd.DataFrame(index=fov_flag_df.index, columns=cycles, dtype=float)

    for rc in cycles:
        cols = [bn for bn, rcc in zip(bit_names, reportercycle) if rcc == rc]
        out[rc] = fov_flag_df[cols].mean(axis=1)

    flagged_fovs_forbias = out.index[(out >= 0.5).any(axis=1)].tolist()
    return out, flagged_fovs_forbias
