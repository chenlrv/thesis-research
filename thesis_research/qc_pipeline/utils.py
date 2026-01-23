from anndata import AnnData


def q(x, percentile):
    return x.quantile(percentile)


def subset_adata_by_fovs(adata: AnnData, fovs: list[str]) -> AnnData:
    fovs = set(map(str, fovs))
    mask = adata.obs["fov"].astype(str).isin(fovs).to_numpy()
    return adata[mask].copy()
