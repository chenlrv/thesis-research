import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import h5py, numpy as np, pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
PAN=["C1qa","C1qb","C1qc","Csf1r","Aif1","Tyrobp","Fcer1g"]; LIN=["Csf1r","Aif1","Tyrobp","Fcer1g"]
def dec(a): return np.array([x.decode() if isinstance(x,bytes) else x for x in a]) if a.dtype.kind in ("O","S") else a
def X(h):
    n=h["X"];e=str(n.attrs.get("encoding-type",""));s=tuple(n.attrs["shape"])
    a=(n["data"][...],n["indices"][...],n["indptr"][...])
    return (csc_matrix(a,shape=s) if "csc" in e else csr_matrix(a,shape=s)).tocsr()
def var(h):
    v=h["var"];k=v.attrs.get("_index","_index");return list(dec(v[k.decode() if isinstance(k,bytes) else k][...]))
def b(h,c):
    nd=h["obs"][c]
    if isinstance(nd,h5py.Group):
        cd=nd["codes"][...];ct=dec(nd["categories"][...])
        return np.isin(np.where(cd>=0,ct[np.clip(cd,0,None)],"False").astype(str),["True","1","1.0","TRUE","true"])
    ar=nd[...]; return np.isin(dec(ar).astype(str),["True","1"]) if ar.dtype.kind in ("S","O") else ar.astype(bool)
for sl in [1,2,3]:
    with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
        Xk=X(h)[~b(h,"pred_tumor_XGBoost")];v=var(h)
    raw={g:np.asarray(Xk[:,v.index(g)].todense()).ravel() for g in set(PAN+LIN)}
    c=lambda gs: np.vstack([raw[g]>0 for g in gs]).sum(0)
    pool=(c(PAN)>=3)&(c(LIN)>=1)
    cs=pd.read_csv(f"D:/thesis-research/score_genes_slice{sl}_v2/cell_scores.csv")
    lab=cs["celltype_v2"].to_numpy()
    print(f"\nslice {sl}: inclusive pool {int(pool.sum()):,} | Stage-1 Myeloid {int((lab=='Myeloid').sum()):,}")
    print("  pool cells by their Stage-1 label:", pd.Series(lab[pool]).value_counts().to_dict())
    print(f"  Stage-1 Myeloid NOT in pool: {int(((lab=='Myeloid')&~pool).sum()):,}")
