import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import h5py, numpy as np, pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
LIN=["Csf1r","Aif1","Tyrobp","Fcer1g"]; COMPL=["C1qa","C1qb","C1qc"]
NEURID=["Nrxn1","Nrxn3","Scg5","Ryr2","Calb1"]; PROF=LIN+COMPL+["Meg3"]+NEURID
def dec(a): return np.array([x.decode() if isinstance(x,bytes) else x for x in a]) if a.dtype.kind in ("O","S") else a
def X(h):
    n=h["X"];e=str(n.attrs.get("encoding-type",""));s=tuple(n.attrs["shape"])
    a=(n["data"][...],n["indices"][...],n["indptr"][...]);return (csc_matrix(a,shape=s) if "csc" in e else csr_matrix(a,shape=s)).tocsr()
def var(h):
    v=h["var"];k=v.attrs.get("_index","_index");return list(dec(v[k.decode() if isinstance(k,bytes) else k][...]))
def bl(h,c):
    nd=h["obs"][c]
    if isinstance(nd,h5py.Group):
        cd=nd["codes"][...];ct=dec(nd["categories"][...])
        return np.isin(np.where(cd>=0,ct[np.clip(cd,0,None)],"False").astype(str),["True","1","1.0","TRUE","true"])
    ar=nd[...];return np.isin(dec(ar).astype(str),["True","1"]) if ar.dtype.kind in ("S","O") else ar.astype(bool)
for sl in (1,3):
    with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
        Xk=X(h)[~bl(h,"pred_tumor_XGBoost")];v=var(h)
    raw={g:np.asarray(Xk[:,v.index(g)].todense()).ravel() for g in PROF if g in v}
    lib=np.asarray(Xk.sum(1)).ravel()
    lindet=np.vstack([raw[g]>0 for g in LIN]).sum(0)
    d=f"D:/thesis-research/score_genes_slice{sl}_v2"
    cs=pd.read_csv(f"{d}/cell_scores.csv"); thr=pd.read_csv(f"{d}/mirrored_fdr_thresholds.csv").set_index("label")["threshold"]
    smy=cs["score_Myeloid"].to_numpy(); lab=cs["celltype_v2"].to_numpy()
    below=(smy<thr["Myeloid"])&(lindet>=2); conf=lab=="Myeloid"
    print(f"\n=== slice {sl} ===  below-FDR & >=2 lineage: {int(below.sum()):,}   (confMyeloid {int(conf.sum()):,})")
    print(f"  their celltype_v2:", pd.Series(lab[below]).value_counts().to_dict())
    print(f"  median lib  below={np.median(lib[below]):.0f}  conf={np.median(lib[conf]):.0f}")
    print(f"  median Myeloid score  below={np.median(smy[below]):.2f}  (FDR thr={thr['Myeloid']:.2f}, conf median={np.median(smy[conf]):.2f})")
    print(f"  {'gene':8}{'below':>8}{'confMyel':>9}{'group':>10}")
    for g in PROF:
        if g in raw:
            grp=("lineage" if g in LIN else "complement" if g in COMPL else "neur-amb" if g=="Meg3" else "neur-id")
            print(f"  {g:8}{raw[g][below].mean():>8.2f}{raw[g][conf].mean():>9.2f}{grp:>10}")
    nd=lambda gs,m: np.vstack([raw[g][m]>0 for g in gs]).sum(0).mean()
    print(f"  #lineage det  below={nd(LIN,below):.2f} conf={nd(LIN,conf):.2f} | #neur-id det below={nd(NEURID,below):.2f} conf={nd(NEURID,conf):.2f}")
    for k in (2,3,4):
        b3=(smy<thr["Myeloid"])&(lindet>=k)
        print(f"  below-FDR & >={k} lineage: {int(b3.sum()):,}")
