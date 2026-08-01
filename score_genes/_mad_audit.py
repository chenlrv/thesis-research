import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import anndata as ad, h5py, numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix
TUMOR={1,2,5,6}; RATIO=1.5
def M(sl): return {"Astrocytes":["Sparcl1","Fgfr3","Glul","Gpx3","S100b","Sox9"],
 "Neurons":["Meg3","Nrxn1","Nrxn3","Scg5","Cx3cl1","Xkr4","Ryr2","Pnoc","Calb1","Sst"],
 "Myeloid":["C1qa","C1qb","C1qc","Csf1r","Aif1","Tyrobp","Fcer1g"],
 "Ependymal":(["Ttr","Adgrv1","Cd24a"] if sl in TUMOR else ["Ttr","Adgrv1","Cd24a","Krt8","Krt18","Krt19","Cldn4","Epcam"]),
 "Vascular":["Cdh5","Pecam1","Flt1","Kdr","Tek","Tie1","Esam","Slc2a1","Clec14a","Adgrl4","Eng","Icam2","Ramp2","Vwf","Rgs5","Pdgfrb","Notch3","Vtn"]}
LIN=["Csf1r","Aif1","Tyrobp","Fcer1g"]
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
def ft(s,f=0.05):
    s=np.asarray(s,float)
    for t in np.unique(np.sort(s[s>0])):
        if int((s<=-t).sum())/max(int((s>=t).sum()),1)<=f: return float(t)
    return np.inf
def sg(a,g,n): 
    sc.tl.score_genes(a,[x for x in g if x in a.var_names],score_name=n,ctrl_size=50,n_bins=25,random_state=0);return a.obs[n].to_numpy()
for sl in [1,3]:
    mk=M(sl);L=list(mk)
    with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
        Xk=X(h)[~bl(h,"pred_tumor_XGBoost")].tocsr();v=var(h)
    lin=np.vstack([np.asarray(Xk[:,v.index(g)].todense()).ravel()>0 for g in LIN]).sum(0)
    a=ad.AnnData(X=Xk.astype(np.float32));a.var_names=pd.Index(v);a.var_names_make_unique()
    sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
    S=pd.DataFrame({l:sg(a,mk[l],f"s_{l}") for l in L})[L]; raw=S.to_numpy(float)
    med=np.median(raw,0); mad=1.4826*np.median(np.abs(raw-med),0); mad=np.where(mad>0,mad,raw.std(0))
    thr={l:ft(S[l].to_numpy()) for l in L}
    print(f"\n=== slice {sl} ===")
    print(f"  {'type':11}{'median':>8}{'MAD':>8}{'FDRthr':>8}")
    for i,l in enumerate(L): print(f"  {l:11}{med[i]:>8.3f}{mad[i]:>8.3f}{thr[l]:>8.3f}")
    scd=raw/mad; order=np.argsort(-scd,1); lab=np.array(L)
    topl=lab[order[:,0]];secl=lab[order[:,1]]
    top=np.take_along_axis(scd,order[:,:1],1)[:,0];sec=np.take_along_axis(scd,order[:,1:2],1)[:,0]
    marg=(top>0)&((sec<=0)|(top>=RATIO*sec))
    topraw=raw[np.arange(len(raw)),order[:,0]];fdrp=topraw>=np.array([thr[l] for l in topl])
    amb=fdrp&~marg
    # MAD effect: myeloid-top ambiguous cells that WOULD pass under raw-ratio margin
    mi=L.index("Myeloid")
    myfdr=raw[:,mi]>=thr["Myeloid"]
    below=(~myfdr)&(lin>=2)
    print(f"  Myeloid MAD vs competitor-median MAD: {mad[mi]:.3f} vs {np.median(np.delete(mad,mi)):.3f}"
          f"  -> raw-ratio bar for Myeloid ~= {RATIO*mad[mi]/np.median(np.delete(mad,mi)):.2f}x")
    myt_amb=amb&(topl=="Myeloid")
    secraw=raw[np.arange(len(raw)),order[:,1]]
    rawmarg=(topraw>0)&((secraw<=0)|(topraw>=RATIO*secraw))
    flip=myt_amb&rawmarg
    print(f"  cells below Myeloid-FDR but >=2 lineage genes (weak myeloid missed by FDR): {int(below.sum()):,}")
    print(f"  Myeloid-top ambiguous: {int(myt_amb.sum()):,}; of these, pass under RAW margin (MAD was the blocker): {int(flip.sum()):,}")
