import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import anndata as ad, h5py, numpy as np, pandas as pd, scanpy as sc
from scipy.sparse import csc_matrix, csr_matrix
TUMOR={1,2,5,6}; RATIO=1.5; COVK={"Astrocytes":2,"Neurons":2,"Myeloid":2,"Ependymal":2,"Vascular":4}
LIN=["Csf1r","Aif1","Tyrobp","Fcer1g"]
BAMg=["Pf4","Maf","Mrc1","Cd163","Cd5l"]; BSTAB=["Pf4","Maf"]; MDMsup=["Plac8","Vcan","Cd14","Ccr1","Fpr1"]
MYE={"current":["C1qa","C1qb","C1qc","Csf1r","Aif1","Tyrobp","Fcer1g"],
     "reb_lineage":["Csf1r","Aif1","Tyrobp","Fcer1g"],
     "reb_1xC1q":["C1qa","Csf1r","Aif1","Tyrobp","Fcer1g"]}
def base(sl): return {"Astrocytes":["Sparcl1","Fgfr3","Glul","Gpx3","S100b","Sox9"],
 "Neurons":["Meg3","Nrxn1","Nrxn3","Scg5","Cx3cl1","Xkr4","Ryr2","Pnoc","Calb1","Sst"],
 "Ependymal":(["Ttr","Adgrv1","Cd24a"] if sl in TUMOR else ["Ttr","Adgrv1","Cd24a","Krt8","Krt18","Krt19","Cldn4","Epcam"]),
 "Vascular":["Cdh5","Pecam1","Flt1","Kdr","Tek","Tie1","Esam","Slc2a1","Clec14a","Adgrl4","Eng","Icam2","Ramp2","Vwf","Rgs5","Pdgfrb","Notch3","Vtn"]}
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
def sg(a,g,n): sc.tl.score_genes(a,[x for x in g if x in a.var_names],score_name=n,ctrl_size=50,n_bins=25,random_state=0);return a.obs[n].to_numpy()
def decide(S,cov,L,lindet):
    raw=S.to_numpy(float); mad=1.4826*np.median(np.abs(raw-np.median(raw,0)),0); mad=np.where(mad>0,mad,raw.std(0)); mad=np.where(mad>0,mad,1.0)
    scd=raw/mad; order=np.argsort(-scd,1); lab=np.array(L)
    topl=lab[order[:,0]];secl=lab[order[:,1]]
    top=np.take_along_axis(scd,order[:,:1],1)[:,0];sec=np.take_along_axis(scd,order[:,1:2],1)[:,0]
    marg=(top>0)&((sec<=0)|(top>=RATIO*sec))
    thr={l:ft(S[l].to_numpy()) for l in L}; topraw=raw[np.arange(len(raw)),order[:,0]]
    fdrp=topraw>=np.array([thr[l] for l in topl])
    covt=np.array([cov[topl[i]][i] for i in range(len(topl))]); covp=covt>=np.array([COVK[l] for l in topl])
    out=np.full(len(topl),"unknown",dtype=object); out[fdrp&~marg]="ambiguous"
    out[fdrp&marg&~covp]="low"; ok=fdrp&marg&covp; out[ok]=topl[ok]
    # lineage rescue (k>=2) on myeloid-in-top-2 ambiguous
    myc=(out=="ambiguous")&((topl=="Myeloid")|(secl=="Myeloid"))&(lindet>=2)
    out[myc]="Myeloid"
    return out
for sl in (1,3):
    with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
        Xk=X(h)[~bl(h,"pred_tumor_XGBoost")].tocsr();v=var(h)
    c=lambda gs: np.vstack([np.asarray(Xk[:,v.index(g)].todense()).ravel()>0 for g in gs if g in v]).sum(0)
    lindet=c(LIN)
    bam_cd=c(BAMg)>=2
    ccr2=np.asarray(Xk[:,v.index("Ccr2")].todense()).ravel()>0
    mdm_cd=(~bam_cd)&ccr2&(c(MDMsup)>=1)&(c(BSTAB)==0)
    a=ad.AnnData(X=Xk.astype(np.float32));a.var_names=pd.Index(v);a.var_names_make_unique()
    sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
    bm=base(sl); Sother={l:sg(a,bm[l],f"s_{l}") for l in bm}
    covother={l:np.asarray((Xk[:,[v.index(g) for g in bm[l] if g in v]]>0).sum(1)).ravel() for l in bm}
    print(f"\n=== slice {sl} ({'CONTROL' if sl not in TUMOR else 'tumor'}) ===")
    print(f"  {'module':13}{'myeloid':>9}{'BAM':>7}{'MDM':>7}")
    for name,genes in MYE.items():
        smy=sg(a,genes,f"s_my_{name}")
        S=pd.DataFrame({**Sother,"Myeloid":smy})[["Astrocytes","Neurons","Myeloid","Ependymal","Vascular"]]
        cov={**covother,"Myeloid":np.asarray((Xk[:,[v.index(g) for g in genes if g in v]]>0).sum(1)).ravel()}
        lab=decide(S,cov,["Astrocytes","Neurons","Myeloid","Ependymal","Vascular"],lindet)
        my=lab=="Myeloid"
        print(f"  {name:13}{int(my.sum()):>9,}{int((my&bam_cd).sum()):>7,}{int((my&mdm_cd).sum()):>7,}")
