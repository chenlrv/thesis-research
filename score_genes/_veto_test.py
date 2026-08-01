import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import h5py, numpy as np
from scipy.sparse import csc_matrix, csr_matrix
PAN=["C1qa","C1qb","C1qc","Csf1r","Aif1","Tyrobp","Fcer1g"]; LIN=["Csf1r","Aif1","Tyrobp","Fcer1g"]
MICRO=["Adgrg1","Hpgds","Gpr183"]; BAMg=["Pf4","Maf","Mrc1","Cd163","Cd5l"]; BSTAB=["Pf4","Maf"]
MDMsup=["Plac8","Vcan","Cd14","Ccr1","Fpr1"]
LYMPH=["Cd3e","Cd3d","Cd6","Ms4a1","Cd79a","Ccl19","Ccl21a/b/d","Nkg7","Klrb1c","Cd8a","Cd8b1","Cd4","Il7r"]
ENDO=["Pecam1","Cdh5","Ackr1","Flt1","Kdr","Vwf"]
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
def num(h,c):
    nd=h["obs"][c]
    if isinstance(nd,h5py.Group):
        cd=nd["codes"][...];ct=dec(nd["categories"][...]).astype(float);return ct[np.clip(cd,0,None)]
    return nd[...].astype(float)
for sl in [1,2,3]:
    with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
        Xk=X(h)[~b(h,"pred_tumor_XGBoost")];v=var(h);cxk=num(h,"CenterX_global_px")[~b(h,"pred_tumor_XGBoost")];cyk=num(h,"CenterY_global_px")[~b(h,"pred_tumor_XGBoost")]
    gi={g:v.index(g) for g in set(PAN+LIN+MICRO+BAMg+MDMsup+LYMPH+ENDO+["Ccr2"]) if g in v}
    c=lambda gs: np.vstack([np.asarray(Xk[:,gi[g]].todense()).ravel()>0 for g in gs if g in gi]).sum(0)
    pool=(c(PAN)>=3)&(c(LIN)>=1)
    bam=pool&(c(BAMg)>=2); mdm=pool&~bam&(np.asarray(Xk[:,gi["Ccr2"]].todense()).ravel()>0)&(c(MDMsup)>=1)&(c(BSTAB)==0)
    micro=pool&~bam&~mdm&(c(MICRO)>=1)
    lym=c(LYMPH)>=2; endo=c(ENDO)>=2; veto=lym|endo
    def rep(nm,mask):
        print(f"  {nm:10} {int(mask.sum()):>6,}  lymph+={int((mask&lym).sum()):>4}  endo+={int((mask&endo).sum()):>4}  CLEAN(neither)={int((mask&~veto).sum()):>6,}")
    role="CONTROL" if sl==3 else "tumor"
    print(f"\nslice {sl} ({role}) pool={int(pool.sum()):,}")
    rep("BAM",bam); rep("MDM",mdm); rep("Microglia",micro)
    if sl==3:
        cl=(cxk>=21776)&(cxk<=29740)&(cyk>=-2660)&(cyk<=5303)
        print(f"  CLUMP: BAM {int((bam&cl).sum())}->{int((bam&cl&~veto).sum())}  "
              f"Micro {int((micro&cl).sum())}->{int((micro&cl&~veto).sum())}  MDM {int((mdm&cl).sum())}->{int((mdm&cl&~veto).sum())}  (after veto)")
