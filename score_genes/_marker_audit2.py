import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import h5py, numpy as np, pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
def _dec(a): return np.array([x.decode() if isinstance(x,bytes) else x for x in a]) if a.dtype.kind in ("O","S") else a
def _X(h):
    n=h["X"];e=str(n.attrs.get("encoding-type",""));s=tuple(n.attrs["shape"]);a=(n["data"][...],n["indices"][...],n["indptr"][...])
    return (csc_matrix(a,shape=s) if "csc" in e else csr_matrix(a,shape=s)).tocsr()
def _var(h):
    v=h["var"];k=v.attrs.get("_index","_index");return list(_dec(v[k.decode() if isinstance(k,bytes) else k][...]))
def _bl(h,c):
    nd=h["obs"][c]
    if isinstance(nd,h5py.Group):
        cd=nd["codes"][...];ct=_dec(nd["categories"][...])
        return np.isin(np.where(cd>=0,ct[np.clip(cd,0,None)],"False").astype(str),["True","1","1.0","TRUE","true"])
    ar=nd[...];return np.isin(_dec(ar).astype(str),["True","1"]) if ar.dtype.kind in ("S","O") else ar.astype(bool)
sl=1
with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
    X=_X(h);var=_var(h);tum=_bl(h,"pred_tumor_XGBoost")
Xk=X[~tum].tocsr()
lab=pd.read_csv(f"D:/thesis-research/score_genes_slice{sl}_v2/final_v3/annotation.csv").celltype_final.to_numpy()
# case-insensitive lookup of custom probes + a few checks
want=["GFAP","TMEM119","Trem2","Lyve1","Ccl2","Cxcl13","GFP","tdTomato","Cx3cr1","Aif1","Csf1r","P2ry12","Aqp4","Foxj1","Snap25"]
low={g.lower():g for g in var}
print("panel presence (case-insensitive) + specificity vs target type:\n")
tgtmap={"GFAP":"Astrocytes","TMEM119":"Microglia","Trem2":"Microglia","Lyve1":"BAM","Cx3cr1":"Microglia",
        "P2ry12":"Microglia","Aqp4":"Astrocytes","Foxj1":"Ependymal","Snap25":"Neurons","Aif1":"Microglia","Csf1r":"Microglia",
        "Ccl2":"Microglia","Cxcl13":"Lymphoid","GFP":"Microglia","tdTomato":"Microglia"}
vi={g:i for i,g in enumerate(var)}
for g in want:
    real=low.get(g.lower())
    if real is None: print(f"   {g:10s} -> NOT ON PANEL"); continue
    tgt=tgtmap.get(g,"Microglia"); m=(lab==tgt)
    det=np.asarray((Xk[:,vi[real]]>0).todense()).ravel()
    p_t=det[m].mean() if m.sum() else float('nan'); p_g=det.mean()
    spec=p_t/max(p_g,1e-9)
    print(f"   {g:10s} -> panel name '{real}'  det_in_{tgt}={100*p_t:5.1f}%  global={100*p_g:5.1f}%  specificity={spec:4.1f}x")
