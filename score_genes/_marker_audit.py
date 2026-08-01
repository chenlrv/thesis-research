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
# candidate markers: (used in current module?) + gold-standard alternatives
CAND={
 "Astrocytes":[("Sparcl1","used"),("Fgfr3","used"),("Glul","used"),("Gpx3","used"),("S100b","used"),("Sox9","used"),
               ("Aqp4","GOLD"),("Aldh1l1","GOLD"),("Gfap","GOLD/custom"),("Slc1a3","GOLD"),("Gja1","alt")],
 "Neurons":[("Meg3","used-AMBIENT"),("Nrxn1","used"),("Snap25","GOLD"),("Syt1","GOLD"),("Rbfox3","GOLD"),("Stmn2","alt"),("Sst","subtype"),("Calb1","subtype")],
 "Microglia":[("Adgrg1","used"),("Hpgds","used"),("Gpr183","used"),
              ("P2ry12","GOLD"),("Tmem119","GOLD/custom"),("Sall1","GOLD"),("Hexb","GOLD"),("Cx3cr1","GOLD"),("Siglech","alt"),("Selplg","alt")],
 "Ependymal":[("Ttr","used-choroid"),("Adgrv1","used"),("Cd24a","used"),("Foxj1","GOLD"),("Ccdc153","alt"),("Rarres2","alt"),("Hdc","alt")],
 "Lymphoid":[("Cd3e","used"),("Cd79a","used"),("Nkg7","used"),("Ccl19","used-NICHE"),("Cd52","alt"),("Ptprc","alt")],
}
sl=1
with h5py.File(f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sl}_adata.h5ad","r") as h:
    X=_X(h);var=_var(h);tum=_bl(h,"pred_tumor_XGBoost")
Xk=X[~tum].tocsr()
lab=pd.read_csv(f"D:/thesis-research/score_genes_slice{sl}_v2/final_v3/annotation.csv").celltype_final.to_numpy()
# map myeloid subtypes to Microglia for the microglia check; keep others
labmap={"Microglia":"Microglia"}
vi={g:i for i,g in enumerate(var)}
import numpy as np
def norm_mean(idx,mask):
    sub=Xk[mask][:,idx]; lib=np.asarray(Xk[mask].sum(1)).ravel(); lib[lib==0]=1
    return float((np.asarray(sub.sum(0)).ravel()/lib.sum())*1e4*mask.sum()/mask.sum()) if False else float(np.asarray((sub>0).mean(0)).ravel()[0])
allmask=np.ones(len(lab),bool)
print(f"slice {sl}: specificity = detection% in target type vs global detection% (present markers only)\n")
for typ,markers in CAND.items():
    tgt = (lab=="Microglia") if typ=="Microglia" else (lab==typ)
    print(f"== {typ} (n={int(tgt.sum()):,}) ==")
    for g,tag in markers:
        if g not in vi: print(f"   {g:10s} {tag:14s} NOT ON PANEL"); continue
        col=Xk[:,vi[g]]
        det=np.asarray((col>0).todense()).ravel()
        p_t=det[tgt].mean(); p_g=det.mean()
        spec=p_t/max(p_g,1e-9)
        flag = "  <-- weak/nonspecific" if (spec<2 and tag in("used","used-AMBIENT","used-NICHE","used-choroid")) else ("  <== GOLD, unused, specific" if (tag.startswith("GOLD") and spec>=3) else "")
        print(f"   {g:10s} {tag:14s} det_in_type={100*p_t:5.1f}%  global={100*p_g:4.1f}%  specificity={spec:4.1f}x{flag}")
    print()
