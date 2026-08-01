import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import h5py, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
TMPL="D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
MYE={"Microglia","BAM","MDM","Myeloid_unresolved"}
def _dec(a): return np.array([x.decode() if isinstance(x,bytes) else x for x in a]) if a.dtype.kind in ("O","S") else a
def _num(h,c):
    nd=h["obs"][c]
    if isinstance(nd,h5py.Group):
        cd=nd["codes"][...];ct=_dec(nd["categories"][...]).astype(float);return ct[np.clip(cd,0,None)]
    return nd[...].astype(float)
def _bl(h,c):
    nd=h["obs"][c]
    if isinstance(nd,h5py.Group):
        cd=nd["codes"][...];ct=_dec(nd["categories"][...])
        return np.isin(np.where(cd>=0,ct[np.clip(cd,0,None)],"False").astype(str),["True","1","1.0","TRUE","true"])
    ar=nd[...];return np.isin(_dec(ar).astype(str),["True","1"]) if ar.dtype.kind in ("S","O") else ar.astype(bool)
for sl in (1,2,3):
    df=pd.read_csv(f"D:/thesis-research/score_genes_slice{sl}_v2/final_v3/annotation.csv")
    with h5py.File(TMPL.format(sl),"r") as h:
        cx=_num(h,"CenterX_global_px");cy=_num(h,"CenterY_global_px");tum=_bl(h,"pred_tumor_XGBoost")
    tx,ty=cx[tum],cy[tum]
    m=df.celltype_final.isin(MYE).to_numpy()
    tot=len(df)+int(tum.sum())
    # colored-by-subtype version
    fig,ax=plt.subplots(figsize=(10,8),dpi=160)
    ax.scatter(df.x[~m],df.y[~m],s=0.5,c="#e8e8e8",linewidths=0,rasterized=True,label="non-myeloid")
    if len(tx): ax.scatter(tx,ty,s=1.0,c="black",linewidths=0,rasterized=True,label=f"tumor ({len(tx):,})")
    for t,col in [("Microglia","#17becf"),("BAM","#d62728"),("MDM","#00a087"),("Myeloid_unresolved","#8c564b")]:
        mm=(df.celltype_final==t).to_numpy()
        ax.scatter(df.x[mm],df.y[mm],s=4,c=col,linewidths=0,rasterized=True,label=f"{t} ({int(mm.sum()):,})")
    ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f"slice {sl} — broad Myeloid compartment (n={int(m.sum()):,}, {100*m.sum()/tot:.1f}%)")
    ax.legend(loc="lower right",markerscale=3,fontsize=8)
    p=f"D:/thesis-research/score_genes_slice{sl}_v2/final_v3/slice{sl}_Myeloid_broad.png"
    fig.savefig(p,bbox_inches="tight");plt.close(fig)
    print(f"slice {sl}: broad myeloid={int(m.sum()):,} -> {p}")
