import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import h5py, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import final_annotation as FA
MYE={"Microglia","BAM","MDM","Myeloid_unresolved"}
for sl in (1,2,3):
    out=f"D:/thesis-research/score_genes_slice{sl}_v2/final_v3"
    df=pd.read_csv(f"{out}/annotation.csv")
    with h5py.File(FA.TMPL.format(sl),"r") as h:
        cx=FA._num(h,"CenterX_global_px");cy=FA._num(h,"CenterY_global_px");tum=FA._bl(h,"pred_tumor_XGBoost")
    tx,ty=cx[tum],cy[tum]
    x=df.x.to_numpy();y=df.y.to_numpy();final=df.celltype_final.to_numpy().astype(object)
    tot=len(df)+int(tum.sum())
    FA.plots(sl,x,y,final,tx,ty,tot,out)   # per-type + ALL, now s=1.2 (== tumor)
    # broad myeloid, matching sizes
    m=np.isin(final,list(MYE))
    fig,ax=plt.subplots(figsize=(10,8),dpi=160)
    ax.scatter(x[~m],y[~m],s=0.6,c="#e8e8e8",linewidths=0,rasterized=True,label="non-myeloid")
    if len(tx): ax.scatter(tx,ty,s=1.2,c="black",linewidths=0,rasterized=True,label=f"tumor ({len(tx):,})")
    for t,col in [("Microglia","#17becf"),("BAM","#d62728"),("MDM","#00a087"),("Myeloid_unresolved","#8c564b")]:
        mm=final==t
        ax.scatter(x[mm],y[mm],s=1.2,c=col,linewidths=0,rasterized=True,label=f"{t} ({int(mm.sum()):,})")
    ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(f"slice {sl} — broad Myeloid compartment (n={int(m.sum()):,}, {100*m.sum()/tot:.1f}%)")
    ax.legend(loc="lower right",markerscale=10,fontsize=8)
    fig.savefig(f"{out}/slice{sl}_Myeloid_broad.png",bbox_inches="tight");plt.close(fig)
    print(f"slice {sl} re-plotted (dots == tumor size)")
