import pandas as pd, numpy as np
from final_annotation import run
run(3, src="D:/thesis-research/resources/cache/decontx/slice_3_decontx.h5ad", outdir="final_v3_decontx")
# clump comparison: original vs decontx
x0,x1,y0,y1=21776,29740,-2660,5303
orig=pd.read_csv("D:/thesis-research/score_genes_slice3_v2/final_v3/annotation.csv")
dec =pd.read_csv("D:/thesis-research/score_genes_slice3_v2/final_v3_decontx/annotation.csv")
print("\n=== clump: original -> decontX ===")
def clc(df): 
    cl=(df.x>=x0)&(df.x<=x1)&(df.y>=y0)&(df.y<=y1); return df.loc[cl,"celltype_final"].value_counts()
co,cd=clc(orig),clc(dec)
for t in ["Microglia","BAM","MDM","Myeloid_unresolved","Lymphoid","Vascular","unassigned"]:
    print(f"  {t:20s} {int(co.get(t,0)):>5,} -> {int(cd.get(t,0)):>5,}")
print("\n=== whole slice totals: original -> decontX ===")
for t in ["Microglia","BAM","MDM","Myeloid_unresolved","Lymphoid","Astrocytes","Neurons","Vascular","Choroid"]:
    print(f"  {t:20s} {int((orig.celltype_final==t).sum()):>6,} -> {int((dec.celltype_final==t).sum()):>6,}")
