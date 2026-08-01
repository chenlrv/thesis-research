import pandas as pd, numpy as np
df=pd.read_csv("D:/thesis-research/score_genes_slice3_v2/final_v3/annotation.csv")
x0,x1,y0,y1=21776,29740,-2660,5303
cl=(df.x>=x0)&(df.x<=x1)&(df.y>=y0)&(df.y<=y1)
print(f"clump cells: {int(cl.sum()):,}  (of {len(df):,} non-tumor)")
for t in ["Microglia","BAM","MDM","Myeloid_unresolved"]:
    tot=int((df.celltype_final==t).sum()); inc=int(((df.celltype_final==t)&cl).sum())
    print(f"  {t:20s} total={tot:>6,}  in clump={inc:>5,}  ({100*inc/max(tot,1):.0f}% of all {t})")
print("\nclump composition:", df.loc[cl,"celltype_final"].value_counts().to_dict())
