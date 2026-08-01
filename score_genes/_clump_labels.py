import pandas as pd, numpy as np
d="D:/thesis-research/score_genes_slice3_v2"
cs=pd.read_csv(f"{d}/cell_scores.csv")            # x,y,celltype_v2 (score_genes strict)
fc=pd.read_csv(f"{d}/annotation_final_codetect.csv")  # x,y,celltype_final (detection override)
# clump bbox from the characterization
x0,x1,y0,y1=21776,29740,-2660,5303
m=(cs["x"]>=x0)&(cs["x"]<=x1)&(cs["y"]>=y0)&(cs["y"]<=y1)
print(f"clump cells: {int(m.sum()):,}")
print("\nscore_genes (celltype_v2) called them:")
print(cs.loc[m,"celltype_v2"].value_counts().to_string())
print("\ndetection-override (celltype_final_codetect) called them:")
print(fc.loc[m.values,"celltype_final"].value_counts().to_string())
