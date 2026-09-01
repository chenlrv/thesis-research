import pandas as pd, numpy as np
df = pd.read_csv("D:/thesis-research/agents/outputs/annotation/method_a_labels.csv", index_col=0)
mye = df[df["is_myeloid"]]
print("n myeloid:", len(mye))
print("myeloid Mrc1>0:", int((mye["cnt_Mrc1"] > 0).sum()), "Cd163>0:", int((mye["cnt_Cd163"] > 0).sum()))
print("myeloid Mrc1>0 OR Cd163>0:", int(((mye["cnt_Mrc1"] > 0) | (mye["cnt_Cd163"] > 0)).sum()))
print("myeloid Mrc1>=1&Cd163>=1:", int(((mye["cnt_Mrc1"] > 0) & (mye["cnt_Cd163"] > 0)).sum()))
print("BAM-called:", int((df["method_a"] == "BAM").sum()))
# how many myeloid have Mrc1+Cd163 summed >=1 (any) vs sig
s = mye["cnt_Mrc1"] + mye["cnt_Cd163"]
print("myeloid with Mrc1+Cd163 sum>=1:", int((s >= 1).sum()), ">=2:", int((s >= 2).sum()))
# specificity: among BAM-called, what is Cx3cr1/microglia contamination
bam = df[df["method_a"] == "BAM"]
print("\nBAM cells: mean Mrc1", round(bam["cnt_Mrc1"].mean(),2), "Cd163", round(bam["cnt_Cd163"].mean(),2),
      "Cx3cr1", round(bam["cnt_Cx3cr1"].mean(),2), "Lyve1", round(bam["cnt_Lyve1"].mean(),2))
