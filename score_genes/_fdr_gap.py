import pandas as pd, numpy as np
for sl in (1,3):
    d=f"D:/thesis-research/score_genes_slice{sl}_v2"
    cs=pd.read_csv(f"{d}/cell_scores.csv")
    thr=pd.read_csv(f"{d}/mirrored_fdr_thresholds.csv").set_index("label")["threshold"]
    myfdr = cs["score_Myeloid"].to_numpy() >= thr["Myeloid"]
    lab = cs["celltype_v2"].to_numpy()
    print(f"\n=== slice {sl} ===  Myeloid passes its FDR: {int(myfdr.sum()):,} cells")
    print("  where those cells ended up (celltype_v2):")
    print(pd.Series(lab[myfdr]).value_counts().to_string())
    gap = myfdr & (lab=="unknown")
    print(f"  >>> Myeloid-FDR-pass but dumped to 'unknown' (your gap): {int(gap.sum()):,}")
