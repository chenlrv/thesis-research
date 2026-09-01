import pandas as pd, numpy as np
OUT = "D:/thesis-research/agents/outputs/annotation"
lab = pd.read_csv(OUT + "/all_method_labels.csv", index_col=0)
FOCAL = ["Microglia", "MDM", "BAM", "Astrocyte"]

# 1) Reporter-gating distortion headline: A vs Bfixed vs Basis microglia call counts
print("=== Microglia recovery (the thin-axis / dead-GFP story) ===")
for m in ["A", "Bfixed", "Basis", "C"]:
    print(f"  {m:7s} Microglia={int((lab[m]=='Microglia').sum()):6d}  "
          f"MDM={int((lab[m]=='MDM').sum()):6d}  BAM={int((lab[m]=='BAM').sum()):5d}  "
          f"micro:(BAM+MDM)={ (lab[m]=='Microglia').sum()/max(1,(lab[m].isin(['BAM','MDM'])).sum()):.2f}")

# 2) Where Basis loses microglia: cells A calls Microglia, what does Basis call?
amic = lab[lab["A"] == "Microglia"]
print(f"\nOf {len(amic)} cells A calls Microglia, Basis (reporter-GAT) calls:")
print(amic["Basis"].value_counts())
print(f"  -> Basis recovers only {(amic['Basis']=='Microglia').mean()*100:.1f}% of A's microglia")
print(f"  Bfixed recovers {(amic['Bfixed']=='Microglia').mean()*100:.1f}% of A's microglia")

# 3) MDM-only-via-dead-tag (expanded)
mdm_tagonly = lab[(lab["Basis"] == "MDM") & (lab["A"] != "MDM") & (lab["Bfixed"] != "MDM")]
print(f"\n=== 'MDM only via dead tag' cells: {len(mdm_tagonly)} ===")
print("  GFP==0 frac:", round((mdm_tagonly["cnt_GFP"] == 0).mean(), 3))
print("  Ccr2==0 & Plac8==0 frac (no transcriptomic MDM evidence):",
      round(((mdm_tagonly["cnt_Ccr2"] == 0) & (mdm_tagonly["cnt_Plac8"] == 0)).mean(), 3))
print("  what A actually calls them:")
print(mdm_tagonly["A"].value_counts())

# 4) Consensus strength among the 3 transcriptome methods (A, Bfixed, C)
print("\n=== Consensus of A/Bfixed/C among focal cells ===")
focal_any = lab[["A", "Bfixed", "C"]].isin(FOCAL).any(axis=1)
sub = lab[focal_any]
print("  unanimous (3/3):", int((sub["consensus3_n"] == 3).sum()),
      f"({(sub['consensus3_n']==3).mean()*100:.1f}%)")
print("  2/3:", int((sub["consensus3_n"] == 2).sum()))
print("  no majority (all differ):", int((sub["consensus3_n"] == 1).sum()))

# 5) Characterize high-disagreement cells: are they thin-microglia (Cx3cr1+ but no TMEM119/P2rx5)?
dis = pd.read_csv(OUT + "/high_disagreement_cells.csv", index_col=0)
print(f"\n=== High-disagreement focal cells: {len(dis)} ===")
print("  median nCount:", float(dis["nCount"].median()))
# thin microglia: Cx3cr1>0 but TMEM119==0 & P2rx5==0
if all(c in dis for c in ["cnt_Cx3cr1", "cnt_TMEM119", "cnt_P2rx5"]):
    thin = (dis["cnt_Cx3cr1"] > 0) & (dis["cnt_TMEM119"] == 0)
    print("  Cx3cr1+ but TMEM119- (ambiguous microglia/myeloid):",
          int(thin.sum()), f"({thin.mean()*100:.1f}%)")
# co-expressers: positive for >=2 lineage-specific markers
mk = ["cnt_Cx3cr1", "cnt_Ccr2", "cnt_Plac8", "cnt_Mrc1", "cnt_Cd163", "cnt_GFAP"]
mk = [c for c in mk if c in dis]
npos = (dis[mk] > 0).sum(axis=1)
print("  cells positive for >=2 different lineage markers (co-expression):",
      int((npos >= 2).sum()), f"({(npos>=2).mean()*100:.1f}%)")
print("  cells positive for 0 of these markers (called focal by smoothing/force-bin):",
      int((npos == 0).sum()), f"({(npos==0).mean()*100:.1f}%)")
