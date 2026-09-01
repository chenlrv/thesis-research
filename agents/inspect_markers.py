"""Find available reference/stock markers + classify control probes."""
import anndata as ad
import numpy as np

a = ad.read_h5ad(r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad")
vn = list(a.var_names)

neg = [v for v in vn if v.lower().startswith("negative")]
sysc = [v for v in vn if v.lower().startswith("systemcontrol")]
print(f"Negative* probes (n={len(neg)}): {neg}")
print(f"SystemControl* probes (n={len(sysc)})  e.g.: {sysc[:5]}")

# candidate stock neuron / endothelial / astro / canonical markers
cands = ["Snap25", "Rbfox3", "Syt1", "Meg3", "Map2", "Stmn2", "Gad1", "Gad2",
         "Slc17a7", "Slc17a6", "Nrgn", "Tubb3", "Eno2", "Syp",
         "Pecam1", "Cldn5", "Flt1", "Cd31",
         "Aqp4", "Gfap", "Aldh1l1", "Slc1a3", "S100b",
         "Cx3cr1", "P2ry12", "Tmem119", "Hexb", "Sall1", "Csf1r", "Aif1", "Itgam",
         "Mrc1", "Cd163", "Lyve1", "Pf4", "Ms4a7",
         "Ptprc", "Cd3e", "Cd8a", "Nkg7", "Mki67", "Mbp", "Plp1", "Mog", "Olig1", "Olig2",
         "Pdgfra", "Cspg4", "Vtn", "Pdgfrb", "Rgs5"]
present = [g for g in cands if g in a.var_names]
absent = [g for g in cands if g not in a.var_names]
print("\nCANDIDATE MARKERS PRESENT:", present)
print("\nCANDIDATE MARKERS ABSENT :", absent)

# any neuron-ish gene in panel
neuronish = [v for v in vn if v.lower() in
             ("snap25","syt1","rbfox3","map2","stmn2","nrgn","syp","eno2","tubb3","meg3","gad1","gad2","slc17a7","slc17a6")]
print("\nNeuron-ish in panel:", neuronish)
