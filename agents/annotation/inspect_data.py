import anndata as ad, numpy as np, scipy.sparse as sp
base = "D:/thesis-research/resources/cache/"
for f in ["with_tumor_prediction/slice_1_adata.h5ad", "slice_1_adata_with_neg.h5ad", "decontx/slice_1_decontx.h5ad"]:
    p = base + f
    try:
        a = ad.read_h5ad(p)
    except Exception as e:
        print(f"\n### {f}: FAILED {e}")
        continue
    print(f"\n### {f}")
    print("shape", a.shape)
    print("obs cols:", list(a.obs.columns))
    print("obsm:", list(a.obsm.keys()))
    print("layers:", list(a.layers.keys()))
    X = a.X
    print("X dtype", X.dtype, "sparse", sp.issparse(X), "max", float(X.max()), "min", float(X.min()))
    d = X[:50].toarray() if sp.issparse(X) else np.asarray(X[:50])
    print("looks integer:", bool(np.allclose(d, np.round(d))))
    # tumor col
    for c in a.obs.columns:
        if "tumor" in c.lower():
            print("  tumor col:", c, "->", a.obs[c].value_counts().to_dict())
    # negative probes
    negs = [g for g in a.var_names if g.lower().startswith("negative")]
    sysc = [g for g in a.var_names if g.lower().startswith("systemcontrol")]
    print("  n Negative probes:", len(negs), "n SystemControl:", len(sysc))
    # check key markers present
    keymark = ["Cx3cr1","TMEM119","C1qa","C1qb","C1qc","Csf1r","Aif1","Tyrobp","Fcer1g","P2rx5","Selplg",
               "Ccr2","Plac8","Cd14","S100a8","S100a9","Mrc1","Cd163","Lyve1","Pf4","GFAP","Sox9","S100b",
               "Sparcl1","Glul","Pecam1","Cdh5","GFP","tdTomato","Nkg7","Klrb1c","Cd3e"]
    panel_lower = {g.lower(): g for g in a.var_names}
    present = [m for m in keymark if m.lower() in panel_lower]
    absent = [m for m in keymark if m.lower() not in panel_lower]
    print("  markers present:", present)
    print("  markers ABSENT:", absent)
