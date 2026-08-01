"""Cross-slice Stage-1 Myeloid summary from the v2 cell_scores.csv files."""
import h5py
import numpy as np
import pandas as pd

TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
ROLE = {1: "tumor", 2: "tumor", 3: "control", 4: "control", 5: "tumor", 6: "tumor"}


def decode(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def tumor_n(sl):
    with h5py.File(TMPL.format(sl), "r") as h5:
        node = h5["obs"]["pred_tumor_XGBoost"]
        if isinstance(node, h5py.Group):
            codes = node["codes"][...]
            cats = decode(node["categories"][...])
            vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
            t = np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
        else:
            arr = node[...]
            t = (np.isin(decode(arr).astype(str), ["True", "1"])
                 if arr.dtype.kind in ("S", "O") else arr.astype(bool))
    return int(t.sum())


print(f"{'sl':>2} {'role':>7} {'tumor':>7} {'non-tumor':>9} {'Myeloid':>8} "
      f"{'%nt':>6} {'ambig':>6} {'lowcov':>6}")
for s in range(1, 7):
    df = pd.read_csv(f"D:/thesis-research/score_genes_slice{s}_v2/cell_scores.csv")
    nt = len(df)
    v = df["celltype_v2"].value_counts()
    my = int(v.get("Myeloid", 0))
    amb = int(v.get("ambiguous", 0))
    low = int(v.get("low_markers_coverage", 0))
    print(f"{s:>2} {ROLE[s]:>7} {tumor_n(s):>7,} {nt:>9,} {my:>8,} "
          f"{100*my/nt:>5.1f}% {amb:>6,} {low:>6,}")
