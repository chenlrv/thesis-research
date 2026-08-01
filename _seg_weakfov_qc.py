import numpy as np
import pandas as pd

PATH = r"resources/cosmx/L321/L321_metadata_file.csv"
POST = "RNA_Basic.run_Cell.Typing.InSituType.1_1_posterior_probability"
cols = ["fov", POST, "Mean.DAPI", "Mean.Membrane", "nCount_RNA", "nFeature_RNA",
        "propNegative", "complexity", "qcCellsFlagged", "Area"]
df = pd.read_csv(PATH, usecols=cols)

# define weak FOVs from per-FOV mean intensity (same rule as the maps)
fovmean = df.groupby("fov")[["Mean.DAPI", "Mean.Membrane"]].mean()
mem_weak = set(fovmean.index[fovmean["Mean.Membrane"] < 0.5 * fovmean["Mean.Membrane"].median()])
dapi_weak = set(fovmean.index[fovmean["Mean.DAPI"] < 0.5 * fovmean["Mean.DAPI"].median()])
weak = mem_weak | dapi_weak
print(f"weak FOVs: {len(weak)} (membrane {len(mem_weak)} | DAPI {len(dapi_weak)})  of {df.fov.nunique()}")

inw = df.fov.isin(weak)
print(f"cells in weak FOVs: {inw.sum():,} ({100*inw.mean():.2f}% of {len(df):,})\n")

qf = df["qcCellsFlagged"].astype(str).str.lower().isin(["true", "1", "1.0"])
metrics = {
    "nCount_RNA (higher=better)":   df["nCount_RNA"],
    "nFeature_RNA (higher=better)": df["nFeature_RNA"],
    "propNegative (lower=better)":  df["propNegative"],
    "complexity":                   df["complexity"],
    "InSituType posterior (higher=better)": df[POST],
    "Area":                         df["Area"],
}
print(f"{'metric':38}{'weak-FOV med':>14}{'rest med':>12}{'ratio':>8}")
for name, s in metrics.items():
    a, b = s[inw].median(), s[~inw].median()
    print(f"{name:38}{a:>14.3f}{b:>12.3f}{(a/b if b else float('nan')):>8.2f}")
print(f"\n{'vendor qcCellsFlagged rate':38}{100*qf[inw].mean():>13.2f}%{100*qf[~inw].mean():>11.2f}%")
