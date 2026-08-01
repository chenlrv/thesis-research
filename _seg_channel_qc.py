import numpy as np
import pandas as pd

FILES = {
    "L321 (slices1-3)": r"resources/cosmx/L321/L321_metadata_file.csv",
    "L34 (slices4-6)":  r"resources/cosmx/L34/L34_metadata_file.csv",
}
CHANS = ["DAPI", "PanCK", "Membrane", "CD45", "G"]

for tag, path in FILES.items():
    mean_cols = [f"Mean.{c}" for c in CHANS]
    max_cols  = [f"Max.{c}"  for c in CHANS]
    df = pd.read_csv(path, usecols=["fov"] + mean_cols + max_cols)
    n = len(df)
    print("=" * 78)
    print(f"{tag}   cells={n:,}   FOVs={df.fov.nunique()}")
    print("-" * 78)
    print(f"{'chan':9}{'med.Mean':>10}{'p5':>8}{'p95':>9}{'%~0':>7}"
          f"{'max.Max':>9}{'%sat':>7}{'FOV-CV':>8}{'FOVmin/med':>11}")
    for c in CHANS:
        m = df[f"Mean.{c}"].to_numpy(float)
        mx = df[f"Max.{c}"].to_numpy(float)
        ceil = mx.max()
        pct_sat = 100 * np.mean(mx >= 0.98 * ceil)        # cells near intensity ceiling
        pct_zero = 100 * np.mean(m <= 1.0)                # cells with ~no signal
        # per-FOV stability of the morphology stain
        fov_mean = df.groupby("fov")[f"Mean.{c}"].mean()
        fov_cv = fov_mean.std() / fov_mean.mean()
        fov_ratio = fov_mean.min() / fov_mean.median()
        print(f"{c:9}{np.median(m):>10.1f}{np.percentile(m,5):>8.1f}"
              f"{np.percentile(m,95):>9.1f}{pct_zero:>7.1f}"
              f"{ceil:>9.0f}{pct_sat:>7.2f}{fov_cv:>8.2f}{fov_ratio:>11.2f}")
    print()
print("Reading guide:")
print(" %~0      = cells with ~zero mean signal (high = weak/absent stain; DAPI should be ~0%)")
print(" %sat     = cells pegged at the intensity ceiling (high = clipped/saturated)")
print(" FOV-CV   = variation of stain across FOVs (low = uniform; >0.4 = uneven illumination/dropout)")
print(" FOVmin/med = weakest FOV vs typical FOV (<<1 = dropout FOVs where seg has no signal)")
