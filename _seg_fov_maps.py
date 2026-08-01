import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = {
    "L321": r"resources/cosmx/L321/L321_metadata_file.csv",
    "L34":  r"resources/cosmx/L34/L34_metadata_file.csv",
}
CHANS = ["DAPI", "Membrane"]          # the two channels that actually drive segmentation
WEAK = 0.5                            # FOV flagged if mean < WEAK * slide median
OUT = "seg_qc_plots"
os.makedirs(OUT, exist_ok=True)

for tag, path in FILES.items():
    cols = ["fov", "CenterX_global_px", "CenterY_global_px"] + [f"Mean.{c}" for c in CHANS]
    df = pd.read_csv(path, usecols=cols)
    g = df.groupby("fov")
    fov = pd.DataFrame({
        "x": g["CenterX_global_px"].mean(),
        "y": g["CenterY_global_px"].mean(),
        "ncell": g.size(),
    })
    for c in CHANS:
        fov[c] = g[f"Mean.{c}"].mean()

    fig, axes = plt.subplots(1, len(CHANS), figsize=(7 * len(CHANS), 6.2))
    for ax, c in zip(axes, CHANS):
        med = fov[c].median()
        rel = fov[c] / med                       # FOV intensity relative to slide median
        sc = ax.scatter(fov.x, fov.y, c=rel, s=14, cmap="viridis",
                        vmin=0.3, vmax=1.7, edgecolors="none")
        weak = rel < WEAK
        ax.scatter(fov.x[weak], fov.y[weak], s=60, facecolors="none",
                   edgecolors="red", linewidths=1.2,
                   label=f"weak (<{WEAK}x): {int(weak.sum())} FOVs")
        ax.set_title(f"{tag}  Mean.{c}  (median={med:.0f})")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(sc, ax=ax, shrink=0.8, label="FOV mean / slide median")
    fig.tight_layout()
    out = os.path.join(OUT, f"{tag}_fov_intensity_DAPI_Membrane.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[{tag}] wrote {out}")

    # list weakest FOVs per channel
    for c in CHANS:
        rel = (fov[c] / fov[c].median())
        weak = rel[rel < WEAK].sort_values()
        print(f"  {c}: {len(weak)} weak FOVs (<{WEAK}x median); weakest 10:")
        for fid, r in weak.head(10).items():
            print(f"     FOV {int(fid):>4}  {r:0.2f}x median   (ncell={int(fov.loc[fid,'ncell'])})")
    print()
