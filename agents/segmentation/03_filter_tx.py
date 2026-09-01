"""
Step 3: Stream-filter the 4.6 GB L321_tx_file.csv down to ONLY the chosen FOVs
(451, 512, 514, 515, 523). Write one CSV per FOV plus a combined CSV.

DO NOT load the whole file. pandas chunksize streaming on the `fov` column.

tx columns: fov, cell_ID, cell, x_local_px, y_local_px, x_global_px,
             y_global_px, z, target, CellComp

Run:
  conda run -n thesis_research python agents/segmentation/03_filter_tx.py
"""
import os
import time
import pandas as pd

TX = (r"D:\20251214_CosMx_ReuvenStein\20251214_CosMx_ReuvenStein.tar\Analysis"
      r"\L321__1__31_12_2025_12_32_59_204\flatFiles\L321\L321_tx_file.csv")
OUT = r"D:\thesis-research\agents\outputs\segmentation\tx_by_fov"
os.makedirs(OUT, exist_ok=True)

CHOSEN = [451, 512, 514, 515, 523]
CHOSEN_SET = set(CHOSEN)

t0 = time.time()
# Accumulate per-FOV frames, write once at the end (these subsets are small).
buckets = {f: [] for f in CHOSEN}
n_total = 0
n_kept = 0
reader = pd.read_csv(TX, chunksize=2_000_000)
for i, ch in enumerate(reader):
    n_total += len(ch)
    sub = ch[ch["fov"].isin(CHOSEN_SET)]
    if len(sub):
        n_kept += len(sub)
        for f, g in sub.groupby("fov"):
            buckets[f].append(g)
    if i % 10 == 0:
        print(f"  chunk {i}: scanned {n_total:,} rows, kept {n_kept:,} "
              f"({time.time()-t0:.0f}s)", flush=True)

print(f"Scanned {n_total:,} transcripts total in {time.time()-t0:.0f}s; "
      f"kept {n_kept:,} in chosen FOVs.")

summary = {}
combined = []
for f in CHOSEN:
    if not buckets[f]:
        print(f"  FOV {f}: NO transcripts found!")
        continue
    df = pd.concat(buckets[f], ignore_index=True)
    path = os.path.join(OUT, f"tx_fov{f}.csv")
    df.to_csv(path, index=False)
    combined.append(df)
    n_assigned = int((df["cell_ID"] != 0).sum())
    summary[f] = {
        "n_tx": int(len(df)),
        "n_assigned": n_assigned,
        "frac_unassigned": float((df["cell_ID"] == 0).mean()),
        "n_cells": int(df.loc[df["cell_ID"] != 0, "cell"].nunique()),
        "n_targets": int(df["target"].nunique()),
    }
    print(f"  FOV {f}: {len(df):,} tx, {summary[f]['n_cells']:,} cells, "
          f"frac_unassigned={summary[f]['frac_unassigned']:.3f} -> {path}")

allc = pd.concat(combined, ignore_index=True)
allc.to_csv(os.path.join(OUT, "tx_all_chosen.csv"), index=False)
print(f"\nWrote combined tx_all_chosen.csv ({len(allc):,} rows)")
print("CellComp value counts (combined):")
print(allc["CellComp"].value_counts(dropna=False).to_string())

import json
with open(os.path.join(OUT, "tx_filter_summary.json"), "w") as fh:
    json.dump({"per_fov": summary, "scan_seconds": time.time() - t0,
               "n_total_scanned": n_total}, fh, indent=2)
print("Wrote tx_filter_summary.json")
