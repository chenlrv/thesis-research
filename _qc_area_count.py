import pandas as pd

# slice -> FOVs is read from each sample's *_fov_slices.csv (slice, start, end ranges).
SAMPLES = ["L321", "L34"]
BASE = r"d:\thesis-research\resources\cosmx"


def slice_to_fovs(sample: str) -> dict[int, set[int]]:
    df = pd.read_csv(rf"{BASE}\{sample}\{sample}_fov_slices.csv")
    mapping: dict[int, set[int]] = {}
    for _, row in df.iterrows():
        fovs = mapping.setdefault(int(row["slice"]), set())
        fovs.update(range(int(row["start"]), int(row["end"]) + 1))
    return mapping


for sample in SAMPLES:
    meta = pd.read_csv(
        rf"{BASE}\{sample}\{sample}_metadata_file.csv",
        usecols=["fov", "qcFlagsCellArea"],
    )
    print(f"\n===== {sample} =====")
    print("qcFlagsCellArea unique values:", meta["qcFlagsCellArea"].unique())

    for slice_id, fovs in sorted(slice_to_fovs(sample).items()):
        s = meta[meta["fov"].isin(fovs)]
        n = len(s)
        fail = (s["qcFlagsCellArea"] == "Fail").sum()
        pct = fail / n * 100 if n else 0.0
        print(f"\n--- Slice {slice_id} | total cells: {n} ---")
        print(s["qcFlagsCellArea"].value_counts(dropna=False).to_string())
        print(f"Slice {slice_id} cells FAILING qcFlagsCellArea: {fail} ({pct:.2f}%)")
