import pandas as pd
import imageio.v2 as iio
import napari
from pathlib import Path

DATA_PATH_TEMPLATE = 'D:/thesis_research_sandbox/resources/{section_id}'
S10_SECTION_DIR = DATA_PATH_TEMPLATE.format(section_id='GSM8199188_ID61-ID62_S10')
S18_SECTION_DIR = DATA_PATH_TEMPLATE.format(section_id='GSM8199189_ID67-ID68_S18')


def run_manual_alignment(section_dir: str):
    base_dir = Path(section_dir)
    fov_csv = base_dir / "fov_positions_file.csv"
    cell_composite_dir = base_dir / "CellComposite"

    fov_df = pd.read_csv(fov_csv)

    x_min = fov_df['x_global_px'].min()
    y_min = fov_df['y_global_px'].min()

    y_max = fov_df['y_global_px'].max()

    print(f"Offsetting by: x_min={x_min}, y_min={y_min}")

    viewer = napari.Viewer()

    for _, row in fov_df.iterrows():
        fov_id = row["fov"]

        x0 = row["x_global_px"] - x_min
        y0 = (y_max - row["y_global_px"])

        img_path = fov_to_filename(cell_composite_dir, fov_id)
        if not img_path.exists():
            print("Missing image for FOV", fov_id, "->", img_path)
            continue

        img = iio.imread(img_path)

        layer = viewer.add_image(
            img,
            name=f"FOV_{fov_id}",
            translate=(y0, x0),  # Napari uses (y, x) / (row, col)
            blending="additive",
        )

    napari.run()

def fov_to_filename(cell_composite_dir: Path, fov_id: int) -> Path:
    return cell_composite_dir / f"CellComposite_F{int(fov_id):03d}.jpg"


run_manual_alignment(S10_SECTION_DIR)