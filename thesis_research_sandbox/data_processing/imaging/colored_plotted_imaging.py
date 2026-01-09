import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

DATA_PATH_TEMPLATE = 'D:/thesis_research_sandbox/resources/{section_id}'
S10_SECTION_DIR = DATA_PATH_TEMPLATE.format(section_id='GSM8199188_ID61-ID62_S10')
S18_SECTION_DIR = DATA_PATH_TEMPLATE.format(section_id='GSM8199189_ID67-ID68_S18')
FIGURES_DIR = '/resources/figures'

cells_s10 = pd.read_csv(f"{S10_SECTION_DIR}\\metadata_file.csv")


def generate_colored_plotted_image(
        fov_positions_file: str,
        metadata_file: str,
        shapes_file: str,
        zoom_xlim: tuple = None,  # e.g. (30000, 45000)
        zoom_ylim: tuple = None,  # e.g. (85000, 105000)
        rotate_deg: float = 0,  # 0, 90, -90, 180
        mirror: str = None,
        fig_size: tuple[int, int] = (20, 28),
        point_size: int = 10,
        dpi: int = 150,
        y_shift: int = 4200):
    fov_df = pd.read_csv(fov_positions_file)

    x_min = fov_df['x_global_px'].min()
    y_max = fov_df['y_global_px'].max()

    shapes_df = pd.read_csv(shapes_file)
    shapes_df["napari_x"] = shapes_df["axis-1"]
    shapes_df["napari_y"] = shapes_df["axis-0"]

    cells_df = pd.read_csv(metadata_file)
    cells_df["napari_x"] = cells_df["CenterX_global_px"] - x_min
    cells_df["napari_y"] = y_max - cells_df["CenterY_global_px"] + y_shift

    region_polygons = {}
    for shape_name, shape_data in shapes_df.groupby("index"):
        points = list(zip(shape_data["napari_x"], shape_data["napari_y"]))
        region_polygons[shape_name] = MplPath(points)

    cells_df["region"] = assign_all_regions(cells_df, region_polygons)

    print("\nRegion counts:")
    region_counts = cells_df["region"].value_counts()
    print(region_counts)

    region_colors = {
        0: '#70AD47', 1: '#4472C4', 2: '#ED7D31',
        3: '#C55A5A',
        4: '#70AD47', 5: '#4472C4', 6: '#ED7D31',
        7: '#C55A5A',

        'Unassigned': '#CCCCCC'
    }

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    x = cells_df["napari_x"].values
    y = cells_df["napari_y"].values
    if rotate_deg != 0:
        x_rot, y_rot = rotate_points(x, y, rotate_deg)
        cells_df["plot_x"] = x_rot
        cells_df["plot_y"] = y_rot
    else:
        cells_df["plot_x"] = cells_df["napari_x"]
        cells_df["plot_y"] = cells_df["napari_y"]

    # Plot each region
    for region_name, color in region_colors.items():
        region_cells = cells_df[cells_df["region"] == region_name]
        if len(region_cells) > 0:
            ax.scatter(
                region_cells["plot_x"],
                region_cells["plot_y"],
                s=point_size,
                color=color,
                alpha=0.8,
                label=f"{region_name} ({len(region_cells):,})",
                edgecolors='none',
                rasterized=True
            )

    # ---- Optional: Apply mirroring ----
    if mirror == "horizontal":
        ax.invert_xaxis()
    elif mirror == "vertical":
        ax.invert_yaxis()

    # ---- Optional: Apply rotation ----

    x = cells_df["napari_x"].values
    y = cells_df["napari_y"].values

    # ---- Optional: zoom ----
    if zoom_xlim is not None:
        ax.set_xlim(zoom_xlim)
    if zoom_ylim is not None:
        ax.set_ylim(zoom_ylim)

    # ---- Final formatting ----
    ax.set_aspect("equal")
    ax.invert_yaxis()
    # ax.legend(loc='upper right', fontsize=14, markerscale=2)
    ax.set_title("Cells Colored by Region", fontsize=18, weight='bold')
    ax.set_xlabel("X (napari coordinates)")
    ax.set_ylabel("Y (napari coordinates)")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    # ---- Optional save ----
    # if save_path is not None:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    #     print(f"\n✓ Saved to: {save_path}")

    print("\n✓ Plot saved as 'cells_colored_final.png'")


def assign_all_regions(cells_df, polygons):
    points = list(zip(cells_df["napari_x"], cells_df["napari_y"]))
    regions = ["Unassigned"] * len(cells_df)

    for region_name, polygon in polygons.items():
        mask = polygon.contains_points(points)
        for i, is_inside in enumerate(mask):
            if is_inside and regions[i] == "Unassigned":
                regions[i] = region_name

    return regions

def rotate_points(x, y, angle_deg):
    """
    Rotate points (x, y) around their center by angle_deg.
    Positive angle = counter-clockwise.
    Returns x_rot, y_rot as np.arrays.
    """
    theta = np.deg2rad(angle_deg)
    x = np.asarray(x)
    y = np.asarray(y)

    cx = 0.5 * (x.min() + x.max())
    cy = 0.5 * (y.min() + y.max())

    xr = x - cx
    yr = y - cy

    x_rot = xr * np.cos(theta) - yr * np.sin(theta) + cx
    y_rot = xr * np.sin(theta) + yr * np.cos(theta) + cy
    return x_rot, y_rot

generate_colored_plotted_image(
    fov_positions_file=f"{S10_SECTION_DIR}\\fov_positions_file.csv",
    metadata_file=f"{S10_SECTION_DIR}\\metadata_file.csv",
    shapes_file=f"{S10_SECTION_DIR}\\Shapes_s10_bottom.csv")


generate_colored_plotted_image(
    fov_positions_file=f"{S10_SECTION_DIR}\\fov_positions_file.csv",
    metadata_file=f"{S10_SECTION_DIR}\\metadata_file.csv",
    shapes_file=f"{S10_SECTION_DIR}\\Shapes_s10_up.csv",
    # zoom_xlim=(0, 20000),
    # zoom_ylim=(0, 40000),
    rotate_deg=90,
    mirror = 'horizontal',
    fig_size = (14, 8),
    point_size = 3,
    dpi = 200
)

generate_colored_plotted_image(
    fov_positions_file=f"{S18_SECTION_DIR}\\fov_positions_file.csv",
    metadata_file=f"{S18_SECTION_DIR}\\metadata_file.csv",
    shapes_file=f"{S18_SECTION_DIR}\\Shapes_s18_rotated.csv",
    mirror='vertical',
    rotate_deg=270,
    fig_size=(14, 8),
    point_size=3,
    dpi=200,
    y_shift=3500
)