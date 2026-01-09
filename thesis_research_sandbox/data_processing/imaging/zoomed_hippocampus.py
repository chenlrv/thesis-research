import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

DATA_PATH_TEMPLATE = 'D:/thesis_research_sandbox/resources/{section_id}'
S10_SECTION_DIR = DATA_PATH_TEMPLATE.format(section_id='GSM8199188_ID61-ID62_S10')
S18_SECTION_DIR = DATA_PATH_TEMPLATE.format(section_id='GSM8199189_ID67-ID68_S18')
FIGURES_DIR = '../../resources/figures'

cells_s10 = pd.read_csv(f"{S10_SECTION_DIR}\\metadata_file.csv")
cells_s18 = pd.read_csv(f"{S18_SECTION_DIR}\\metadata_file.csv")


def create_zoomed_figure(figure_id: str, cells_metadata: pd.DataFrame, zoom_xmin: int, zoom_xmax: int, zoom_ymin: int, zoom_ymax: int):
    x_min = cells_metadata["CenterX_global_px"].min()
    y_max = cells_metadata["CenterY_global_px"].max()

    cells_metadata["x_plot"] = cells_metadata["CenterX_global_px"] - x_min
    cells_metadata["y_plot"] = y_max - cells_metadata["CenterY_global_px"]

    fig = plt.figure(figsize=(20, 30), dpi=300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.add_axes(ax)


    ax.scatter(cells_metadata["x_plot"], cells_metadata["y_plot"],
               s=70,
               color="black",
               alpha=0.7,
               edgecolors='none')

    ax.set_aspect("equal")
    ax.set_xlim(zoom_xmin, zoom_xmax)
    ax.set_ylim(zoom_ymin, zoom_ymax)
    ax.invert_yaxis()

    plt.savefig(f"{FIGURES_DIR}/{figure_id}_hippocampus_full.png", dpi=300, facecolor='white')
    # add_color()
    # rotate()

def add_color():
    regions_polygons = {
        'ISO': [  # Orange - upper right
            (41000, 80000),
            (52000, 80000),
            (52000, 87000),
            (48000, 90000),
            (41000, 88000),
        ],

        'WM': [  # Blue - left and wrapping around
            (28000, 80000),
            (32000, 80000),
            (35000, 85000),
            (38000, 88000),
            (40000, 92000),
            (38000, 95000),
            (35000, 98000),
            (32000, 100000),
            (28000, 100000),
        ],

        'HIP': [  # Green - upper right hippocampus
            # (44000, 88000),
            # (50000, 88000),
            # (50000, 95000),
            # (48000, 98000),
            # (44000, 98000),
            # (38000, 90000),
            # (44000, 90000),
            # (44000, 98000),
            # (42000, 100000),
            # (38000, 98000),
            # (32000, 92000),
            # (38000, 92000),
            # (38000, 100000),
            # (35000, 104000),
            # (32000, 102000),
            # (36000, 100000),
            # (44000, 100000),
            # (44000, 108000),
            # (40000, 110000),
            (36000, 108000),
        ],

        'BS': [  # Red - brain stem (lower left)
            (28000, 108000),
            (38000, 108000),
            (38000, 120000),
            (28000, 120000),
        ],
    }


    def point_in_polygon(x, y, polygon):
        """Check if point (x, y) is inside polygon"""
        path = Path(polygon)
        return path.contains_point((x, y))


    def assign_region_by_polygon(row):
        """Assign region based on polygon membership"""
        x = row["x_plot"]
        y = row["y_plot"]

        # Check each region in priority order
        for region, polygon in regions_polygons.items():
            if point_in_polygon(x, y, polygon):
                return region

        # Default to WM if not in any polygon
        return "WM"


    # Apply the assignment
    cells_s10["brain_region"] = cells_s10.apply(assign_region_by_polygon, axis=1)

    print("\nRegion distribution:")
    print(cells_s10["brain_region"].value_counts())

    region_colors = {
        'WM': '#4472C4',  # Blue
        'ISO': '#ED7D31',  # Orange
        'HIP': '#70AD47',  # Green
        'BS': '#C55A5A',  # Red
    }

    # Create colored figure
    fig = plt.figure(figsize=(20, 30), dpi=300, facecolor='white')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.add_axes(ax)

    # Plot each region with its color
    for region in ['WM', 'ISO', 'HIP', 'BS']:
        if region in cells_s10["brain_region"].values:
            region_cells = cells_s10[cells_s10["brain_region"] == region]
            color = region_colors.get(region, '#999999')

            ax.scatter(region_cells["x_plot"], region_cells["y_plot"],
                       s=70,
                       color=color,
                       alpha=0.7,
                       edgecolors='none',
                       rasterized=True)

    ax.set_aspect("equal")
    # ax.set_xlim(zoom_xmin, zoom_xmax)
    # ax.set_ylim(zoom_ymin, zoom_ymax)
    ax.invert_yaxis()

    plt.savefig("hippocampus_colored.png", dpi=300, facecolor='white', edgecolor='white')
    plt.close()
    print("Saved: hippocampus_colored.png")

create_zoomed_figure('S10_bottom', cells_s10,zoom_xmin=28_000, zoom_xmax=52_000, zoom_ymin=80_000, zoom_ymax=120_000)
# create_zoomed_figure('S10_up', cells_s10,zoom_xmin=0, zoom_xmax=25_000, zoom_ymin=0, zoom_ymax=35_000)

