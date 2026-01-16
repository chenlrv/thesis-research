import pandas as pd
import matplotlib.pyplot as plt



def generate_fov_arrangement_plot(fov_positions_path: str) -> None:
    """Generate a plot showing the arrangement of FOVs based on their global positions.

    Args:
        fov_path (str): Path to the CSV file containing FOV positions.
    """
    fov_df = pd.read_csv(fov_positions_path)

    FOV_COL = "fov"
    X_COL = "x_global_px"
    Y_COL = "y_global_px"

    fov_df[FOV_COL] = (
        fov_df[FOV_COL].astype(str)
        .str.replace(r"^FOV[_ ]*", "", regex=True)
    )
    fov_df[FOV_COL] = pd.to_numeric(fov_df[FOV_COL], errors="coerce").astype("Int64")


    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    x = fov_df[X_COL].to_numpy(float)
    y = fov_df[Y_COL].to_numpy(float)
    labels = fov_df[FOV_COL].to_numpy()

    ax.scatter(x, y, s=100, marker="s", c="#cfe8f3", edgecolors="none")

    for xi, yi, label in zip(x, y, labels):
        if pd.isna(label):
            continue
        ax.text(xi, yi, str(int(label)), ha="center", va="center", fontsize=3, color="black")

    ax.set_title("FOV Arrangement", fontsize=16, fontweight="bold")
    ax.set_xlabel("Global X Position (px)")
    ax.set_ylabel("Global Y Position (px)")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()
