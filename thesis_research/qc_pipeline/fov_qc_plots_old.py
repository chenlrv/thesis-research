def general(adata):
    fov_df = _get_fov_df_from_obs(adata)
    fov_df = _add_neighbor_baseline_and_log2fc(fov_df, k_neighbors=8)
    fov_df, ratio_threshold = _flag_significant_signal_loss(fov_df, max_totalcounts_loss=0.8)
    subtitle = f"Signal loss threshold: > {int(ratio_threshold * 100)}%"
    plot_fovqc_like_scratchspace(fov_df, subtitle=subtitle)
    return fov_df


def _get_fov_df_from_obs(adata: AnnData) -> pd.DataFrame:
    fov_col = "fov"
    x_col = "CenterX_global_px"
    y_col = "CenterY_global_px"
    ncount_col = "nCount"

    cols = [fov_col, x_col, y_col, ncount_col]
    df = adata.obs[cols].copy()

    centers = (
        df.dropna(subset=[fov_col, x_col, y_col])
        .groupby(fov_col)[[x_col, y_col]]
        .mean()
        .rename(columns={x_col: "x", y_col: "y"})
    )

    metrics = df.dropna(subset=[fov_col]).groupby(fov_col)[ncount_col].first()

    fov_df = centers.join(metrics, how="inner").reset_index()
    return fov_df


def _add_neighbor_baseline_and_log2fc(fov_df: pd.DataFrame, k_neighbors: int = 8) -> pd.DataFrame:
    counts_col = "nCount"
    df = fov_df.copy().dropna(subset=["x", "y", counts_col])
    pts = df[["x", "y"]].to_numpy(float)

    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(df))).fit(pts)
    _, idx = nn.kneighbors(pts)

    baselines = []
    for i in range(len(df)):
        neighbours = idx[i, 1:]  # skip itself
        baselines.append(np.median(df.iloc[neighbours][counts_col].to_numpy(float)))

    df["baseline_nCount"] = np.asarray(baselines)
    df["ratio"] = df[counts_col] / np.clip(df["baseline_nCount"], 1e-9, None)
    df["log2fc"] = np.log2(np.clip(df["ratio"], 1e-9, None))
    return df


def _flag_significant_signal_loss(df: pd.DataFrame, max_totalcounts_loss: float):
    ratio_threshold = 1.0 - max_totalcounts_loss  # 0.8 for 20% max loss
    new_df = df.copy()
    new_df["is_significant_signal_loss"] = new_df["ratio"] < ratio_threshold
    return new_df, ratio_threshold


def _plot_fov_qc_like_scratchspace(df: pd.DataFrame, subtitle: str) -> None:
    v = df["log2fc"].to_numpy(float)
    bins = [-np.inf, -2, -1, 0, 1, np.inf]
    labels = ["<-2", "-1", "0", "1", ">2"]
    cat = pd.cut(v, bins=bins, labels=labels)

    color_map = {
        ">2": "#b2182b",
        "1": "#ef3b2c",
        "0": "#bdbdbd",
        "-1": "#3182bd",
        "<-2": "#08519c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    ax1, ax2 = axes

    # --- Left: log2FC compared to similar regions ---
    ax1.scatter(df["x"], df["y"], s=850, marker="s", c="white", edgecolors="black", linewidths=0.7)

    for lab in labels:
        m = cat.astype(str) == lab
        ax1.scatter(df.loc[m, "x"], df.loc[m, "y"], s=6, c=color_map[lab], alpha=0.85, linewidths=0)

    ax1.set_title(
        "Log2 fold-change in total counts compared to similar regions",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_xlabel("X_px")
    ax1.set_ylabel("Y_px")
    ax1.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color_map[k], markersize=6, label=k
        )
        for k in [">2", "1", "0", "-1", "<-2"]
    ]
    ax1.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    # --- Right: flagged FOVs ---
    ax2.scatter(
        df["x"], df["y"], s=850, marker="s", c="#9ecae1", edgecolors="black", linewidths=0.7
    )

    flagged = df["flag_signal_loss"].to_numpy(bool)
    ax2.scatter(
        df.loc[flagged, "x"],
        df.loc[flagged, "y"],
        s=850,
        marker="s",
        c="#e377c2",
        edgecolors="black",
        linewidths=0.9,
    )

    for _, r in df.loc[flagged].iterrows():
        ax2.text(
            r["x"],
            r["y"],
            str(int(r["fov"])),
            ha="center",
            va="center",
            fontsize=9,
            color="green",
            fontweight="bold",
        )

    ax2.set_title("Flagged FOVs", fontsize=10, fontweight="bold")
    ax2.set_xlabel("X_px")
    ax2.set_ylabel("Y_px")
    ax2.set_aspect("equal", adjustable="box")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
