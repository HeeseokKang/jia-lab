from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# Add repository root to import path so configs resolve consistently.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import RESULT_ROOT  # noqa: E402


BACKGROUND_DIR = RESULT_ROOT / "background"
PLOTS_DIR = RESULT_ROOT / "plots"
CHANNELS = ("561", "647")


def load_channel_df(channel: str) -> pd.DataFrame:
    csv_path = BACKGROUND_DIR / f"quadrant_background_{channel}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Background CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def compute_metrics(df: pd.DataFrame, channel: str) -> dict[str, float | str]:
    row_means = df.groupby("row")["image_mean"].mean()
    col_means = df.groupby("col")["image_mean"].mean()
    well_means = df.groupby(["row", "col"])["image_mean"].mean()
    tp_std = df.groupby("timepoint")["image_mean_median_subtracted"].std()

    return {
        "channel": channel,
        "row_gradient_range": float(row_means.max() - row_means.min()),
        "col_gradient_range": float(col_means.max() - col_means.min()),
        "well_nonuniform_std": float(well_means.std()),
        "tp_mediansub_std_mean": float(tp_std.mean()),
    }


def plot_heatmaps(channel_tables: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(1, len(channel_tables), figsize=(6 * len(channel_tables), 5))
    if len(channel_tables) == 1:
        axes = [axes]

    for ax, channel in zip(axes, CHANNELS):
        df = channel_tables[channel]
        pivot = (
            df.groupby(["row", "col"], as_index=False)["image_mean"]
            .mean()
            .pivot(index="row", columns="col", values="image_mean")
            .sort_index()
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="inferno")
        ax.set_title(f"Channel {channel}: mean image intensity")
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns.tolist())
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.tolist())
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Row/Col background gradient heatmap", fontsize=13)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "background_gradient_heatmap_561_647.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    tables = {ch: load_channel_df(ch) for ch in CHANNELS}
    metrics = [compute_metrics(tables[ch], ch) for ch in CHANNELS]
    metrics_df = pd.DataFrame(metrics)
    metrics_csv = BACKGROUND_DIR / "background_gradient_metrics_561_647.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    heatmap_path = plot_heatmaps(tables)

    print(f"[DONE] Metrics CSV: {metrics_csv}")
    print(f"[DONE] Heatmap plot: {heatmap_path}")
    for row in metrics:
        print(
            "[METRIC] "
            f"ch={row['channel']} "
            f"row_range={row['row_gradient_range']:.3f} "
            f"col_range={row['col_gradient_range']:.3f} "
            f"well_std={row['well_nonuniform_std']:.3f} "
            f"tp_mediansub_std_mean={row['tp_mediansub_std_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
