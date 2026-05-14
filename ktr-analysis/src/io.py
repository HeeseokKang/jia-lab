"""Dataset index: map KTR timelapse TIFF paths to metadata columns."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage import filters, morphology

# R{region}_{fov}_{z}_YYYYMMDD_<name>_<channel>_KTR.tiff
# Example: R0_1_0_20260505_Heeseok_mKate2_KTR.tiff
_KTR_FILENAME = re.compile(
    r"^R(?P<region>\d+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<date>\d{8})_(?P<name>.+)_(?P<channel>BF|mKate2|mTagBFP2)_KTR\.(?:tiff|tif)$",
    re.IGNORECASE,
)

_CANONICAL_CHANNEL = {
    "bf": "BF",
    "mkate2": "mKate2",
    "mtagbfp2": "mTagBFP2",
}

_FOV_CHANNELS: tuple[str, ...] = ("BF", "mTagBFP2", "mKate2")

IMAGE_SUFFIXES = (".tif", ".tiff")


def parse_dataset(data_root: Path | str) -> pd.DataFrame:
    """Scan *data_root* for KTR-style TIFFs and return a tidy file table.

    Filename shape:
    ``R{{region}}_{{fov}}_{{z}}_YYYYMMDD_<name>_{{channel}}_KTR.tiff``

    *timepoint* is inferred from the first all-numeric path segment under
    *data_root* (same as before), e.g. ``.../12/R0_1_0_..._KTR.tiff`` → ``12``.

    Columns: ``timepoint``, ``region`` (``R0``, ``R1``, …), ``fov``, ``z``,
    ``channel`` (``BF``, ``mKate2``, ``mTagBFP2``), ``filepath`` (absolute).
    """
    root = Path(data_root).expanduser().resolve()
    rows: list[dict[str, object]] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        match = _KTR_FILENAME.match(path.name)
        if match is None:
            continue

        timepoint = _infer_timepoint(path, root)
        if timepoint is None:
            continue

        region_id = int(match.group("region"))
        channel_raw = match.group("channel")
        channel = _CANONICAL_CHANNEL[channel_raw.lower()]

        rows.append(
            {
                "timepoint": timepoint,
                "region": f"R{region_id}",
                "fov": int(match.group("fov")),
                "z": int(match.group("z")),
                "channel": channel,
                "filepath": str(path.resolve()),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["timepoint", "region", "fov", "z", "channel", "filepath"]
        )

    df = df.astype({"timepoint": "int64", "fov": "int64", "z": "int64"})
    return df.sort_values(
        ["timepoint", "region", "fov", "z", "channel", "filepath"],
        kind="mergesort",
    ).reset_index(drop=True)


def get_fov_timeseries(df: pd.DataFrame, region: str, fov: int) -> dict[str, list[str]]:
    """Build ordered filepath lists for one FOV across timepoints.

    Rows are filtered to *region* and *fov*. For each channel in
    ``BF``, ``mTagBFP2``, ``mKate2``, paths are sorted by *timepoint*
    ascending. If several *z* slices share the same *timepoint*, the
    smallest *z* is kept so each timepoint maps to one 2D frame.
    """
    sub = df[(df["region"] == region) & (df["fov"] == fov)]
    out: dict[str, list[str]] = {}
    for ch in _FOV_CHANNELS:
        ch_df = sub.loc[sub["channel"] == ch, ["timepoint", "z", "filepath"]].sort_values(
            ["timepoint", "z"], kind="mergesort"
        )
        ch_df = ch_df.drop_duplicates(subset=["timepoint"], keep="first")
        out[ch] = ch_df["filepath"].astype(str).tolist()
    return out


def load_fov_stack(df: pd.DataFrame, region: str, fov: int) -> dict[str, np.ndarray]:
    """Load a single-FOV timelapse stack per channel as *(T, H, W)* arrays.

    Uses :func:`get_fov_timeseries` then reads each path with ``tifffile.imread``.
    Raises ``ValueError`` if any expected channel is missing (no paths) or if
    channel timepoint counts differ.
    """
    series = get_fov_timeseries(df, region, fov)
    counts = {ch: len(series[ch]) for ch in _FOV_CHANNELS}
    missing = [ch for ch, n in counts.items() if n == 0]
    if missing:
        raise ValueError(
            f"No TIFF paths for region={region!r}, fov={fov}; "
            f"missing channels: {', '.join(missing)}"
        )
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        detail = ", ".join(f"{ch}={counts[ch]}" for ch in _FOV_CHANNELS)
        raise ValueError(
            f"Timepoint count mismatch for region={region!r}, fov={fov}: {detail}"
        )

    stacks: dict[str, np.ndarray] = {}
    for ch, paths in series.items():
        planes = [tifffile.imread(p) for p in paths]
        stacks[ch] = np.stack(planes, axis=0)
    return stacks


def _normalize_for_display(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Scale one frame to [0, 1] using percentile clipping (display only)."""
    lo, hi = np.percentile(image, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    scaled = (image.astype(np.float32) - float(lo)) / float(hi - lo)
    return np.clip(scaled, 0.0, 1.0)


def _segment_nuclei_placeholder(image: np.ndarray) -> np.ndarray:
    """Return a simple nucleus mask placeholder for QC overlays."""
    blur = filters.gaussian(image.astype(np.float32), sigma=1.0, preserve_range=True)
    threshold = filters.threshold_otsu(blur)
    mask = blur > threshold
    mask = morphology.binary_opening(mask, morphology.disk(1))
    return morphology.remove_small_objects(mask, min_size=50)


def show_fov_qc(
    df: pd.DataFrame,
    region: str,
    fov: int,
    timepoints: list[int] | None = None,
    figsize: tuple[float, float] = (14, 8),
) -> matplotlib.figure.Figure:
    """Visual QC panel for one FOV across selected timepoints.

    Builds a 3-column grid (BF, mTagBFP2, mKate2) for each selected timepoint.
    Percentile normalization is used for display only. The mTagBFP2 panel includes
    a placeholder nucleus mask overlay to quickly inspect nuclear signal quality.
    """
    stacks = load_fov_stack(df, region, fov)
    n_timepoints = stacks["BF"].shape[0]

    if timepoints is None:
        selected = sorted({0, n_timepoints // 2, n_timepoints - 1})
    else:
        selected = sorted(set(int(tp) for tp in timepoints))
        bad = [tp for tp in selected if tp < 0 or tp >= n_timepoints]
        if bad:
            raise ValueError(
                f"Invalid timepoints for region={region!r}, fov={fov}: {bad}. "
                f"Valid range is [0, {n_timepoints - 1}]."
            )

    fig, axes = plt.subplots(len(selected), 3, figsize=figsize, squeeze=False)
    channels = ("BF", "mTagBFP2", "mKate2")

    for row_idx, tp in enumerate(selected):
        for col_idx, ch in enumerate(channels):
            ax = axes[row_idx, col_idx]
            frame = stacks[ch][tp]
            frame_norm = _normalize_for_display(frame)
            ax.imshow(frame_norm, cmap="gray")

            if ch == "mTagBFP2":
                nuclei_mask = _segment_nuclei_placeholder(frame)
                ax.imshow(nuclei_mask, cmap="autumn", alpha=0.35, interpolation="nearest")

            ax.set_title(f"{region} | FOV {fov} | t={tp} | {ch}", fontsize=10)
            ax.axis("off")

    fig.tight_layout()
    return fig


def _infer_timepoint(path: Path, root: Path) -> int | None:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return None
    for part in rel_parts[:-1]:
        if part.isdigit():
            return int(part)
    return None


if __name__ == "__main__":
    data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
    table = parse_dataset(data_root)
    figure = show_fov_qc(table, region="R0", fov=0)
    plt.show()
