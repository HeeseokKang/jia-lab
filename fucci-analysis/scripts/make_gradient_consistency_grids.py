from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile


RAW_ROOT = Path(
    "/mnt/nas1/Projects/Voltage_CellCycle/Data/20260413_FUCCI_Timelapse/"
    "timelapse_plus_bf_2026-04-16_12-11-45.768458"
)
VALIDATION_ROOT = Path(
    "/home/heeseok/github/jia-lab/fucci-analysis/analysis/20260413_validation"
)
CORRECTED_ROOT = VALIDATION_ROOT / "corrected"
PLOTS_DIR = VALIDATION_ROOT / "plots"

WELLS = ("R0_0", "R0_8", "R1_0")
TIMEPOINTS = (0, 30, 66)
CELLLINE = "3T3"
Z = 0


def image_path(root: Path, timepoint: int, well: str, channel: str) -> Path:
    return root / str(timepoint) / f"{well}_{Z}_{CELLLINE}_FUCCI_{channel}.tiff"


def auto_contrast(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.7))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def make_grid(
    source_root: Path,
    channel: str,
    title: str,
    out_name: str,
    cmap: str,
) -> Path:
    paths = [
        [image_path(source_root, tp, well, channel) for tp in TIMEPOINTS]
        for well in WELLS
    ]
    missing = [str(p) for row in paths for p in row if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required images:\n" + "\n".join(missing))

    fig, axes = plt.subplots(
        len(WELLS), len(TIMEPOINTS), figsize=(10.5, 10.5), constrained_layout=True
    )
    for r, well in enumerate(WELLS):
        for c, tp in enumerate(TIMEPOINTS):
            ax = axes[r, c]
            img = tifffile.imread(paths[r][c])
            if img.ndim != 2:
                raise ValueError(
                    f"Expected 2D image, got shape {img.shape}: {paths[r][c]}"
                )
            ax.imshow(
                auto_contrast(img),
                cmap=cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={tp}", fontsize=13)
            if c == 0:
                ax.set_ylabel(well, fontsize=13, rotation=0, labelpad=28, va="center")
            ax.text(
                0.02,
                0.98,
                f"{well}, t={tp}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=2),
            )

    fig.suptitle(title, fontsize=16)
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        make_grid(
            RAW_ROOT,
            "647",
            "647 raw: gradient consistency across wells and timepoints "
            "(auto-contrast per panel)",
            "gradient_consistency_647_wells_timepoints.png",
            "gray",
        ),
        make_grid(
            RAW_ROOT,
            "561",
            "561 raw: no structured gradient across wells and timepoints "
            "(auto-contrast per panel)",
            "gradient_consistency_561_wells_timepoints.png",
            "gray",
        ),
        make_grid(
            CORRECTED_ROOT,
            "647",
            "647 corrected: gradient removed across wells and timepoints "
            "(auto-contrast per panel)",
            "correction_consistency_647_wells_timepoints.png",
            "gray",
        ),
    ]
    for out in outputs:
        print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
