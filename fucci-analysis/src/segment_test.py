"""Single-image Cellpose-SAM segmentation smoke test.

Runs Cellpose-SAM on one BF frame (one well, one timepoint), saves the label
mask as .npy, and writes a QC overlay PNG (BF + mask outline) so we can eyeball
segmentation quality before scaling up.

Adjust WELL_ID / TIMEPOINT below to retest a different field.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from cellpose import models
from cellpose import utils as cp_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.paths import RESULT_ROOT  # noqa: E402


# Read raw images directly from local scratch (faster than NAS for Cellpose I/O).
DATA_ROOT = Path(
    "/data/Project_Data/Voltage_CellCycle/Data/20260413_FUCCI_Timelapse/"
    "timelapse_plus_bf_2026-04-16_12-11-45.768458"
)

# Test target: timepoint folders for this dataset go from 0 to 66 (inclusive).
WELL_ID = "R1_1"
TIMEPOINT = 30
CELLLINE = "3T3"
Z = 0
CHANNEL = "BF"

OUTPUT_DIR = RESULT_ROOT / "segmentation_test"


def auto_contrast(image: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.7) -> np.ndarray:
    arr = image.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, (lo_pct, hi_pct))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def main() -> None:
    img_path = (
        DATA_ROOT
        / str(TIMEPOINT)
        / f"{WELL_ID}_{Z}_{CELLLINE}_FUCCI_{CHANNEL}.tiff"
    )
    if not img_path.exists():
        raise FileNotFoundError(f"Input BF image not found: {img_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{WELL_ID}_t{TIMEPOINT:03d}_{CHANNEL}"

    print(f"[LOAD] {img_path}")
    img = tifffile.imread(str(img_path))
    if img.ndim != 2:
        raise ValueError(f"Expected 2D BF image, got shape {img.shape}: {img_path}")

    print("[CELLPOSE] Loading Cellpose-SAM (cpsam) on GPU if available")
    model = models.CellposeModel(gpu=True, model_type="cpsam")

    print("[CELLPOSE] Running eval (diameter=None, default thresholds)")
    masks, flows, styles = model.eval(img, diameter=None)

    n_cells = int(masks.max())
    print(f"[RESULT] Cells detected: {n_cells}")

    mask_path = OUTPUT_DIR / f"{stem}_mask.npy"
    np.save(mask_path, masks.astype(np.int32))
    print(f"[SAVED] mask  -> {mask_path}")

    outlines = cp_utils.masks_to_outlines(masks)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    ax.imshow(auto_contrast(img), cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    # Red outline as RGBA so the BF underneath stays visible.
    overlay = np.zeros((*outlines.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = outlines.astype(np.float32)
    ax.imshow(overlay, interpolation="nearest")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{stem} | Cellpose-SAM | {n_cells} cells", fontsize=12)

    overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"
    fig.savefig(overlay_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
