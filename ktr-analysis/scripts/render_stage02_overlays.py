"""Stage 02 smoke-run QC overlays.

Human-validator stage. Reads the smoke masks already on disk under
``<data_root>/analysis/segmentation/validation/R0_fov0/<cell>/`` and the
matching raw TIFFs, then emits publication-readable PNG overlays for
manual inspection of nucleus boundary quality, merge/split behavior,
boundary drift, BF failure modes, and longitudinal plausibility.

Not a quantitative rerun. Not the full 567-frame sweep. Tracking is not
exercised. Per the agreed layout (2026-05-19):

  - H2B cell: (H2B raw + my boundary) | (H2B raw + Bill reference boundary)
  - BF cell:  (BF raw + my boundary)  | (H2B raw + my boundary)
                                         ↑ nuclei vs whole-cell diagnostic

CLI:
    python scripts/render_stage02_overlays.py \
        --data-root /data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from matplotlib.colors import to_rgba
from skimage.morphology import binary_dilation
from skimage.segmentation import find_boundaries

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.io import get_fov_timeseries, parse_dataset  # noqa: E402

REGION = "R0"
FOV = 0
TIMEPOINTS = (0, 100, 566)
CELLS = ("stardist_h2b", "stardist_bf", "cellpose_h2b", "cellpose_bf")

MINE_COLOR = "#ffcc00"
REF_COLOR = "#00ccff"


def _normalize_for_display(image: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(image, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _object_count(labels: np.ndarray) -> int:
    return int(np.unique(labels[labels != 0]).size)


def _boundary_rgba(labels: np.ndarray, color: str) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    if labels.max() == 0:
        return out
    # Outer boundaries, dilated once so they read at print scale on 1200×1200.
    bd = find_boundaries(labels, mode="outer")
    bd = binary_dilation(bd)
    out[bd] = to_rgba(color)
    return out


def _draw_panel(ax, image_norm: np.ndarray, labels: np.ndarray, title: str, color: str) -> None:
    ax.imshow(image_norm, cmap="gray", interpolation="nearest")
    ax.imshow(_boundary_rgba(labels, color), interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def _load_h2b_raw(h2b_paths: list[str], tp: int) -> np.ndarray:
    return _normalize_for_display(tifffile.imread(h2b_paths[tp]))


def _load_bf_raw(bf_paths: list[str], tp: int) -> np.ndarray:
    return _normalize_for_display(tifffile.imread(bf_paths[tp]))


def render_cell_tp(
    cell: str,
    tp: int,
    masks_root: Path,
    h2b_paths: list[str],
    bf_paths: list[str],
    ref_zarr: zarr.Group,
    ref_scene: str,
    out_dir: Path,
) -> Path:
    labels = np.load(masks_root / cell / f"t{tp:04d}.npy")
    n_obj = _object_count(labels)
    is_h2b = cell.endswith("_h2b")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)

    if is_h2b:
        h2b_norm = _load_h2b_raw(h2b_paths, tp)
        _draw_panel(
            axes[0],
            h2b_norm,
            labels,
            f"{cell} (this run)\nH2B raw + boundary   N = {n_obj}",
            MINE_COLOR,
        )
        ref_labels = np.asarray(ref_zarr[ref_scene]["nuc_labels"][tp])
        ref_n = _object_count(ref_labels)
        _draw_panel(
            axes[1],
            h2b_norm,
            ref_labels,
            f"Bill reference (StarDist 2D_versatile_fluo)\nH2B raw + boundary   N = {ref_n}",
            REF_COLOR,
        )
    else:
        bf_norm = _load_bf_raw(bf_paths, tp)
        _draw_panel(
            axes[0],
            bf_norm,
            labels,
            f"{cell} (this run)\nBF raw + boundary   N = {n_obj}",
            MINE_COLOR,
        )
        h2b_norm = _load_h2b_raw(h2b_paths, tp)
        _draw_panel(
            axes[1],
            h2b_norm,
            labels,
            f"same {cell} boundary on H2B raw\nnuclei vs whole-cell diagnostic   N = {n_obj}",
            MINE_COLOR,
        )

    fig.suptitle(
        f"Stage 02 smoke QC — {REGION} / fov {FOV} — t = {tp:04d}",
        fontsize=12,
    )
    out_path = out_dir / f"{cell}__t{tp:04d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_matrix_grid(
    masks_root: Path,
    h2b_paths: list[str],
    bf_paths: list[str],
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(CELLS), len(TIMEPOINTS), figsize=(13, 17), constrained_layout=True
    )
    for row, cell in enumerate(CELLS):
        is_h2b = cell.endswith("_h2b")
        for col, tp in enumerate(TIMEPOINTS):
            labels = np.load(masks_root / cell / f"t{tp:04d}.npy")
            raw_norm = _load_h2b_raw(h2b_paths, tp) if is_h2b else _load_bf_raw(bf_paths, tp)
            _draw_panel(
                axes[row, col],
                raw_norm,
                labels,
                f"{cell}   t={tp:04d}   N={_object_count(labels)}",
                MINE_COLOR,
            )
    fig.suptitle(
        "Stage 02 smoke matrix — R0 / fov 0 — boundaries on input channel",
        fontsize=13,
    )
    out_path = out_dir / "matrix_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument(
        "--acquisition",
        default="timelapse_2026-05-05_18-12-11.466141",
        help="Acquisition subdir name under <data_root>.",
    )
    p.add_argument(
        "--reference-zarr",
        default=None,
        help="Defaults to <data_root>/cyto_ring_output/data.zarr.",
    )
    p.add_argument("--reference-scene", default="R0_R0")
    args = p.parse_args()

    data_root: Path = args.data_root.resolve()
    acquisition_dir = data_root / args.acquisition
    ref_zarr_path = Path(args.reference_zarr) if args.reference_zarr else data_root / "cyto_ring_output" / "data.zarr"

    masks_root = data_root / "analysis" / "segmentation" / "validation" / f"{REGION}_fov{FOV}"
    out_dir = data_root / "analysis" / "figures" / "stage02_smoke_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] data_root      = {data_root}")
    print(f"[render] acquisition    = {acquisition_dir}")
    print(f"[render] reference zarr = {ref_zarr_path} :: {args.reference_scene}/nuc_labels")
    print(f"[render] masks_root     = {masks_root}")
    print(f"[render] out_dir        = {out_dir}")

    df = parse_dataset(acquisition_dir)
    series = get_fov_timeseries(df, region=REGION, fov=FOV)
    h2b_paths = series["mTagBFP2"]
    bf_paths = series["BF"]
    print(f"[render] tiff counts    : H2B={len(h2b_paths)}, BF={len(bf_paths)}")

    ref_zarr = zarr.open(str(ref_zarr_path), mode="r")

    for cell in CELLS:
        for tp in TIMEPOINTS:
            written = render_cell_tp(
                cell, tp, masks_root, h2b_paths, bf_paths, ref_zarr, args.reference_scene, out_dir
            )
            print(f"  wrote {written.name}")

    grid = render_matrix_grid(masks_root, h2b_paths, bf_paths, out_dir)
    print(f"[render] summary grid   : {grid.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
