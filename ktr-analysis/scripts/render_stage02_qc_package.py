"""Stage 02 visual QC package (06a / 06b / 06c).

A human-review package built on the COMPLETED full-sweep masks
(``<data_root>/analysis/segmentation/validation/R0_fov0/<cell>/``) + raw TIFFs
+ Bill's reference zarr. Reuses the rendering utilities from
``render_stage02_overlays.py`` (normalisation, boundary-RGBA, object count,
colors) rather than re-implementing them.

Three output sub-packages (proposed 2026-05-20):

  06a_h2b_individual/   H2B raw grayscale  |  H2B (StarDist) contours only
                        -> nucleus-segmentation QC
  06b_bf_individual/    BF raw grayscale   |  BF whole-cell (cpsam) contours only
                        -> BF whole-cell segmentation QC
  06c_cross_overlay/    BF raw + cpsam whole-cell + DAPI nuclei (fill) + Bill
                        whole-cell contours -> containment / agreement inspection

Frames are a curated, review-efficient subset: representative (density sweep +
best case) and failure-focused (the cellpose_bf low-recall clusters found in the
full-sweep analysis). Not the full 567 frames. No GPU, no tracking.

CLI:
    python scripts/render_stage02_qc_package.py \
        --data-root /data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Reuse the existing Stage 02 overlay utilities (don't re-implement).
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from render_stage02_overlays import (  # noqa: E402
    _boundary_rgba,
    _normalize_for_display,
    _object_count,
)
from src.io import get_fov_timeseries, parse_dataset  # noqa: E402
from src.segmentation.metrics import dapi_coverage  # noqa: E402

REGION = "R0"
FOV = 0
H2B_CELL = "stardist_h2b"   # locked nuclear GT segmenter
BF_CELL = "cellpose_bf"     # locked BF whole-cell prior (cpsam)

# Layer colors (consistent with render_stage02_overlays MINE/REF).
BF_COLOR = "#ffcc00"        # cpsam BF whole-cell contour
BILL_CELL_COLOR = "#00ccff"  # Bill cell_labels whole-cell contour
NUC_FILL_COLOR = "#ff44aa"  # DAPI nuclei (Bill nuc_labels), filled
H2B_SEG_COLOR = "#ffcc00"   # StarDist H2B contour

# Curated frames (from the full-sweep analysis 20260520).
REPRESENTATIVE = {
    0: "rep · low-density start (~223 cells)",
    200: "rep · peak BF recall (0.946)",
    280: "rep · mid-density (~495 cells)",
    566: "rep · high-density (~1040 cells)",
}
FAILURE = {
    20: "fail · early low-recall dip",
    40: "fail · early low-recall dip",
    140: "fail · mid recall dip (~0.87)",
    540: "fail · late high-density dip",
    560: "fail · late high-density dip",
}


def _fill_rgba(labels: np.ndarray, color: str, alpha: float = 0.40) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    r, g, b, _ = to_rgba(color)
    out[labels > 0] = (r, g, b, alpha)
    return out


def _blank_ax(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def render_06a(tp, note, h2b_norm, h2b_labels, out_dir) -> Path:
    n = _object_count(h2b_labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    axes[0].imshow(h2b_norm, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"H2B raw (mTagBFP2)   t={tp:04d}", fontsize=11)
    axes[1].imshow(h2b_norm, cmap="gray", interpolation="nearest")
    axes[1].imshow(_boundary_rgba(h2b_labels, H2B_SEG_COLOR), interpolation="nearest")
    axes[1].set_title(f"StarDist nuclei contours only   N={n}", fontsize=11)
    for a in axes:
        _blank_ax(a)
    fig.suptitle(f"06a H2B nucleus-seg QC — {REGION}/fov{FOV} — {note}", fontsize=12)
    p = out_dir / f"t{tp:04d}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def render_06b(tp, note, bf_norm, bf_labels, out_dir) -> Path:
    n = _object_count(bf_labels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    axes[0].imshow(bf_norm, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"BF raw   t={tp:04d}", fontsize=11)
    axes[1].imshow(bf_norm, cmap="gray", interpolation="nearest")
    axes[1].imshow(_boundary_rgba(bf_labels, BF_COLOR), interpolation="nearest")
    axes[1].set_title(f"cpsam whole-cell contours only   N={n}", fontsize=11)
    for a in axes:
        _blank_ax(a)
    fig.suptitle(f"06b BF whole-cell QC — {REGION}/fov{FOV} — {note}", fontsize=12)
    p = out_dir / f"t{tp:04d}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def render_06c(tp, note, bf_norm, bf_labels, nuc_labels, bill_cell_labels, recall, out_dir) -> Path:
    n_bf = _object_count(bf_labels)
    n_nuc = _object_count(nuc_labels)
    n_bill = _object_count(bill_cell_labels)
    fig, ax = plt.subplots(figsize=(11, 11), constrained_layout=True)
    ax.imshow(bf_norm, cmap="gray", interpolation="nearest")
    ax.imshow(_fill_rgba(nuc_labels, NUC_FILL_COLOR, alpha=0.40), interpolation="nearest")
    ax.imshow(_boundary_rgba(bf_labels, BF_COLOR), interpolation="nearest")
    ax.imshow(_boundary_rgba(bill_cell_labels, BILL_CELL_COLOR), interpolation="nearest")
    _blank_ax(ax)
    legend = [
        Patch(facecolor=NUC_FILL_COLOR, alpha=0.4, label=f"DAPI nuclei (Bill nuc_labels)  N={n_nuc}"),
        Line2D([0], [0], color=BF_COLOR, lw=3, label=f"BF whole-cell (cpsam)  N={n_bf}"),
        Line2D([0], [0], color=BILL_CELL_COLOR, lw=3, label=f"Bill whole-cell (cell_labels)  N={n_bill}"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)
    ax.set_title(
        f"06c containment/agreement — {REGION}/fov{FOV} — t={tp:04d} — {note}\n"
        f"DAPI recall by cpsam={recall:.3f}  (nuclei inside a yellow contour = recovered)",
        fontsize=11,
    )
    p = out_dir / f"t{tp:04d}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--acquisition", default="timelapse_2026-05-05_18-12-11.466141")
    ap.add_argument("--reference-zarr", default=None)
    ap.add_argument("--reference-scene", default="R0_R0")
    args = ap.parse_args()

    data_root: Path = args.data_root.resolve()
    acquisition_dir = data_root / args.acquisition
    ref_zarr_path = Path(args.reference_zarr) if args.reference_zarr else data_root / "cyto_ring_output" / "data.zarr"
    masks_root = data_root / "analysis" / "segmentation" / "validation" / f"{REGION}_fov{FOV}"

    pkg_root = data_root / "analysis" / "figures" / "stage02_qc_package"
    dirs = {k: pkg_root / k for k in ("06a_h2b_individual", "06b_bf_individual", "06c_cross_overlay")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    df = parse_dataset(acquisition_dir)
    series = get_fov_timeseries(df, region=REGION, fov=FOV)
    h2b_paths, bf_paths = series["mTagBFP2"], series["BF"]
    ref = zarr.open(str(ref_zarr_path), mode="r")[args.reference_scene]

    frames = {**REPRESENTATIVE, **FAILURE}
    manifest_rows = []
    for tp in sorted(frames):
        note = frames[tp]
        h2b_norm = _normalize_for_display(tifffile.imread(h2b_paths[tp]))
        bf_norm = _normalize_for_display(tifffile.imread(bf_paths[tp]))
        h2b_labels = np.load(masks_root / H2B_CELL / f"t{tp:04d}.npy")
        bf_labels = np.load(masks_root / BF_CELL / f"t{tp:04d}.npy")
        nuc_labels = np.asarray(ref["nuc_labels"][tp])
        bill_cell_labels = np.asarray(ref["cell_labels"][tp])
        recall = dapi_coverage(nuc_labels, bf_labels)["recall"]

        render_06a(tp, note, h2b_norm, h2b_labels, dirs["06a_h2b_individual"])
        render_06b(tp, note, bf_norm, bf_labels, dirs["06b_bf_individual"])
        render_06c(tp, note, bf_norm, bf_labels, nuc_labels, bill_cell_labels, recall, dirs["06c_cross_overlay"])
        manifest_rows.append({
            "timepoint": tp,
            "category": "representative" if tp in REPRESENTATIVE else "failure",
            "note": note,
            "n_stardist_h2b": _object_count(h2b_labels),
            "n_cellpose_bf": _object_count(bf_labels),
            "n_bill_nuc": _object_count(nuc_labels),
            "n_bill_cell": _object_count(bill_cell_labels),
            "cpsam_dapi_recall": round(float(recall), 4),
        })
        print(f"  rendered t{tp:04d}  ({note})  recall={recall:.3f}")

    man_path = pkg_root / "manifest.csv"
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\n[qc-package] {len(frames)} frames x 3 views -> {pkg_root}")
    print(f"[qc-package] manifest: {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
