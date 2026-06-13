"""Launch napari pre-loaded for tiny GT-lineage annotation (Stage 03).

Loads H2B raw + stardist_h2b masks for R0/fov0 t0-100 and seeds two Points
layers (24 continuity candidates at t0, 10 division seeds at their division
frames) so the annotator can build the GT lineage. This script only SETS UP the
viewer; the annotation itself is manual.

Run in the napari env (do not run headless / in tmux without a display):
    /opt/miniconda/envs/napari-env/bin/python scripts/launch_gt_napari.py \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse [--t0 0 --t1 100]

Annotation protocol: see analysis/tracking/gt_lineage_prep/README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.io import get_fov_timeseries, parse_dataset  # noqa: E402
from src.tracking.gt_annotator import attach_gt_annotator  # noqa: E402

REGION, FOV = "R0", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--acquisition", default="timelapse_2026-05-05_18-12-11.466141")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=100)
    ap.add_argument(
        "--out-csv", type=Path, default=None,
        help="Where to autosave the GT CSV. Default: "
             "<data_root>/analysis/tracking/gt_lineage_prep/gt_lineage_filled.csv. "
             "Pass a separate path (e.g. a demo file) to leave the real GT untouched.",
    )
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    tps = list(range(args.t0, args.t1 + 1))
    masks_dir = data_root / "analysis" / "segmentation" / "validation" / f"{REGION}_fov{FOV}" / "stardist_h2b"
    gp = data_root / "analysis" / "tracking" / "gt_lineage_prep"
    out_csv = args.out_csv if args.out_csv is not None else gp / "gt_lineage_filled.csv"

    try:
        import napari
    except Exception:
        print("napari not importable in this interpreter. Use the napari env:")
        print("  /opt/miniconda/envs/napari-env/bin/python scripts/launch_gt_napari.py --data-root <root>")
        return 1

    df = parse_dataset(data_root / args.acquisition)
    h2b_paths = get_fov_timeseries(df, region=REGION, fov=FOV)["mTagBFP2"]
    raw = np.stack([tifffile.imread(h2b_paths[tp]) for tp in tps])
    masks = np.stack([np.load(masks_dir / f"t{tp:04d}.npy") for tp in tps]).astype(np.int32)

    cand = pd.read_csv(gp / "gt_candidates.csv")
    cand_pts = np.column_stack([np.zeros(len(cand)), cand["centroid_y"], cand["centroid_x"]])  # (t,y,x) at t0
    div = pd.read_csv(gp / "gt_division_candidates.csv")
    div_pts = np.column_stack([div["division_frame"] - args.t0, div["centroid_y"], div["centroid_x"]])

    v = napari.Viewer()
    v.add_image(raw, name="H2B raw", colormap="gray", contrast_limits=[float(raw.min()), float(np.percentile(raw, 99.5))])
    masks_layer = v.add_labels(masks, name="stardist_h2b")
    v.add_points(cand_pts, name="continuity seeds (24)", size=18, face_color="transparent",
                 border_color="#ffcc00", properties={"gt_cell_id": cand["gt_cell_id"].to_numpy()}, text="gt_cell_id")
    v.add_points(div_pts, name="division seeds (10)", size=22, face_color="transparent",
                 border_color="#ff44aa", properties={"div_seed_id": div["div_seed_id"].to_numpy()}, text="div_seed_id")

    # Click-driven annotation: click a nucleus to record (gt_cell_id, timepoint,
    # mask_label, centroid) for the current frame; d/x/l set events; autosaves to
    # gt_lineage_filled.csv. See src/tracking/gt_annotator.py.
    attach_gt_annotator(
        v, masks_layer=masks_layer, out_csv=out_csv,
        t_offset=args.t0, start_id=1, region=REGION, fov=FOV,
    )
    print(f"napari ready. Autosaving GT to: {out_csv}")
    print("Click cells to annotate (panel on the right / README.md).")
    print("Hotkeys: d=division  x=death  l=leaves_fov  n=clear  u=undo  s=save  [ ]=prev/next id  c=new id")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
