"""Launch napari for HITL segmentation-correction (false-negative recovery).

Loads H2B raw + the reference StarDist masks for a frame window, **resumes** any
frames already committed (reloads corrected masks over the originals), overlays
the miss candidates from ``collect_seg_misses.py`` as review prompts, and docks
the seg-correction widget. Setup only — the correction is manual.

Run in the napari env (needs a display; do NOT run headless / in tmux):
    /opt/miniconda/envs/napari-env/bin/python scripts/launch_seg_review.py \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse [--t0 0 --t1 100]

Workflow: see the docked "Seg correction" panel.
  click a missed nucleus -> recovered.   f = add at cursor.   u = undo add.
  c = commit frame (export image+mask pair, mark done).   s = checkpoint frame.
  split/merge: pick up the Labels brush (keys 2/3/4); Ctrl+Z undoes brush edits.
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
from src.segmentation.seg_correction import attach_seg_corrector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--acquisition", default="timelapse_2026-05-05_18-12-11.466141")
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=100)
    ap.add_argument("--frames", type=str, default=None,
                    help="explicit comma-separated timepoints (e.g. a stratified "
                         "sample: 0,30,60,...). Overrides --t0/--t1; loads only these "
                         "frames as a (non-contiguous) stack.")
    ap.add_argument("--frames-file", type=Path, default=None,
                    help="text/CSV file with one timepoint per line (header 'timepoint' "
                         "or a bare int per line). Overrides --t0/--t1.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="seg_corrections root. Default: "
                         "<data_root>/analysis/segmentation/seg_corrections/<region>_fov<fov>.")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    # frame selection: explicit list (non-contiguous) wins over the --t0/--t1 range
    explicit = None
    if args.frames:
        explicit = [int(x) for x in args.frames.replace(" ", "").split(",") if x != ""]
    elif args.frames_file:
        lines = Path(args.frames_file).read_text().splitlines()
        explicit = [int(x.split(",")[0]) for x in lines
                    if x.strip() and not x.lower().startswith("timepoint")]
    if explicit is not None:
        tps = sorted(set(explicit))
    else:
        tps = list(range(args.t0, args.t1 + 1))
    tp_to_idx = {tp: i for i, tp in enumerate(tps)}
    base = data_root / "analysis" / "segmentation"
    masks_dir = base / "validation" / f"{args.region}_fov{args.fov}" / "stardist_h2b"
    review_dir = base / "seg_review" / f"{args.region}_fov{args.fov}"
    out_dir = args.out_dir or (base / "seg_corrections" / f"{args.region}_fov{args.fov}")
    corrected_mask_dir = out_dir / "masks"

    try:
        import napari
    except Exception:
        print("napari not importable here. Use the napari env:")
        print("  /opt/miniconda/envs/napari-env/bin/python scripts/launch_seg_review.py --data-root <root>")
        return 1

    df = parse_dataset(data_root / args.acquisition)
    h2b = get_fov_timeseries(df, region=args.region, fov=args.fov)["mTagBFP2"]
    raw = np.stack([tifffile.imread(h2b[tp]) for tp in tps])

    # resume: prefer a committed corrected mask over the original reference
    masks, resumed = [], 0
    for tp in tps:
        corr = corrected_mask_dir / f"t{tp:04d}.tif"
        if corr.exists():
            masks.append(tifffile.imread(corr).astype(np.int32)); resumed += 1
        else:
            masks.append(np.load(masks_dir / f"t{tp:04d}.npy").astype(np.int32))
    masks = np.stack(masks)

    v = napari.Viewer()
    v.add_image(raw, name="H2B raw", colormap="gray",
                contrast_limits=[float(raw.min()), float(np.percentile(raw, 99.5))])
    masks_layer = v.add_labels(masks, name="reference masks (editable)")

    # miss prompts (if the collector has been run)
    mc = review_dir / "miss_candidates.csv"
    if mc.exists():
        cand = pd.read_csv(mc)
        cand = cand[cand["timepoint"].isin(tps)]
        if len(cand):
            z = cand["timepoint"].map(tp_to_idx).to_numpy()
            pts = np.column_stack([z, cand["centroid_y"], cand["centroid_x"]])
            v.add_points(pts, name="miss prompts", size=22, face_color="transparent",
                         border_color="#ff3333", symbol="ring")
            print(f"{len(cand)} miss prompts loaded from collect_seg_misses.py")
    else:
        print(f"(no miss_candidates.csv yet — run collect_seg_misses.py first for review prompts)")

    attach_seg_corrector(
        v, image_layer=v.layers["H2B raw"], masks_layer=masks_layer,
        out_dir=out_dir, t_offset=args.t0, region=args.region, fov=args.fov,
        timepoints=(tps if explicit is not None else None),
    )
    span = (f"frames {tps[0]}..{tps[-1]} (n={len(tps)}, non-contiguous)"
            if explicit is not None else f"window t{args.t0}-{args.t1}")
    print(f"napari ready. {span}; resumed {resumed} committed frame(s).")
    print(f"corrections -> {out_dir}/{{images,masks}}/  + manifest.csv")
    print("Click missed nuclei (red rings = prompts). c=commit  f=add@cursor  u=undo  s=checkpoint.")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
