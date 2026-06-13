"""Re-measure Stage 1 — re-segment with the FINE-TUNED StarDist model.

Additive A/B: identical to the baseline seg (same mTagBFP2/H2B channel, same
`_normalize_for_stardist` 1-99.8 from stardist_validate) EXCEPT the model weights
are the fine-tuned model loaded from its basedir. Writes to a NEW parallel dir
`segmentation/validation/{region}_fov{fov}/stardist_ft_h2b/` so the baseline
`stardist_h2b` masks are never touched.

Env: ktr-segtrack (TF+StarDist GPU). MUST `conda activate ktr-segtrack`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.io import parse_dataset, get_fov_timeseries  # noqa: E402
from src.segmentation.stardist_validate import _normalize_for_stardist  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--acq", default="timelapse_2026-05-05_18-12-11.466141",
                    help="acquisition subdir under data-root")
    ap.add_argument("--model-basedir", required=True, type=Path)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--region", default="R0")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--channel", default="mTagBFP2")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=100)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    acq_dir = args.data_root.resolve() / args.acq
    df = parse_dataset(acq_dir)
    series = get_fov_timeseries(df, region=args.region, fov=args.fov)
    paths = series[args.channel]

    from stardist.models import StarDist2D
    model = StarDist2D(None, name=args.model_name, basedir=str(args.model_basedir))
    print(f"[seg-ft] model {args.model_basedir}/{args.model_name} "
          f"thresholds={model.thresholds}", flush=True)

    t_start = time.time()
    counts = []
    for tp in range(args.t0, args.t1 + 1):
        out = args.out_dir / f"t{tp:04d}.npy"
        img = tifffile.imread(paths[tp])
        norm = _normalize_for_stardist(img)
        labels, _ = model.predict_instances(norm, verbose=False)
        np.save(out, labels.astype(np.int32))
        n = int(len(np.unique(labels)) - 1)
        counts.append(n)
        if tp % 20 == 0 or tp == args.t1:
            print(f"[seg-ft] t{tp:04d}: {n} nuclei", flush=True)
    print(f"[seg-ft] done {len(counts)} frames, mean {np.mean(counts):.1f} nuclei/frame, "
          f"{time.time()-t_start:.1f}s -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
