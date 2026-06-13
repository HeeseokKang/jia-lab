"""Collect reference-segmentation MISS candidates (false negatives).

Step 1 of the seg-correction HITL loop. Compares the H2B raw against the current
reference StarDist masks and flags bright nucleus-sized blobs the mask does NOT
cover, so the operator reviews the highest-miss frames first instead of scanning
all 567 frames blindly.

This is a *cheap detector to prioritise human review*, not a segmenter: it finds
likely false negatives (missed / dim / isolated / clumped nuclei) and ranks
frames by how many it finds. Every flag is a hypothesis for the human to confirm
or reject in napari — never a label.

Output (under <data_root>/analysis/segmentation/seg_review/<region>_fov<fov>/):
  - miss_candidates.csv   one row per candidate: timepoint, y, x, area, mean_int, reason
  - frame_priority.csv    per-frame miss count, sorted desc (review order)

Run in any env with numpy/skimage/tifffile (e.g. the napari env):
    /opt/miniconda/envs/napari-env/bin/python scripts/collect_seg_misses.py \
        --data-root /data/.../20260505_ERKKTR_H2B_BF_Timelapse [--t0 0 --t1 566 --stride 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.measure import regionprops

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.io import get_fov_timeseries, parse_dataset  # noqa: E402

REGION, FOV = "R0", 0
NEIGHBOUR_R = 60      # px; for the "isolated" reason flag
NUC_FLOOR_PCT = 10    # "nucleus-bright" = >= 10th pct of intensity inside real nuclei


def _nucleus_size_bounds(masks_dir: Path, tps: list[int]) -> tuple[float, float]:
    """Data-driven nucleus area window from a few reference frames."""
    areas = []
    for tp in tps[:: max(1, len(tps) // 10)]:
        m = np.load(masks_dir / f"t{tp:04d}.npy")
        _, counts = np.unique(m[m > 0], return_counts=True)
        areas.extend(counts.tolist())
    if not areas:
        return 30.0, 1500.0
    med = float(np.median(areas))
    return max(15.0, 0.3 * med), 4.0 * med


def _frame_misses(raw: np.ndarray, mask: np.ndarray, amin: float, amax: float):
    """Yield (y, x, area, mean_int, reason) for missed-nucleus candidates.

    Anchored to the reference itself: "nucleus-bright" is calibrated from the
    intensity *inside existing nuclei* (so it adapts per frame and catches dim
    real nuclei without firing on cytoplasmic haze or watershed boundaries).
    A nucleus-bright region the mask does NOT cover, of plausible nucleus size,
    is a false-negative candidate for the human to confirm or reject.
    """
    rf = raw.astype(np.float32)
    covered = mask > 0
    if not covered.any() or float(rf.max() - rf.min()) < 1e-6:
        return
    floor = float(np.percentile(rf[covered], NUC_FLOOR_PCT))  # dim-nucleus brightness
    sm = gaussian(rf, sigma=1.0, preserve_range=True)
    cand = (sm >= floor) & (~covered)
    cand = ndi.binary_opening(cand, iterations=1)  # drop watershed slivers
    lab, n = ndi.label(cand)
    if n == 0:
        return
    cov_lbls = np.unique(mask[covered])
    cov_centroids = np.array(ndi.center_of_mass(covered, mask, cov_lbls))
    for rp in regionprops(lab, intensity_image=rf):
        if rp.area < amin:
            continue
        y, x = rp.centroid
        reason = []
        if rp.area > amax:
            reason.append("clump")
        d = np.hypot(cov_centroids[:, 0] - y, cov_centroids[:, 1] - x)
        if d.min() > NEIGHBOUR_R:
            reason.append("isolated")
        yield (float(y), float(x), int(rp.area), float(rp.intensity_mean),
               "+".join(reason) if reason else "missed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--acquisition", default="timelapse_2026-05-05_18-12-11.466141")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--fov", type=int, default=FOV)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=566)
    ap.add_argument("--stride", type=int, default=1,
                    help="frame stride; use e.g. 10 for a fast first prioritisation pass")
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    tps = list(range(args.t0, args.t1 + 1, args.stride))
    masks_dir = (data_root / "analysis" / "segmentation" / "validation"
                 / f"{args.region}_fov{args.fov}" / "stardist_h2b")
    out_dir = (data_root / "analysis" / "segmentation" / "seg_review"
               / f"{args.region}_fov{args.fov}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_dataset(data_root / args.acquisition)
    h2b = get_fov_timeseries(df, region=args.region, fov=args.fov)["mTagBFP2"]
    amin, amax = _nucleus_size_bounds(masks_dir, list(range(args.t0, args.t1 + 1)))
    print(f"nucleus area window: [{amin:.0f}, {amax:.0f}] px  (clump > {amax:.0f})")

    rows, per_frame = [], []
    dim_thr = None
    for i, tp in enumerate(tps):
        raw = tifffile.imread(h2b[tp])
        mask = np.load(masks_dir / f"t{tp:04d}.npy")
        cands = list(_frame_misses(raw, mask, amin, amax))
        for (y, x, area, mi, reason) in cands:
            rows.append({"timepoint": tp, "centroid_y": y, "centroid_x": x,
                         "area": area, "mean_int": mi, "reason": reason})
        per_frame.append({"timepoint": tp, "n_miss": len(cands)})
        if (i + 1) % 25 == 0 or i == len(tps) - 1:
            print(f"  scanned {i + 1}/{len(tps)} frames  (t={tp}, misses so far={len(rows)})")

    cand_df = pd.DataFrame(rows)
    if not cand_df.empty:
        dim_thr = cand_df["mean_int"].quantile(0.25)
        cand_df.loc[cand_df["mean_int"] <= dim_thr, "reason"] = (
            cand_df["reason"].astype(str) + "+dim")
        cand_df["reason"] = cand_df["reason"].str.replace("missed+dim", "dim", regex=False)
    cand_df.to_csv(out_dir / "miss_candidates.csv", index=False)

    prio = (pd.DataFrame(per_frame).sort_values("n_miss", ascending=False)
            .reset_index(drop=True))
    prio.to_csv(out_dir / "frame_priority.csv", index=False)

    n = len(cand_df)
    top = prio.head(10)
    print(f"\n{n} miss candidates over {len(tps)} frames "
          f"(~{n / max(1, len(tps)):.1f}/frame).")
    print(f"reasons: {cand_df['reason'].value_counts().to_dict() if n else '{}'}")
    print("highest-miss frames (review these first):")
    for _, r in top.iterrows():
        print(f"  t={int(r['timepoint']):4d}  misses={int(r['n_miss'])}")
    print(f"\nwrote:\n  {out_dir/'miss_candidates.csv'}\n  {out_dir/'frame_priority.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
