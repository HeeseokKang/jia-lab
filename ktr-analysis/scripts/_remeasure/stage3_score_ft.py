"""Re-measure Stage 3 — score the FINE-TUNED tracks against the lineage GT.

Reuses the scoring functions in scripts/score_gt.py. Two metrics:
  - DIVISION recall/precision: centroid-based (match_divisions) -> mask-independent,
    valid on the ft tracks as-is.
  - ID-SWAP: score_gt joins GT.mask_label to tracker.cell_in_frame_id. GT's mask_label
    was assigned from BASELINE masks, so for the ft variant we RE-DERIVE each GT cell's
    label from the ft masks (ft_mask[tp][round(cy), round(cx)]) before the join — i.e.
    the mapping is on the ft masks, not baseline. (The #1 correctness risk in the spec.)

Writes gt_scores_ft.json (NOT gt_scores.json) + a baseline-vs-ft comparison.
Env: any with pandas/numpy (no GPU).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def _load_score_gt():
    spec = importlib.util.spec_from_file_location("score_gt", REPO / "scripts" / "score_gt.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--ft-traj", required=True, type=Path, help="ft trajectories.parquet")
    ap.add_argument("--ft-masks", required=True, type=Path, help="ft stardist_ft_h2b dir (t####.npy)")
    ap.add_argument("--match-frames", type=int, default=2)
    ap.add_argument("--match-px", type=float, default=30.0)
    args = ap.parse_args()

    sg = _load_score_gt()
    base = args.data_root.resolve() / "analysis" / "tracking"
    gt = pd.read_csv(base / "gt_lineage_prep" / "gt_lineage_filled.csv")

    traj = pd.read_parquet(args.ft_traj)
    # score_gt functions expect a 'timepoint' column (the ft schema already has it).

    # --- division: centroid-based, valid as-is ---
    division = sg.match_divisions(sg.gt_divisions(gt), sg.tracker_divisions(traj),
                                  args.match_frames, args.match_px)

    # --- id-swap: re-derive GT mask_label on the ft masks ---
    mask_cache: dict[int, np.ndarray] = {}

    def ft_label(tp: int, cy: float, cx: float) -> int:
        if tp not in mask_cache:
            mask_cache[tp] = np.load(args.ft_masks / f"t{tp:04d}.npy")
        m = mask_cache[tp]
        yy = min(int(round(cy)), m.shape[0] - 1)
        xx = min(int(round(cx)), m.shape[1] - 1)
        return int(m[yy, xx])

    gt_ft = gt.copy()
    gt_ft["mask_label"] = [ft_label(int(r.timepoint), float(r.centroid_y), float(r.centroid_x))
                           for r in gt.itertuples()]
    id_swap = sg.id_swap_rate(gt_ft, traj)

    results = {"ultrack_ft": {"id_swap": id_swap, "division": division}}
    out = base / "gt_scores_ft.json"
    out.write_text(json.dumps(
        {"params": vars(args) | {"data_root": str(args.data_root)}, "results": results},
        indent=2, default=str))

    # --- baseline vs ft comparison ---
    baseline = json.loads((base / "gt_scores.json").read_text())["results"]["ultrack"]
    b_div, f_div = baseline["division"], division
    b_sw, f_sw = baseline["id_swap"], id_swap
    print("\n================ baseline (md25) vs fine-tuned ================")
    print(f"{'metric':28s} {'baseline':>12s} {'fine-tuned':>12s}")
    rows = [
        ("division recall", b_div["recall"], f_div["recall"]),
        ("division precision", b_div["precision"], f_div["precision"]),
        ("gt_divisions", b_div["gt_divisions"], f_div["gt_divisions"]),
        ("tracker_divisions", b_div["tracker_divisions"], f_div["tracker_divisions"]),
        ("matched divisions", b_div["matched"], f_div["matched"]),
        ("id_swap_rate", b_sw["id_swap_rate"], f_sw["id_swap_rate"]),
        ("id_swaps", b_sw["id_swaps"], f_sw["id_swaps"]),
        ("links_evaluated", b_sw["links_evaluated"], f_sw["links_evaluated"]),
        ("unmapped_links", b_sw["unmapped_links"], f_sw["unmapped_links"]),
    ]
    for name, b, f in rows:
        print(f"{name:28s} {str(b):>12s} {str(f):>12s}")
    print(f"\n[score-ft] wrote {out}")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
