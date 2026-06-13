#!/usr/bin/env bash
# Unattended review-package build for Stage-03 tracking (R0/fov0, t0-100).
#
# Produces, into  <dataset>/analysis/tracking/_review_<DATE>/ :
#   - bf_ultrack_t0-100.mp4       BF + track-coloured outlines (label-free identity)
#   - h2b_reference_t0-100.mp4    H2B-reference tracks on the SAME BF view (upper bound)
#   - divisions_bf.csv / divisions_h2b.csv   division-birth worklist (tick in napari)
#   - README.txt                  what to look at + the exact napari scoring command
#
# Pure python+ffmpeg, no Claude in the loop -> zero permission prompts. Safe to
# launch in tmux and walk away. Idempotent: existing mp4s are skipped on re-run.
# Everything is logged to _review_<DATE>/run_review.log
set -u

PY=/opt/miniconda/envs/ultrack_env/bin/python
REPO=/home/heeseok/github/jia-lab/ktr-analysis
ROOT=/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
DATE=$(date +%Y%m%d)
REVIEW="$ROOT/analysis/tracking/_review_${DATE}"
T0=0; T1=100; FPS=8

BF_TRAJ="$ROOT/analysis/tracking/_bench_bf/bf_ultrack/R0_fov0/trajectories.parquet"
BF_MASKS="$ROOT/analysis/segmentation/validation/R0_fov0/cellpose_bf"
H2B_TRAJ="$ROOT/analysis/tracking/_bench_maxdist/md12_t0-100/R0_fov0/trajectories.parquet"
H2B_MASKS="$ROOT/analysis/segmentation/validation/R0_fov0/stardist_h2b"

mkdir -p "$REVIEW"
LOG="$REVIEW/run_review.log"
exec > >(tee -a "$LOG") 2>&1
echo "=========================================================="
echo "[$(date '+%F %T')] run_review start -> $REVIEW"
cd "$REPO" || { echo "FATAL: repo not found"; exit 1; }

render () {  # name traj masks outfile
  local name="$1" traj="$2" masks="$3" out="$4"
  if [ -s "$out" ]; then echo "[$(date '+%T')] SKIP $name (exists)"; return 0; fi
  echo "[$(date '+%T')] RENDER $name -> $out"
  "$PY" scripts/render_track_videos.py \
    --data-root "$ROOT" --traj "$traj" --masks "$masks" \
    --channel BF --t0 "$T0" --t1 "$T1" --fps "$FPS" --out "$out"
  echo "[$(date '+%T')] $name exit=$?"
}

render bf_ultrack  "$BF_TRAJ"  "$BF_MASKS"  "$REVIEW/bf_ultrack_t0-100.mp4"
render h2b_ref      "$H2B_TRAJ" "$H2B_MASKS" "$REVIEW/h2b_reference_t0-100.mp4"

# --- division-birth worklists (non-fatal if it fails) ---
echo "[$(date '+%T')] divisions worklist"
"$PY" - "$BF_TRAJ" "$REVIEW/divisions_bf.csv" "$H2B_TRAJ" "$REVIEW/divisions_h2b.csv" <<'PY' || echo "WARN: worklist step failed"
import sys, pandas as pd
for traj, out in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    df = pd.read_parquet(traj)
    d = df[df["parent_track_id"].notna()].sort_values(["timepoint", "track_id"])
    rows = (d.groupby("track_id")
              .first().reset_index()
              .loc[:, ["timepoint", "track_id", "parent_track_id",
                       "centroid_x", "centroid_y"]])
    rows = rows.rename(columns={"timepoint": "birth_frame"})
    rows["verdict_real_division"] = ""   # human fills: y / n
    rows["notes"] = ""
    rows.to_csv(out, index=False)
    print(f"  {out}: {len(rows)} daughter births")
PY

# --- README ---
cat > "$REVIEW/README.txt" <<EOF
Stage-03 tracking review package  ($(date '+%F %T'))
R0 / fov0 / t${T0}-${T1}

WHAT THE VIDEOS SHOW
  Outline colour == track_id, held constant for the whole movie.
    * same cell keeps its colour  -> identity preserved (good)
    * a cell's colour suddenly changes / two cells swap colours -> ID SWAP
    * a cell splits into two new colours + white ring -> division birth
  bf_ultrack_t0-100.mp4    label-free identity (cellpose_bf + ultrack) -- the thing under test
  h2b_reference_t0-100.mp4 fluorescent-nucleus identity (stardist_h2b + ultrack) -- the ceiling
  Same BF background in both, so you can compare the two identity sources directly.

WHAT TO EYEBALL (note timepoints + rough x,y as you watch)
  1. Where does BF identity break but H2B holds? (colour flips / swaps)
  2. Divisions: does the white ring sit on a real mother->2-daughter split?
     Scan +-5 frames for real divisions with NO ring (misses).
  3. Cells that fragment (colour keeps changing on a clearly-continuous cell).

THEN SCORE IN NAPARI (needs the desktop display, not headless)
  /opt/miniconda/envs/napari-env/bin/python \\
      $REPO/scripts/launch_gt_napari.py --data-root $ROOT
  Protocol: $ROOT/analysis/tracking/gt_lineage_prep/README.md
  Worklists to tick: divisions_bf.csv, divisions_h2b.csv (fill verdict_real_division y/n)

NOTE: cellpose_bf masks here were the provisional cpsam-default run
(segmentation/meta.json: bf_prior_provisional=true). Read the BF result as a
floor, not the final label-free number.
EOF

echo "[$(date '+%F %T')] run_review DONE"
ls -la "$REVIEW"
