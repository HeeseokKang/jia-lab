#!/usr/bin/env bash
# Unattended *full-movie* automation (R0/fov0, t0-566). Run AFTER run_review.sh,
# in tmux. Pure python+ffmpeg, no Claude in the loop -> zero permission prompts.
#
# Stages (safe first, heavy last; each idempotent + logged; a heavy-stage failure
# does NOT lose the safe outputs):
#   1. render full-movie H2B-reference video  (track already exists: tracking/R0_fov0)
#   2. run full-movie BF tracking              (cellpose_bf, t0-566, md=12) -- NEW heavy compute
#   3. render full-movie BF video              (from stage 2)
# Outputs land in  <dataset>/analysis/tracking/_review_<DATE>/  alongside run_review.sh's.
#
# NOTE: stage-2 BF masks are the provisional cpsam-default run -> read full-movie
# BF tracking as a long-range fragmentation/identity DIAGNOSTIC, not a final number.
set -u

PY=/opt/miniconda/envs/ultrack_env/bin/python
REPO=/home/heeseok/github/jia-lab/ktr-analysis
ROOT=/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
DATE=$(date +%Y%m%d)
REVIEW="$ROOT/analysis/tracking/_review_${DATE}"
ANALYSIS="$ROOT/analysis"
T0=0; T1=566; FPS=12

H2B_TRAJ="$ANALYSIS/tracking/R0_fov0/trajectories.parquet"          # existing prod full-movie
H2B_MASKS="$ANALYSIS/segmentation/validation/R0_fov0/stardist_h2b"
BF_MASKS="$ANALYSIS/segmentation/validation/R0_fov0/cellpose_bf"
BF_FULL_OUT="$ANALYSIS/tracking/_bench_bf/bf_ultrack_full"          # NEW, additive
BF_FULL_TRAJ="$BF_FULL_OUT/R0_fov0/trajectories.parquet"

mkdir -p "$REVIEW"
LOG="$REVIEW/run_auto_full.log"
exec > >(tee -a "$LOG") 2>&1
echo "=========================================================="
echo "[$(date '+%F %T')] run_auto_full start"
cd "$REPO" || { echo "FATAL: repo not found"; exit 1; }

render () {  # name traj masks outfile
  local name="$1" traj="$2" masks="$3" out="$4"
  if [ -s "$out" ]; then echo "[$(date '+%T')] SKIP render $name (exists)"; return 0; fi
  [ -s "$traj" ] || { echo "[$(date '+%T')] SKIP render $name (no traj: $traj)"; return 0; }
  echo "[$(date '+%T')] RENDER $name -> $out"
  "$PY" scripts/render_track_videos.py \
    --data-root "$ROOT" --traj "$traj" --masks "$masks" \
    --channel BF --t0 "$T0" --t1 "$T1" --fps "$FPS" --out "$out"
  echo "[$(date '+%T')] render $name exit=$?"
}

# --- Stage 1: SAFE -- full-movie H2B-reference video (track already exists) ---
render h2b_full "$H2B_TRAJ" "$H2B_MASKS" "$REVIEW/h2b_reference_FULL_t0-566.mp4"

# --- Stage 2: HEAVY -- full-movie BF tracking (NEW). ~30-90 min; may be RAM-heavy ---
if [ -s "$BF_FULL_TRAJ" ]; then
  echo "[$(date '+%T')] SKIP bf full-track (exists)"
else
  echo "[$(date '+%T')] TRACK bf full-movie t${T0}-${T1} (md=12, min_area=50)"
  "$PY" scripts/track_ultrack_scene.py \
    --data-root "$ROOT" --region R0 --fov 0 --t0 "$T0" --t1 "$T1" \
    --cell cellpose_bf --masks-stage validation \
    --max-distance 12 --min-area 50 --out-root "$BF_FULL_OUT" --force
  echo "[$(date '+%T')] bf full-track exit=$?"
fi

# --- Stage 3: render full-movie BF video (depends on stage 2) ---
render bf_ultrack_full "$BF_FULL_TRAJ" "$BF_MASKS" "$REVIEW/bf_ultrack_FULL_t0-566.mp4"

echo "[$(date '+%F %T')] run_auto_full DONE"
ls -la "$REVIEW"
