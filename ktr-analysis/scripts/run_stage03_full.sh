#!/bin/bash
# =============================================================================
# Stage 03 FULL production run: fresh StarDist -> Ultrack, all 32 scenes.
#
#   *** DO NOT RUN until the smoke campaign passes AND there is explicit
#       operator approval. Guarded by RUN_STAGE03_CONFIRM=YES. ***
#
# Mask-source policy (resolved by data 2026-05-21): tracking division recall is
# mask-source dependent (fresh StarDist 0.80 vs Bill nuc_labels 0.30), so this
# run RE-RUNS StarDist as the production substrate (not Bill's nuc_labels).
#
# Two phases on disjoint resources, each restartable + failure-isolated:
#   A. StarDist  (GPU, ktr-segtrack)  -> analysis/segmentation/full/{scene}/stardist_h2b/
#   B. Ultrack   (CPU, ultrack_env)   -> analysis/tracking/{scene}/trajectories.parquet
# Per-scene _SUCCESS markers make both phases idempotent: re-running skips done
# scenes, so a crash resumes from the next incomplete scene.
#
# Usage (detached, dedicated tmux session, monitor via logfile only):
#   tmux new-session -d -s ktr_stage03_full \
#     "RUN_STAGE03_CONFIRM=YES WORKERS=4 bash scripts/run_stage03_full.sh > LOG 2>&1"
# =============================================================================
set -u

REPO=/home/heeseok/github/jia-lab/ktr-analysis
DATA=/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
SEG_PY=/opt/miniconda/envs/ktr-segtrack/bin/python      # phase A needs the activate.d GPU hook (see below)
TRK_PY=/opt/miniconda/envs/ultrack_env/bin/python
WORKERS="${WORKERS:-4}"                                  # Ultrack concurrent FOV processes; SET FROM SMOKE
REGIONS=(R0 R1)
FOVS=$(seq 0 15)
cd "$REPO" || exit 2

if [[ "${RUN_STAGE03_CONFIRM:-NO}" != "YES" ]]; then
  echo "REFUSING TO RUN: set RUN_STAGE03_CONFIRM=YES to launch the full 32-scene run."
  echo "This is the guarded production driver; run the smoke campaign first."
  exit 3
fi

ts() { date '+%F %T %Z'; }
echo "[$(ts)] STAGE 03 FULL START | workers=$WORKERS | $(( ${#REGIONS[@]} * $(echo "$FOVS" | wc -l) )) scenes"

# ---- Phase A: StarDist (GPU, serial) ---------------------------------------
echo "[$(ts)] PHASE A: fresh StarDist (GPU, serial over scenes)"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate ktr-segtrack            # triggers cu12 activate.d LD_LIBRARY_PATH hook
export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1
seg_fail=0
for R in "${REGIONS[@]}"; do
  for F in $FOVS; do
    echo "[$(ts)]   segA $R/fov$F"
    if ! python scripts/seg_stardist_scene.py --region "$R" --fov "$F" --stage full; then
      echo "[$(ts)]   segA FAILED $R/fov$F (continuing)"; seg_fail=$((seg_fail+1))
    fi
  done
done
conda deactivate
echo "[$(ts)] PHASE A done | seg failures=$seg_fail"

# ---- Phase B: Ultrack (CPU, parallel worker pool) --------------------------
echo "[$(ts)] PHASE B: Ultrack (CPU, $WORKERS-way parallel)"
# emit "region fov" lines, hand to xargs -P. Each process is single-Ultrack-worker
# (n-workers 1) to avoid oversubscribing 24 cores with WORKERS processes.
scene_list=$(for R in "${REGIONS[@]}"; do for F in $FOVS; do echo "$R $F"; done; done)
echo "$scene_list" | xargs -P "$WORKERS" -n2 bash -c '
  R="$0"; F="$1"
  echo "[trkB $(date +%H:%M:%S)] start '"$R"'/fov$F" >&2
  '"$TRK_PY"' '"$REPO"'/scripts/track_ultrack_scene.py \
      --data-root '"$DATA"' --region "$R" --fov "$F" --masks-stage auto --n-workers 1 \
    || echo "[trkB $(date +%H:%M:%S)] FAILED $R/fov$F (continuing)" >&2
'
echo "[$(ts)] PHASE B done"

# ---- Aggregate -------------------------------------------------------------
echo "[$(ts)] scenes with tracking _SUCCESS:"
find "$DATA/analysis/tracking" -maxdepth 2 -name _SUCCESS | wc -l
echo "[$(ts)] STAGE 03 FULL COMPLETE"
