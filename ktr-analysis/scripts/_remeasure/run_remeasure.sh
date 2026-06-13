#!/usr/bin/env bash
# Closed-loop re-measure with the fine-tuned StarDist model (additive A/B).
# Fair A/B: ONLY the seg model differs from baseline. Tracking params = baseline md25.
# Baseline (stardist_h2b masks, gt_scores.json) is never touched; all outputs are _ft.
set -euo pipefail

DR=/data/Project_Data/Voltage_CellCycle/20260505_ERKKTR_H2B_BF_Timelapse
ANALYSIS=$DR/analysis
REPO=$HOME/github/jia-lab/ktr-analysis
MODEL_BASEDIR=$ANALYSIS/segmentation/models
MODEL_NAME=stardist_ft_20260611
FT_MASKS=$ANALYSIS/segmentation/validation/R0_fov0/stardist_ft_h2b
FT_TRACK_ROOT=$ANALYSIS/tracking/_ft
FT_TRAJ=$FT_TRACK_ROOT/R0_fov0/trajectories.parquet

source /opt/miniconda/etc/profile.d/conda.sh
cd "$REPO"

echo "### Stage 1 — re-seg t0-100 with fine-tuned model (ktr-segtrack)"
conda activate ktr-segtrack
python scripts/_remeasure/stage1_seg_ft.py --data-root "$DR" \
  --model-basedir "$MODEL_BASEDIR" --model-name "$MODEL_NAME" \
  --region R0 --fov 0 --t0 0 --t1 100 --out-dir "$FT_MASKS"
conda deactivate

echo "### Stage 2 — Ultrack on ft masks, baseline params md25 (ultrack_env)"
conda activate ultrack_env
python scripts/track_ultrack_scene.py --data-root "$DR" --region R0 --fov 0 \
  --t0 0 --t1 100 --masks-stage validation --cell stardist_ft_h2b \
  --max-distance 25 --min-area 10 --max-area 100000 \
  --out-root "$FT_TRACK_ROOT"

echo "### Stage 3 — score ft tracks vs lineage GT (gt_scores_ft.json)"
python scripts/_remeasure/stage3_score_ft.py --data-root "$DR" \
  --ft-traj "$FT_TRAJ" --ft-masks "$FT_MASKS" --match-frames 2 --match-px 30

echo "### DONE"
