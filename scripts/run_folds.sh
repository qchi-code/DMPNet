#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODE="${1:?Usage: run_folds.sh train|test CHAOST2|SABS|CMR 1|2}"
DATASET="${2:?Missing dataset}"
SETTING="${3:?Missing setting}"
if [[ "$DATASET" == CMR && "$SETTING" != 1 ]]; then
  echo 'CMR is evaluated only under Setting 1' >&2; exit 1
fi
if [[ "$DATASET" == SABS ]]; then
  LABELS='[1,2,3,6]'
  N_SV=5000
elif [[ "$DATASET" == CMR ]]; then
  LABELS='[1,2,3]'
  N_SV=1000
else
  LABELS='[1,2,3,4]'
  N_SV=5000
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
LOG_ROOT="${LOG_ROOT:-$ROOT/runs}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-$ROOT/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/trained}"
FOLD_PROTOCOL="${FOLD_PROTOCOL:-legacy}"
NUM_WORKERS="${NUM_WORKERS:-2}"
if [[ "$DATASET" == CHAOST2 && "$SETTING" == 1 ]]; then
  DEFAULT_SEED=2025
elif [[ "$DATASET" == CHAOST2 ]]; then
  DEFAULT_SEED=2024
else
  DEFAULT_SEED=2023
fi
SEED="${SEED:-$DEFAULT_SEED}"
FOLDS="${FOLDS:-0 1 2 3 4}"
N_PART="${N_PART:-3}"
for FOLD in $FOLDS; do
  echo "[$MODE] $DATASET setting=$SETTING fold=$FOLD"
  COMMON=("dataset=$DATASET" "setting=$SETTING" "eval_fold=$FOLD"
          "n_sv=$N_SV" "test_label=$LABELS" "fold_protocol=$FOLD_PROTOCOL" "num_workers=$NUM_WORKERS"
          "seed=$SEED" "path.log_dir=$LOG_ROOT" "path.$DATASET.data_dir=$DATA_ROOT/$DATASET")
  if [[ "$MODE" == train ]]; then
    python train.py with "${COMMON[@]}" "mode=train" "n_steps=${N_STEPS:-40000}" \
      "pretrained_weights=$PRETRAINED_WEIGHTS" "save_snapshot_every=${SNAPSHOT_EVERY:-10000}"
  elif [[ "$MODE" == test ]]; then
    CKPT="$CHECKPOINT_ROOT/${DATASET}_setting${SETTING}_fold${FOLD}.pth"
    if [[ ! -f "$CKPT" ]]; then
      echo "Missing trained checkpoint: $CKPT" >&2; exit 1
    fi
    COMPAT=()
    if [[ "${LEGACY_CHECKPOINT:-0}" == 1 ]]; then
      COMPAT=("use_mb_head=True" "legacy_extra_scales=True" "legacy_uot_row_normalize=True")
    fi
    python test.py with "${COMMON[@]}" "${COMPAT[@]}" "mode=test" \
      "reload_model_path=$CKPT" "n_part=$N_PART" "supp_idx=${SUPP_IDX:-2}"
  else
    echo "Unknown mode $MODE" >&2; exit 1
  fi
done
