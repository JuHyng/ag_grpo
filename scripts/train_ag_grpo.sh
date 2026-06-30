#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 && "$1" != --* ]]; then
  DATASET=${DATASET:-$1}
  shift
else
  DATASET=${DATASET:-math}
fi

CONFIG=${CONFIG:-configs/${DATASET}_ag_grpo.yaml}
MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints}
WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_MODE

TIME_STAMP=${TIME_STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-llada/ag_grpo/${DATASET}/time:${TIME_STAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}

python -u train.py \
  --config "${CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --dataset "${DATASET}" \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
