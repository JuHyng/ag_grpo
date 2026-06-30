#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

DATASET=sudoku CONFIG=configs/sudoku_ag_grpo.yaml bash scripts/train_ag_grpo.sh
