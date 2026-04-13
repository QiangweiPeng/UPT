#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/lustre/home/2300017748/.conda/envs/ProGen/bin/python"

RUN_DIR="${1:-${REPO_ROOT}/results/checkpoints/norman/norman_norman_simulation_1_0.75_X_pca50_delta10_regm1_seed42_iter3000}"
SOLVER="${2:-rk4}"
N_STEPS="${3:-100}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" experiments/norman/infer_norman_test.py \
  --run-dir "${RUN_DIR}" \
  --solver "${SOLVER}" \
  --n-steps "${N_STEPS}"
