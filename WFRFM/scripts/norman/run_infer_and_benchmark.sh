#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${1:-}"
SOLVER="${2:-rk4}"
N_STEPS="${3:-100}"
MODEL_NAME="${4:-wfrfm_mass_50d_xpca_${SOLVER}_${N_STEPS}}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: $0 <run_dir> [solver] [n_steps] [model_name]" >&2
  exit 1
fi

"${SCRIPT_DIR}/infer_norman_test.sh" "${RUN_DIR}" "${SOLVER}" "${N_STEPS}"
"${SCRIPT_DIR}/benchmark_non_w.sh" "${RUN_DIR}" "${SOLVER}" "${N_STEPS}" "${MODEL_NAME}"
