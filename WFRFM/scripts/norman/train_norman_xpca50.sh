#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/lustre/home/2300017748/.conda/envs/ProGen/bin/python"

RUN_NAME="${1:-norman_norman_simulation_1_0.75_X_pca50_delta10_regm1_seed42_iter3000}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" experiments/norman/train_norman_simulation.py \
  --condition-rep-path data/norman/norman_embeddings_filtered.pkl \
  --n-comps 50 \
  --n-iterations 3000 \
  --run-name "${RUN_NAME}"
