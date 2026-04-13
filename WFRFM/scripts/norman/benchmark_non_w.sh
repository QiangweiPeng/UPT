#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="/lustre/home/2300017748/.conda/envs/ProGen/bin/python"

RUN_DIR="${1:-${REPO_ROOT}/results/checkpoints/norman/norman_norman_simulation_1_0.75_X_pca50_delta10_regm1_seed42_iter3000}"
SOLVER="${2:-rk4}"
N_STEPS="${3:-100}"
MODEL_NAME="${4:-wfrfm_mass_50d_xpca_${SOLVER}_${N_STEPS}}"

OUTPUT_SUBDIR="test_inference_1to1_${SOLVER}_steps${N_STEPS}"
PRED_PATH="${RUN_DIR}/${OUTPUT_SUBDIR}/predictions_test_genes_1to1.h5ad"
SUMMARY_TAG="benchmark_non_w_${SOLVER}_${N_STEPS}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" experiments/benchmark_simulation_non_w.py \
  --truth-adata evaluation/norman/norman_simulation_1_0.75_test.h5ad \
  --pred-adata "${PRED_PATH}" \
  --ctrl-adata evaluation/norman/norman_simulation_1_0.75_ctrl.h5ad \
  --truth-condition-key condition \
  --pred-condition-key target_condition \
  --model-name "${MODEL_NAME}" \
  --dataset-name norman \
  --output-csv "${RUN_DIR}/${SUMMARY_TAG}_per_condition.csv" \
  --summary-csv "${RUN_DIR}/${SUMMARY_TAG}_summary.csv" \
  --reference-summary-csv evaluation/results/norman/norman_summary.csv \
  --reference-model cellflow \
  --compare-output-csv "${RUN_DIR}/${SUMMARY_TAG}_vs_cellflow.csv" \
  --compare-output-json "${RUN_DIR}/${SUMMARY_TAG}_vs_cellflow.json"
