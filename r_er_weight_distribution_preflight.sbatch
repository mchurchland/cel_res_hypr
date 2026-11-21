#!/bin/bash
#SBATCH --job-name=er_weights_check
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("${PYTHON_BIN}"); fi
"${RUNNER[@]}" "${OUTROOT:?}/snapshot/er_weight_distribution.py" preflight \
    --run-dir "${OUTROOT}" --device cuda:0 --batch-size "${BATCH_SIZE:-24}"
