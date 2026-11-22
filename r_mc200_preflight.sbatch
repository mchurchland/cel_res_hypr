#!/bin/bash
#SBATCH --job-name=mc200_preflight
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:?Submit r_mc200.sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("$PYTHON_BIN"); fi
"${RUNNER[@]}" "$OUTROOT/snapshot/mc200_recompute.py" preflight --run-dir "$OUTROOT" --device cuda:0
