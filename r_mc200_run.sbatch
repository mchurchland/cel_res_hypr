#!/bin/bash
#SBATCH --job-name=mc200
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-39%40
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${PROJECT_ROOT:?Submit r_mc200.sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${SLURM_ARRAY_TASK_MIN:-0}" != 0 || "${SLURM_ARRAY_TASK_STEP:-1}" != 1 ]]; then exit 1; fi
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("$PYTHON_BIN"); fi
"${RUNNER[@]}" "$OUTROOT/snapshot/mc200_recompute.py" run --run-dir "$OUTROOT" --device cuda:0 \
    --rank "${SLURM_ARRAY_TASK_ID:?}" --split "${SLURM_ARRAY_TASK_COUNT:?}"
