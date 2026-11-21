#!/bin/bash
#SBATCH --job-name=er_weights
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${PROJECT_ROOT:?Submit through r_er_weight_distribution.sbatch}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${SLURM_ARRAY_TASK_MIN:-0}" != 0 || "${SLURM_ARRAY_TASK_STEP:-1}" != 1 ]]; then echo 'Use an array starting at zero with step one' >&2; exit 1; fi
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("${PYTHON_BIN}"); fi
"${RUNNER[@]}" "${OUTROOT:?}/snapshot/er_weight_distribution.py" run \
    --run-dir "${OUTROOT}" --device cuda:0 --batch-size "${BATCH_SIZE:-24}" \
    --rank "${SLURM_ARRAY_TASK_ID:?}" --split "${SLURM_ARRAY_TASK_COUNT:?}"
