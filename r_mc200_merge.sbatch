#!/bin/bash
#SBATCH --job-name=mc200_merge
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:?Submit r_mc200.sbatch}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("$PYTHON_BIN"); fi
"${RUNNER[@]}" "$OUTROOT/snapshot/mc200_recompute.py" merge --run-dir "$OUTROOT"
