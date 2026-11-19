#!/bin/bash
#SBATCH --job-name=arch_analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("${PYTHON_BIN}"); fi
"${RUNNER[@]}" "${OUTROOT:?}/snapshot/architecture_mc_narma.py" merge --run-dir "${OUTROOT}"
"${RUNNER[@]}" "${OUTROOT}/snapshot/tools/analyze_architecture_mc_narma.py" \
    --run-dir "${OUTROOT}" --out-dir "${ANALYSIS_DIR:?}" --bootstrap "${BOOTSTRAP:-1000}"
