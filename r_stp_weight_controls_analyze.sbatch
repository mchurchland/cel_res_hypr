#!/bin/bash
#SBATCH --job-name=stp_analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
: "${RUN_ID:?Pass RUN_ID from the sweep job}"
OUTROOT="${OUTROOT:-results/stp_weight_controls/${RUN_ID}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/stp_weight_controls/${RUN_ID}}"
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("${PYTHON_BIN}"); fi
"${RUNNER[@]}" stp_weight_controls.py --merge --out-dir "${OUTROOT}"
"${RUNNER[@]}" tools/analyze_stp_weight_controls.py --results-dir "${OUTROOT}" --out-dir "${ANALYSIS_DIR}"
