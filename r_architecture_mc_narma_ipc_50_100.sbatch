#!/bin/bash
# From the repository root: sbatch r_architecture_mc_narma_ipc_50_100.sbatch
# Queues GPU preflight -> GPU array -> merge and PNG/CSV analysis.
#SBATCH --job-name=arch_upper_ipc
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=architecture-upper-ipc-submit-%j.out
#SBATCH --error=architecture-upper-ipc-submit-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"

export TOTAL_REPEATS="${TOTAL_REPEATS:-100}"
export ARRAY_TASKS="${ARRAY_TASKS:-20}"
export CONDA_ENV="${CONDA_ENV:-cel}"
export RUN_ID="${RUN_ID:-architecture_upper_ipc_${SLURM_JOB_ID:?}}"
export OUTROOT="${OUTROOT:-results/architecture_mc_narma_ipc_50_100/${RUN_ID}}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/architecture_mc_narma_ipc_50_100/${RUN_ID}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# 50% is an overlap point with run 271846; new fractions are 60–100%.
read -r -a FRACTIONS <<< "${EI_FRACTIONS:-0.5 0.6 0.7 0.8 0.9 1.0}"

bash tools/submit_architecture_mc_narma.sh \
    --fractions "${FRACTIONS[@]}" \
    --overlap-parent "${PARENT_RUN_DIR:-results/architecture_mc_narma/architecture_mc_narma_271846}" \
    --with-ipc --ipc-max-delay "${IPC_MAX_DELAY:-10}" \
    --ipc-max-degree "${IPC_MAX_DEGREE:-3}" "$@"
