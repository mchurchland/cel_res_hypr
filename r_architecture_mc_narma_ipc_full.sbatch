#!/bin/bash
# Run from the repository root: sbatch r_architecture_mc_narma_ipc_full.sbatch
# Queues GPU preflight -> GPU array -> full-range merge/analysis.
#SBATCH --job-name=arch_full_ipc
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=architecture-full-ipc-submit-%j.out
#SBATCH --error=architecture-full-ipc-submit-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export TOTAL_REPEATS="${TOTAL_REPEATS:-100}"
export ARRAY_TASKS="${ARRAY_TASKS:-20}"
export CONDA_ENV="${CONDA_ENV:-cel}"
export RUN_ID="${RUN_ID:-architecture_full_ipc_${SLURM_JOB_ID:?}}"
export OUTROOT="${OUTROOT:-results/architecture_mc_narma_ipc_full/${RUN_ID}}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/architecture_mc_narma_ipc_full/${RUN_ID}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# IPC at every fraction. Existing MC/NARMA scores at 0–50% are reused.
read -r -a FRACTIONS <<< "${EI_FRACTIONS:-0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1}"
bash tools/submit_architecture_mc_narma.sh \
    --fractions "${FRACTIONS[@]}" \
    --overlap-parent "${PARENT_RUN_DIR:-results/architecture_mc_narma/architecture_mc_narma_271846}" \
    --reuse-parent-through .5 \
    --with-ipc --ipc-max-delay "${IPC_MAX_DELAY:-10}" \
    --ipc-max-degree "${IPC_MAX_DEGREE:-3}" "$@"
