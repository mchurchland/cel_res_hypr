#!/bin/bash
# From the Thesis repository root: sbatch r_ws_binary_halfnormal.sbatch
# Computes only the two missing WS controls; reuses completed comparators.
#SBATCH --job-name=ws_weights_submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=ws-weights-submit-%j.out
#SBATCH --error=ws-weights-submit-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export TOTAL_REPEATS="${TOTAL_REPEATS:-100}"
export ARRAY_TASKS="${ARRAY_TASKS:-20}"
export CONDA_ENV="${CONDA_ENV:-cel}"
export RUN_ID="${RUN_ID:-ws_binary_halfnormal_${SLURM_JOB_ID:?}}"
export OUTROOT="${OUTROOT:-results/ws_binary_halfnormal/${RUN_ID}}"
export ANALYSIS_DIR="${ANALYSIS_DIR:-analysis/ws_binary_halfnormal/${RUN_ID}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash tools/submit_architecture_mc_narma.sh \
    --families ws_binary ws_halfnormal \
    --fractions 0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1 \
    --comparison-parent "${PARENT_RUN_DIR:-results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871}" \
    --with-ipc --ipc-max-delay 10 --ipc-max-degree 3 "$@"
