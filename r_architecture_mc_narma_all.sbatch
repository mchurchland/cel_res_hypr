#!/bin/bash
# Submit from the repository root: sbatch r_architecture_mc_narma_all.sbatch
# This short setup job queues GPU preflight -> GPU array -> merge/analysis.
# Simulation takes place in the GPU jobs, not in this setup job.
#SBATCH --job-name=arch_submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=architecture-submit-%j.out
#SBATCH --error=architecture-submit-%j.err

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"

# Change these defaults here, or supply them through sbatch --export.
export TOTAL_REPEATS="${TOTAL_REPEATS:-100}"
export ARRAY_TASKS="${ARRAY_TASKS:-20}"
export CONDA_ENV="${CONDA_ENV:-cel}"
export RUN_ID="${RUN_ID:-architecture_mc_narma_${SLURM_JOB_ID:?}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

bash tools/submit_architecture_mc_narma.sh "$@"
