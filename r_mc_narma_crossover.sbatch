#!/bin/bash
#SBATCH --job-name=mc_narma_cross
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Submit from the repository root after creating logs/. No installation in jobs.
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLBACKEND=Agg

if [[ "${SLURM_ARRAY_TASK_MIN:-0}" != "0" || "${SLURM_ARRAY_TASK_STEP:-1}" != "1" ]]; then
  echo "Use a contiguous Slurm array starting at zero." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-local}}"
OUTROOT="${OUTROOT:-results/crossover_mc_narma10/${RUN_ID}}"
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then
  RUNNER=("${PYTHON_BIN}")
fi

"${RUNNER[@]}" mc_narma_crossover.py \
  --out-dir "${OUTROOT}/chunk_${SLURM_ARRAY_TASK_ID:-0}" \
  --ce-adj "${CE_ADJ:-Connectome/ce_adj.npy}" \
  --ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}" \
  --families cel_matched matched_er \
  --normalization-modes spectral_radius original_radius \
  --sign-fracs 0 0.1 0.2 0.3 0.4 0.5 \
  --rho-values 0.6 0.95 1.05 \
  --leak-values 0.6 1.0 \
  --input-scales 0.001 0.05 0.01 0.1 0.5 \
  --n-repeats "${TOTAL_REPEATS:-100}" \
  --seed "${BASE_SEED:-20261117}" \
  --repeat-seed-stride 100000 \
  --repeat-split "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --repeat-rank "${SLURM_ARRAY_TASK_ID:-0}" \
  --cuda "${CUDA_DEVICE:-0}" \
  --resume
