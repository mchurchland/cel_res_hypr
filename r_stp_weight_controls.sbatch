#!/bin/bash
#SBATCH --job-name=stp_weights
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
# Keep Numba caches off a read-only checkout and separate across array tasks.
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/stp-numba-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"

if [[ "${SLURM_ARRAY_TASK_MIN:-0}" != "0" || "${SLURM_ARRAY_TASK_STEP:-1}" != "1" ]]; then
  echo "Use a contiguous Slurm array starting at zero." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-${SLURM_ARRAY_JOB_ID:-local}}"
OUTROOT="${OUTROOT:-results/stp_weight_controls/${RUN_ID}}"
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV:-cel}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("${PYTHON_BIN}"); fi

EXTRA=()
# Optional mechanistic diagnostic; omitted by default at the user's request.
if [[ "${GAIN_CONTROL:-0}" == "1" ]]; then EXTRA+=(--gain-control); fi
if [[ "${GAIN_CONTROL:-0}" != "0" && "${GAIN_CONTROL:-0}" != "1" ]]; then
  echo "GAIN_CONTROL must be 0 or 1." >&2
  exit 1
fi

"${RUNNER[@]}" stp_weight_controls.py \
  --out-dir "${OUTROOT}" \
  --ce-adj "${CE_ADJ:-Connectome/ce_adj.npy}" \
  --ce-unknown-sign-weights "${CE_UNKNOWN_SIGN_WEIGHTS:-Connectome/ce_unknown_sign_weights.npy}" \
  --sensory-labels "${SENSORY_LABELS:-configs/stp_weight_controls/sensory_neurons.csv}" \
  --node-names "${NODE_NAMES:-Connectome/ce_nodes.txt}" \
  --ei-percent 0 10 20 30 40 50 \
  --families ce_empirical ce_shuffled ce_binary er_empirical er_signed_empirical er_halfnormal er_binary \
  --normalizations own_radius shared_ce_scale \
  --input-scales 0.1 0.5 \
  --input-schemes full restricted_matched_norm \
  --parameter-sets fast reference slow \
  --mixed-fractions 0.5 \
  --initial-radius 0.95 \
  --n-repeats "${TOTAL_REPEATS:-50}" \
  --seed "${BASE_SEED:-20261120}" \
  --washout "${WASHOUT:-500}" \
  --train-length "${TRAIN_LENGTH:-1500}" \
  --validation-length "${VALIDATION_LENGTH:-500}" \
  --test-length "${TEST_LENGTH:-500}" \
  --mc-max-delay "${MC_MAX_DELAY:-30}" \
  --narma-burnin "${NARMA_BURNIN:-1000}" \
  --repeat-split "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --repeat-rank "${SLURM_ARRAY_TASK_ID:-0}" \
  --backend cuda \
  --cuda-device "${CUDA_DEVICE:-0}" \
  --gpu-batch-size "${GPU_BATCH_SIZE:-64}" \
  --workers 1 \
  --resume "${EXTRA[@]}"
