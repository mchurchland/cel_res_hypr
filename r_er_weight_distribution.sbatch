#!/bin/bash
# Submit from Thesis on Orca: sbatch r_er_weight_distribution.sbatch
# Controller freezes the experiment and submits a GPU-gated GPU array.
#SBATCH --job-name=er_weights_submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=er-weights-submit-%j.out
#SBATCH --error=er-weights-submit-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Thesis repository root}}"
export PROJECT_ROOT="$(pwd -P)"
export RUN_ID="${RUN_ID:-er_weights_${SLURM_JOB_ID:?}}"
export OUTROOT="$(realpath -m "${OUTROOT:-results/er_weight_distribution/${RUN_ID}}")"
export ANALYSIS_DIR="$(realpath -m "${ANALYSIS_DIR:-analysis/er_weight_distribution/${RUN_ID}}")"
export CONDA_ENV="${CONDA_ENV:-cel}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export BOOTSTRAP="${BOOTSTRAP:-1000}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ARRAY_TASKS="${ARRAY_TASKS:-20}"
if [[ ! "${ARRAY_TASKS}" =~ ^[1-9][0-9]*$ ]]; then echo 'Invalid ARRAY_TASKS' >&2; exit 1; fi
mkdir -p logs
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("${PYTHON_BIN}"); fi
"${RUNNER[@]}" er_weight_distribution.py prepare --run-dir "${OUTROOT}" \
    --parent-run "${PARENT_RUN_DIR:-results/architecture_mc_narma_ipc_full/architecture_full_ipc_271871}" \
    --repeats "${TOTAL_REPEATS:-100}" "$@"
PREFLIGHT=$(sbatch --parsable --partition="${PARTITION:-normal}" "${OUTROOT}/snapshot/r_er_weight_distribution_preflight.sbatch")
PREFLIGHT="${PREFLIGHT%%;*}"
if [[ ! "${PREFLIGHT}" =~ ^[0-9]+$ ]]; then echo 'Invalid preflight submission response' >&2; exit 1; fi
SWEEP=$(sbatch --parsable --partition="${PARTITION:-normal}" --dependency="afterok:${PREFLIGHT}" \
    --kill-on-invalid-dep=yes --array="0-$((ARRAY_TASKS-1))" "${OUTROOT}/snapshot/r_er_weight_distribution_run.sbatch")
SWEEP="${SWEEP%%;*}"
if [[ ! "${SWEEP}" =~ ^[0-9]+$ ]]; then echo 'Invalid sweep submission response' >&2; exit 1; fi
ANALYSIS=$(sbatch --parsable --partition="${PARTITION:-normal}" --dependency="afterok:${SWEEP}" \
    --kill-on-invalid-dep=yes "${OUTROOT}/snapshot/r_er_weight_distribution_analyze.sbatch")
echo "GPU preflight: ${PREFLIGHT}; GPU array: ${SWEEP}; merge/analysis: ${ANALYSIS%%;*}"
echo "Results: ${OUTROOT}"
echo "Analysis: ${ANALYSIS_DIR}"
