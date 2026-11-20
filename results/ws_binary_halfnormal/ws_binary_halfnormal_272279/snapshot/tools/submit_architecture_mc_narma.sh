#!/bin/bash
# Orca GPU submission; freezes sources/data and gates the array on a GPU audit.
# Extra arguments configure `prepare`, e.g. --normalizations own_radius shared_ce_scale.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v sbatch >/dev/null
export PROJECT_ROOT="$(pwd -P)"
export RUN_ID="${RUN_ID:-architecture_mc_narma_$(date -u +%Y%m%dT%H%M%SZ)}"
export OUTROOT="$(realpath -m "${OUTROOT:-results/architecture_mc_narma/${RUN_ID}}")"
export ANALYSIS_DIR="$(realpath -m "${ANALYSIS_DIR:-analysis/architecture_mc_narma/${RUN_ID}}")"
export CONDA_ENV="${CONDA_ENV:-cel}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export BOOTSTRAP="${BOOTSTRAP:-1000}"
export PYTHON_BIN="${PYTHON_BIN:-}"
ARRAY_TASKS="${ARRAY_TASKS:-20}"
if [[ ! "${ARRAY_TASKS}" =~ ^[1-9][0-9]*$ ]]; then echo 'Invalid ARRAY_TASKS' >&2; exit 1; fi
mkdir -p logs
RUNNER=(conda run --no-capture-output -n "${CONDA_ENV}" python)
if [[ -n "${PYTHON_BIN}" ]]; then RUNNER=("${PYTHON_BIN}"); fi
"${RUNNER[@]}" architecture_mc_narma.py prepare --run-dir "${OUTROOT}" \
    --repeats "${TOTAL_REPEATS:-100}" --seed "${BASE_SEED:-20261119}" "$@"
PREFLIGHT_SUBMISSION=$(sbatch --parsable --partition="${PARTITION:-normal}" \
    "${OUTROOT}/snapshot/r_architecture_mc_narma_preflight.sbatch")
PREFLIGHT_JOB="${PREFLIGHT_SUBMISSION%%;*}"
if [[ ! "${PREFLIGHT_JOB}" =~ ^[0-9]+$ ]]; then echo "Unexpected sbatch response: ${PREFLIGHT_SUBMISSION}" >&2; exit 1; fi
SWEEP_SUBMISSION=$(sbatch --parsable --partition="${PARTITION:-normal}" \
    --dependency="afterok:${PREFLIGHT_JOB}" --kill-on-invalid-dep=yes \
    --array="0-$((ARRAY_TASKS-1))" "${OUTROOT}/snapshot/r_architecture_mc_narma.sbatch")
SWEEP_JOB="${SWEEP_SUBMISSION%%;*}"
if [[ ! "${SWEEP_JOB}" =~ ^[0-9]+$ ]]; then echo "Unexpected sbatch response: ${SWEEP_SUBMISSION}" >&2; exit 1; fi
ANALYSIS_SUBMISSION=$(sbatch --parsable --partition="${PARTITION:-normal}" \
    --dependency="afterok:${SWEEP_JOB}" --kill-on-invalid-dep=yes \
    "${OUTROOT}/snapshot/r_architecture_mc_narma_analyze.sbatch")
echo "Preflight: ${PREFLIGHT_JOB}; GPU array: ${SWEEP_JOB}; merge/analysis: ${ANALYSIS_SUBMISSION%%;*}"
echo "Results: ${OUTROOT}"
echo "Analysis (PNG + CSV + Markdown): ${ANALYSIS_DIR}"
echo "Resume with the same settings and RUN_ID=${RUN_ID}; existing completed shards are validated and reused."
