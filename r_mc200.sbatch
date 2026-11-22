#!/bin/bash
# On Orca, from Thesis: sbatch r_mc200.sbatch
# Controller submits a gated 40-worker GPU array and a compact result merge.
#SBATCH --job-name=mc200_submit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=mc200-submit-%j.out
#SBATCH --error=mc200-submit-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PROJECT_ROOT="$(pwd -P)"
export OUTROOT="$(realpath -m "${OUTROOT:-results/mc200_recompute/mc200_${SLURM_JOB_ID:?}}")"
export CONDA_ENV="${CONDA_ENV:-cel}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ARRAY_TASKS="${ARRAY_TASKS:-40}"
if [[ ! "$ARRAY_TASKS" =~ ^[1-9][0-9]*$ ]]; then exit 1; fi
mkdir -p logs
RUNNER=(conda run --no-capture-output -n "$CONDA_ENV" python)
if [[ -n "${PYTHON_BIN:-}" ]]; then RUNNER=("$PYTHON_BIN"); fi
"${RUNNER[@]}" mc200_recompute.py prepare --run-dir "$OUTROOT" "$@"
PREFLIGHT=$(sbatch --parsable "$OUTROOT/snapshot/r_mc200_preflight.sbatch")
PREFLIGHT="${PREFLIGHT%%;*}"
SWEEP=$(sbatch --parsable --dependency="afterok:$PREFLIGHT" --kill-on-invalid-dep=yes \
    --array="0-$((ARRAY_TASKS-1))%40" "$OUTROOT/snapshot/r_mc200_run.sbatch")
SWEEP="${SWEEP%%;*}"
MERGE=$(sbatch --parsable --dependency="afterok:$SWEEP" --kill-on-invalid-dep=yes "$OUTROOT/snapshot/r_mc200_merge.sbatch")
echo "Preflight: $PREFLIGHT; MC200 GPU array: $SWEEP; merge: ${MERGE%%;*}"
echo "Results: $OUTROOT"
