#!/bin/bash
# Submit the experiment, then merge/analyze only after the entire array succeeds.
# Run this script yourself on Orca; it does not install anything.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v sbatch >/dev/null
mkdir -p logs
export RUN_ID="${RUN_ID:-stp_weights_$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_SUBMISSION=$(sbatch --parsable --partition="${PARTITION:-normal}" "$@" r_stp_weight_controls.sbatch)
SWEEP_JOB="${SWEEP_SUBMISSION%%;*}"
if [[ ! "${SWEEP_JOB}" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch response: ${SWEEP_SUBMISSION}" >&2
  exit 1
fi
echo "Sweep job: ${SWEEP_JOB}; RUN_ID=${RUN_ID}"
echo "Results: ${OUTROOT:-results/stp_weight_controls/${RUN_ID}}"
echo "Resume with: RUN_ID=${RUN_ID} bash tools/submit_stp_weight_controls.sh"
ANALYSIS_SUBMISSION=$(sbatch --parsable --partition="${PARTITION:-normal}" \
  --dependency="afterok:${SWEEP_JOB}" r_stp_weight_controls_analyze.sbatch)
echo "Merge/analysis job: ${ANALYSIS_SUBMISSION%%;*}"
echo "Analysis: ${ANALYSIS_DIR:-analysis/stp_weight_controls/${RUN_ID}}"
