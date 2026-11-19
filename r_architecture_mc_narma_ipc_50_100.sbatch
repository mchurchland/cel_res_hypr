#!/bin/bash
# Compatibility entry point. Now includes the lower-half IPC backfill.
# Canonical command: sbatch r_architecture_mc_narma_ipc_full.sbatch
#SBATCH --job-name=arch_full_ipc
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=architecture-full-ipc-submit-%j.out
#SBATCH --error=architecture-full-ipc-submit-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
exec bash r_architecture_mc_narma_ipc_full.sbatch "$@"
