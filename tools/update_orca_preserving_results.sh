#!/bin/bash
# Run from Thesis on Orca after the cleanup commit reaches GitHub.
# Unlike a normal pull, this preserves formerly tracked result files in place.
# Usage from an older checkout that does not have this script yet:
#   git fetch origin
#   git show origin/main:tools/update_orca_preserving_results.sh | bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
remote="${1:-origin}"
branch="${2:-main}"
git fetch "${remote}"
target="refs/remotes/${remote}/${branch}"
git rev-parse --verify "${target}^{commit}" >/dev/null
if ! git merge-base --is-ancestor HEAD "${target}"; then
    echo 'Stopped: local commits diverge from the remote. No local branch or files were changed.' >&2
    exit 1
fi
# These folders contain locally generated output. Preserve even their reports
# and scripts during this update; executable sweep code lives outside them.
code_paths=(. ':(exclude)results/**' ':(exclude)final_results/**'
    ':(exclude)analysis/**' ':(exclude)logs/**'
    ':(exclude)architecture_variant_figures/**' ':(exclude)metric_figures/**'
    ':(exclude)build/**' ':(exclude)**/__pycache__/**'
    ':(exclude)*.pyc' ':(exclude)*.pyo' ':(exclude)*.out'
    ':(exclude)*.err' ':(exclude)*.log' ':(exclude)*.egg-info/**')
if ! git diff --quiet --ignore-submodules=untracked -- "${code_paths[@]}" ||
   ! git diff --cached --quiet --ignore-submodules=untracked -- "${code_paths[@]}"; then
    echo 'Stopped: local source-code edits or staged changes exist. Save them before updating. Results were untouched.' >&2
    exit 1
fi
# A newly tracked source file must not overwrite a local untracked file.
while IFS= read -r -d '' path; do
    if [[ -e "${path}" || -L "${path}" ]]; then
        echo "Stopped: new remote source path already exists locally: ${path}" >&2
        exit 1
    fi
done < <(git diff --name-only --diff-filter=A -z HEAD "${target}" -- "${code_paths[@]}")
backup="backup/orca-before-output-cleanup-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git branch "${backup}" HEAD
# --mixed updates HEAD and the index only: no working-tree data are deleted.
git reset --mixed "${target}" >/dev/null
# Update source files only; all generated directories remain byte-for-byte in place.
git restore --source="${target}" --worktree -- "${code_paths[@]}"
echo "Updated source to ${remote}/${branch}; preserved all generated folders in place."
echo "Previous commit retained at ${backup}."
