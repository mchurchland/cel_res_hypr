# Keep generated experiment output outside Git

Version source code, Slurm files, configuration, documentation and the small
source connectomes in `Connectome/`. Generated output stays on disk and is
transferred directly with rsync; it is not a code-deployment mechanism.

Ignored output includes `results/`, `final_results/`, `logs/`, root Slurm
stdout/stderr, Python/build caches, and generated files in `analysis/`,
`architecture_variant_figures/` and `metric_figures/`. Source scripts and Markdown
reports in the latter three folders remain eligible for version control.
Frozen snapshots and cached data are ignored even when they contain scripts.
Avoid `git add -f` for generated output.

`.gitignore` does not affect files already tracked. The cleanup removes those
paths from Git's index using `git rm --cached`; all local files remain intact.
It does not rewrite history, so older commits still contain their original data.
For a fresh code-only checkout after the cleanup is published, a shallow clone
avoids downloading that old history:

```bash
git clone --depth 1 https://github.com/mchurchland/Thesis.git Thesis-code
```

## First update on Orca after this cleanup

A normal pull of a commit that deletes tracked outputs can remove those files
from the cluster checkout. Use the data-preserving helper for this transition,
from the existing `Thesis` directory:

```bash
git fetch origin
git show origin/main:tools/update_orca_preserving_results.sh | bash
```

The helper fetches the remote, requires a fast-forward and clean source edits,
retains a backup branch, updates HEAD/index without changing the working tree,
then restores source paths only. Generated folders and existing frozen jobs
remain in place, including outputs that were previously tracked. It stops if
new remote source files would overwrite local untracked files. It does not
install packages, submit jobs, rewrite shared history, or push anything.

Subsequent ordinary pulls no longer track the ignored outputs. Download data
directly, for example from the local Thesis directory:

```bash
rsync -avP milesc@login.orca.pdx.edu:~/Thesis/results/ ./results/
```
