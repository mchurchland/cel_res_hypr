# Refactoring audit

Audit date: 2026-11-09. Original revision: `749f5ab`.
Branch: `refactor/research-codebase`.

## Scope and baseline

Scientific results and source data are archival inputs. `thesis_tex/` is an
untracked manuscript tree whose contents must not be edited. No relocation is
needed. Before changing implementations, SHA-256 checksums were recorded for
8,651 tracked data/result files and 607 manuscript files in a temporary audit
manifest. The initial working tree had only the untracked `thesis_tex/` tree.

The complete tracked directory inventory is in [baseline/inventory.json](baseline/inventory.json).
All Python modules, their imports, definitions, and original line counts are
listed in [baseline/modules.json](baseline/modules.json).

Both `python -m unittest network_stats.test_stats -v` and
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q` passed all 15 existing
tests. The deterministic CPU experiment in `tools/smoke_experiment.py` was run
against the original implementation. Its 54 rows cover 27 controls under both
normalization modes. [baseline/baseline.json](baseline/baseline.json) records the
graph, seed, grid, simulation settings, dependency versions, and trajectory;
[baseline/invariance_variants.csv](baseline/invariance_variants.csv) records the
original CSV schema and values.

For the first real-weight row: MC = 1.8750983476638794,
IPC = 2.031848927261308, KR = 3.803112268447876,
GR = 2.408902406692505; target radius = 0.8,
measured radius = 0.7999995946884155. These are small synthetic regression
values, not estimates of manuscript results. Python 3.13.9, PyTorch 2.7.1,
NumPy 2.2.5, SciPy 1.16.0, pandas 2.3.1, and one CPU thread were used.

## Existing workflow and dependencies

`r_*.sbatch` launches `inv_arc_test.py`, parameter-justification scripts, or
the triad experiment. The main CLI assigns grid/repeat partitions and seeds,
loads the processed connectome, and dispatches variants.
`reservoir_variants.py` builds each control, evaluates metrics through
`network_stats/run_one.py`, and writes or selectively patches CSV rows.
`util/util.py` combines loading, preprocessing, topology construction, weight
controls, normalization, and seeding. `util/merge.py` and `util/graph_utils.py`
aggregate chunk outputs. `graph_hist.py` and specialized figure scripts consume
merged results. `GRAPHING_README.md` identifies canonical manuscript inputs.

| Area | Audit finding | Priority |
|---|---|---|
| `graph_hist.py` (5,721 lines) | CLI, styling, statistics, tables, and many figure families share one module | High |
| `util/util.py` (912 lines) | Loading and graph generation share hidden module globals | High |
| `reservoir_variants.py` (809 lines) | Configuration, evaluation, dispatch, and CSV persistence are mixed | High |
| Tests | Good KR/GR protocol coverage; little trajectory, MC/IPC, architecture, or aggregation coverage | High |
| Reproducibility | No dependency manifests or recorded CPU numerical baseline | High |
| Documentation | Incorrect SVD/covariance explanation, IPC Cholesky claim, obsolete builder return values | Medium |
| Duplicates | Identical path-versioning, CV, hyperparameter averaging, and seed helpers | Medium |
| Public interfaces | Other figure scripts import private `graph_hist` helpers; tests patch `run_one` globals | High compatibility risk |

Processed arrays, original spreadsheets, all result families, logs, figure
assets, sensitivity scripts, and the archived `depreciated/` launchers must
remain available. The legacy root entry points and SLURM command lines are
compatibility interfaces.

## Scientific issues to preserve and isolate

The independent Critic identified the following historical behavior. None is
authorized for a scientific correction in this refactor.

1. **High:** `--unknown-sign-inhibitory-frac` is validated but not forwarded by
   the main CLI. The assignment helper always uses its default fraction there.
2. **Medium:** `sign_test_er` hardcodes ER probability 0.1; `ctx.er_p` does not
   control this branch. Even with edge-count matching, changing the initial
   probability changes the RNG path and graph realization.
3. **Medium:** `run_reservoir_with_pre` uses the default tensor dtype for its
   state buffers. Production float32 works; float64 weights fail under the
   default float32 setting. The other trajectory functions preserve `W.dtype`.
4. **Medium:** partition ranks prefer zero-based interpretation. Values
   `1..split-1` are ambiguous; `rank == split` aliases the last partition.
5. IPC evaluates single-delay Legendre targets with squared Pearson
   correlation, excluding mixed-delay polynomial products. It is a restricted
   IPC measure. Ridge uses `torch.linalg.solve` with `(alpha + 1e-6) I`.
6. Centered effective rank weights singular values, not their squares. Constant
   ensembles have rank zero after centering; the absolute cutoff is `1e-12`.

Separate corrections should first define the intended protocol, compare old
and corrected runs, label affected results, and receive scientific approval.

## Prioritized implementation plan

1. **Baseline and regression guardrails:** preserve original numbers, add
   deterministic CPU and controlled-example tests, record known limitations.
2. **Network construction:** extract connectome IO, topology, weight controls,
   normalization, and the builder into cohesive modules; retain `util.util`
   reexports and unchanged numerical expressions/RNG ordering.
3. **Experiments:** extract simulation configuration, evaluation helpers, CSV
   schema/persistence, grid/partition/subset helpers; simplify repeated variant
   dispatch without changing labels or seeds. Keep legacy imports.
4. **Analysis and plotting:** extract independent feature families and shared
   helpers; preserve backend selection, styles, grouping, CSV parsing, figure
   options, and imported names. Do not merge superficially similar column
   normalizers: plotting and aggregation intentionally differ.
5. **Research documentation and acceptance:** installation manifests,
   architecture/protocol guide, reproduction commands, changelog, final Critic
   report, full CPU suite, CLI smoke checks, figure comparison, protected-file
   checksum verification.

Each bounded change receives an independent ACCEPT/REVISE/REJECT review before
commit. CPU numerical comparisons use `rtol=2e-5, atol=2e-6` to accommodate
float32 BLAS/eigensolver/SVD differences across systems; the current environment
is also checked for exact preservation. GPU/cluster production sweeps are not
rerun as part of this architectural change.
