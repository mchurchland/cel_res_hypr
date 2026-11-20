# Audited architecture MC/NARMA10 sweep

Run from the repository root on Orca, with the existing GPU-enabled `cel`
Conda environment. No packages are installed by jobs. Required: PyTorch with
CUDA, NumPy, SciPy, scikit-learn, NetworkX (with `directed_edge_swap`), pandas,
and Matplotlib. EchoTorch's NARMA class is included unchanged with its license;
the full EchoTorch package is not required.

```bash
sbatch r_architecture_mc_narma_all.sbatch
```

This submits a short setup job that queues a GPU preflight, a dependent GPU
array, and dependent merge/analysis. Simulation jobs each request one GPU.
Set `TOTAL_REPEATS` and `ARRAY_TASKS` in that sbatch file to change their defaults.
You can also invoke `bash tools/submit_architecture_mc_narma.sh` directly.
Preflight runs unit tests, checks all planned NARMA streams, compares actual CUDA
recurrence and scoring against CPU references, and evaluates representative
conditions for every selected model/normalization. Failure prevents the array
from starting. GPU requests never fall back to CPU. The final merge/plotting job
uses CPUs because it does not simulate reservoirs.

Defaults:

| Setting | Values |
|---|---|
| Models | Ten controls below |
| Inhibitory edge fraction | 0%, 10%, 20%, 30%, 40%, 50% |
| Radius target | 0.6, 0.95, 1.05 |
| Leak | 0.6, 1.0 |
| Input scale | 0.01, 0.1, 0.5, 1.5 |
| Bias | 0, fixed |
| Normalizations | Own radius, per-family fixed reference, shared CE multiplier |
| Realizations | 100, paired across all conditions |
| Grid | 24 settings; 432,000 paired MC/NARMA evaluations total |
| State/readout precision | float64 / float64 |
| Washout / training / validation / test | 500 / 1500 / 500 / 500 |
| MC delays | 1–30 primary; 31–60 retained as a tail diagnostic |
| Input projection | Same scalar input to all nodes through a shared random vector |
| STP | None |

The ten models are CE empirical, CE shuffled magnitudes within sign classes,
CE directed degree rewiring, CE binary, CE half-normal, ER exact signed CE
weights, ER binary, ER half-normal, directed WS-like exact signed CE weights,
and CE with unknown-sign edges removed. ER is a uniform directed fixed-edge-count
random graph; WS rewiring probability is 0.1. The removed condition retains 297
nodes and full-CE magnitude units, but has 1,740 rather than 3,604 edges. It is
not density-matched to the full-CE ER controls.

Own-radius normalization equalizes each current matrix radius. Fixed-reference
uses each model's radius at the CE known-edge sign fraction (24.2529%), held
constant across E/I fractions. Shared-CE uses CE's multiplier at each current
fraction for every model; it varies across fractions. Exact signed-weight
matching after scaling holds under the shared multiplier. In the other protocols,
raw weights match and the applied multipliers can differ.

Change repeat count or array size without editing Python:

```bash
TOTAL_REPEATS=20 ARRAY_TASKS=10 bash tools/submit_architecture_mc_narma.sh
```

Additional arguments are forwarded to `prepare`, e.g.:

```bash
TOTAL_REPEATS=50 bash tools/submit_architecture_mc_narma.sh \
  --normalizations own_radius shared_ce_scale --inputs .01 .1 .5
```

Other environment options: `CONDA_ENV`, `PYTHON_BIN`, `PARTITION`, `BATCH_SIZE`,
`BASE_SEED`, `RUN_ID`, `OUTROOT`, `ANALYSIS_DIR`, `BOOTSTRAP`. Array simulation
jobs request one GPU, four CPU cores, 24 GB RAM, and 12 hours. This wall-time
request is a limit, not a measured runtime prediction. A production GPU estimate
should be based on preflight timings.

Results: `results/architecture_mc_narma/<RUN_ID>/`. This contains the immutable
manifest, frozen code/data snapshot, cached network spectra/task streams,
preflight evidence, hashed completed CSV shards, and merged
`architecture_mc_narma.ALL.csv`.

Analysis: `analysis/architecture_mc_narma/<RUN_ID>/`. Outputs include PNG mean
performance curves, paired-configuration win fractions, input-conditioned
advantage maps, hyperparameter variance contributions including interactions,
CSV tables, dynamics diagnostics, and a Markdown report. Whole-realization
bootstrap resampling preserves pairing across models and configurations.

Resume with the same `RUN_ID` and preparation settings. Completed shards are
validated and reused. Source/settings changes require a new run ID. A nonempty
analysis directory is never overwritten; use a fresh `ANALYSIS_DIR` if an earlier
analysis was interrupted. Keep the recorded snapshot if updating the repository
while jobs are queued. Jobs execute the snapshot, not the working checkout.

## Important protocol change

The audit found that ReservoirPy 0.4.2, used by the archived crossover run, uses
NARMA lag/history indices that differ from its documented conventional equation.
The new protocol uses the unmodified, pinned EchoTorch implementation of the
conventional recurrence. This and the stabilized readout define a new protocol;
new and archived NARMA scores must not be silently pooled. Existing data/code
remain unchanged. See `analysis/architecture_pipeline_audit_2026-11-19/AUDIT.md`.

Only nonfinite generated NARMA targets trigger a prespecified seed replacement,
before evaluating networks. Every rejected seed and accepted seed is cached;
all conditions share the accepted stream. Finite poor performance is retained.
No test targets choose readout penalties. Reservoir states stay continuous
through train/validation/test; readouts fit training data only.

Local checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_architecture_pipeline.py'
```
