# Reproducing experiments and figures

## Environment and baseline

Use Python 3.13.9 for the closest match to the recorded CPU baseline:

```bash
python -m pip install -e '.[dev]' -c requirements-baseline.txt
export MPLCONFIGDIR=/tmp/matplotlib-cel
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pytest -q
python -m unittest network_stats.test_stats
```

`requirements-baseline.txt` constrains the original direct dependency versions.
It does not lock transitive dependencies, hardware, BLAS, fonts, or CPU/CUDA
wheel variants. `docs/baseline/baseline.json` records Python/library versions,
seed, graph, full small-experiment configuration, trajectory, schema, and 54
rows captured before implementation changes at `749f5ab`.

```bash
python -m tools.smoke_experiment --out-dir /tmp/cel-cpu-check
```

The regression suite compares those outputs with `rtol=2e-5, atol=2e-6`.
The tolerance allows small float32 eigensolver/SVD/BLAS differences; labels,
settings, shapes, schema, and random assignments are checked separately. On the
recorded environment, the refactored 54 rows match exactly. These short-series
numbers are regression fixtures, not replacements for production results.

## Inputs and a minimal connectome run

`Connectome/ce_adj.npy`, `ce_nodes.txt`, and `ce_unknown_sign_weights.npy`
are the committed processed inputs. Source spreadsheets remain at the repository
root. Inspect `python util/read_xls.py --help` for explicit regeneration options;
regenerate into a separate destination if comparing preprocessing. Excel import
requires the optional `excel` dependencies. Input checksums are recorded in
`baseline/input_checksums.json`.

The [README](../README.md#a-small-complete-workflow) gives an executable
connectome → raw CSV → merged summaries → histogram example using `/tmp` outputs.
It computes all four metrics at one target radius, three leaks, four input scales,
zero bias, and one repeat on CPU, using the production time-series settings.
Add `--rank-only` for a faster KR/GR check; this leaves MC/IPC blank.

Use a new output directory per experiment/configuration. Preserve original
inputs and archived result trees when investigating a scientific correction.
`--out-dir` chooses the raw output location. `--csv-name` overrides the raw CSV
name. The merge command's historical default variant glob is `ce_real_reps.csv`,
so supply `--glob-variants` explicitly for current `invariance_variants.csv`
outputs. `util/merge.py` emits three tables; `graph_hist.py` generates figures.

## Production settings and repeat assignment

Production grids are defined in `cel_res_hypr/experiments/sampling.py`:

| Parameter | Default values |
|---|---|
| Spectral radius | 0.6, 0.8, 0.95, 1.05 |
| Leak | 0.6, 0.8, 1.0 |
| Input scale | 0.1, 0.5, 1.0, 1.5 |
| Uniform bias half-range | 0.0, 0.1 |

The full default grid has 96 combinations. `--rho-values`, `--neuron-biases`, and
specialized experiment options can replace parts of it. Grid order is the
Cartesian-product order `(rho, leak, input_scale, bias)`. Parameter and repeat
partitions are independent, contiguous partitions. Use zero-based ranks: the
legacy parser's ambiguous one-based alias is preserved, not recommended.

The main `r_data.sbatch`, `r_data_shuf.sbatch`, and `r_data_frac.sbatch` launchers
encode 200 repeats across 20 array tasks, a base seed of 12345, and repeat seed
stride 100000. Inspect each committed launcher for its exact job selection,
normalization, graph data, paths, and cluster resource settings. The fixed-radius
shuffle launcher uses its own array configuration. Current primary launchers
default to KR/GR rank-only reruns; `RANK_ONLY=0 KR_ONLY=0` selects all metrics
where those switches are supported. Do not mix those reruns with incompatible
older metric definitions.

Key seed assignments in the main sweep are:

| Random choice | Seed |
|---|---|
| Repeat base | `seed + repeat_index * repeat_seed_stride` |
| Reservoir/input construction | Repeat base + 1 |
| Empirical shuffle/sample overrides | Repeat base; signed/unsigned binary topology controls use repeat base + 1 |
| Neuron bias | Construction seed + 17000003 |
| KR streams | Repeat base + 29000003 |
| GR streams | Repeat base + 23000003 |
| Unknown-sign assignment | Repeat base + `unknown_sign_seed_offset` (default 23000000) |
| Disjoint node subsets | Repeat base + `subset_seed_offset` (default 7000000) |

Construction deliberately resets the global PyTorch generator. Training inputs
for MC/IPC use the following global stream; rank and bias sampling use dedicated
generators. Do not insert random draws or reorder construction steps when
comparing runs. Keep normalization settings and input/output subset choices
with each run; seed equality alone does not describe an experiment.

Retain `src`, `shuffle_id`, `seed`, normalization, and grid columns. Original
SLURM paths use `chunk_<id>/`; aggregation uses source plus repeat identity and
averages duplicate grid points before dispersion. Preserve order for legacy
rows whose repeats are inferred from contiguous blocks. Different normalization
conditions require separate analysis or distinct mode suffixes.

## Major experiment families

| Goal | Launcher / analysis |
|---|---|
| Architecture comparison | `r_data.sbatch` |
| Topology/weight shuffles | `r_data_shuf.sbatch` |
| Family-baseline-radius controls | `r_data_shuf_baseline_rho.sbatch` |
| Negative-edge-fraction sweeps | `r_data_frac.sbatch` |
| Sign/normalization ablation | `r_sign_norm_ablation.sbatch` |
| Triad activation | `r_triad_sign_fraction.sbatch`, `network_stats/triad_sign_fraction_experiment.py` |
| MC/IPC delays and polynomial orders | `r_mc_delay.sbatch`, `r_ipc_delay.sbatch`, `r_ipc_order.sbatch`, `r_ipc_methodology.sbatch` |
| GR shared-tail sensitivity | `r_gr_common_tail.sbatch` |

Methodology scripts in `Parameter_justification/` recompute their experiments;
they are not generally plot-only readers. The GR sensitivity run fixes the
seven-step prefix and varies the shared continuation, using the same
length-three reference stream as production for each seed.

## Canonical results and manuscript figures

Use [GRAPHING_README.md](../GRAPHING_README.md) for exact final inputs, flags,
figure layouts, and filenames. Its primary architecture/shuffle/sign-fraction
inputs are the curated `final_results/**/combined.ALL.GRKR_erank.rank_updated.csv`
files. Commit `1b2344a` removed the three obsolete `final_results_GRKR_*` and
`final_results_KR_*` result trees from the checkout. Their alternative/intermediate
calculations remain available in Git history, for example through
`git show 1b2344a^:<historical-file-path>`. The curated `final_results/` tree was
unchanged by that cleanup.

The generic merged outputs are `combined.ALL.csv`, `mean_by_group.ALL.csv`, and
`dispersion_by_group.ALL.csv`. Plot merged tables rather than individual chunks.
Production-specific figure instructions may copy selected outputs into stable
manuscript filenames; for exploratory validation, use a temporary output root.
Some plot paths version existing outputs, while others replace a stable path.
GR is excluded from plots labeled higher-is-better mean performance and has a
separate synthesis script, `plot_generalization_rank_summary.py`.

Validate eight plotting paths without touching manuscript figures:

```bash
python -m tools.plot_validation \
  --out-dir /tmp/cel-plot-check \
  --reference-manifest docs/baseline/plot_comparison.json
```

The output directory must be empty. This creates 23 synthetic PNG/CSV/LaTeX
artifacts, a log, and a hash manifest. Saved image DPI is capped at 50 to control
memory use. The recorded environment reproduces all 23 original hashes exactly;
font/library/backend changes can affect raster hashes across environments.

To compare against the original source checkout using the same driver:

```bash
mkdir -p /tmp/cel-original-source
git archive 749f5ab -- '*.py' | tar -x -C /tmp/cel-original-source
python tools/plot_validation.py \
  --source-root /tmp/cel-original-source \
  --out-dir /tmp/cel-original-plots \
  --reference-manifest docs/baseline/plot_comparison.json
```

The original repository remains accessible through Git; no scientific data or
published figures need to be copied or overwritten for this comparison.

## Verification scope and remaining uncertainty

Tests cover trajectories/orientation, spectral scaling, topology/sign invariants,
controlled MC/IPC targets, rank ensembles/degeneracies, seeds, CSV updates,
aggregation, CLI execution, and representative real rendering. The package is
also built and checked from an isolated temporary installation. Historical
limitations and proposed separate corrections are listed in
[AUDIT.md](AUDIT.md) and [CRITIC_REPORT.md](CRITIC_REPORT.md).

Production GPU/SLURM sweeps and every full-resolution manuscript figure were
not rerun. No numerical algorithm or historical scientific issue was silently
corrected. The checksum evidence in `baseline/preservation.json` records the
state at the end of the refactor (`89466f0`), before the intentional deletions in
`1b2344a`; it is not a claim that those deleted outputs remain in the checkout.
The local `thesis_tex/` checkout is maintained separately and ignored by the
parent repository. It is not distributed with this repository; obtain the
manuscript separately when needed. Its files remain untouched by this cleanup.
