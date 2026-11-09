# C. elegans reservoir computing

Research software for testing how connectome topology, synaptic weights, and
excitatory/inhibitory edge balance affect reservoir performance and robustness
across hyperparameter choices. Biological, synthetic, and shuffled networks are
compared using linear memory capacity (MC), restricted information processing
capacity (IPC), kernel rank (KR), and shared-tail generalization rank (GR).
Higher MC/IPC/KR indicates greater capacity or separation; lower GR indicates a
lower-dimensional final-state ensemble under the specified shared-tail protocol.
Robustness is summarized primarily by the coefficient of variation across a
fixed hyperparameter grid.

The repository contains reusable Python components, compatible research scripts,
SLURM launchers, original data, and archived results. The refactor preserves
scientific equations, seeds, grids, CSV schemas, and statistical grouping.
Historical scientific limitations are documented in [docs/AUDIT.md](docs/AUDIT.md).

## Installation

The recorded CPU baseline uses Python **3.13.9**. Package metadata allows Python
3.10+, but other Python/dependency combinations have not received the same
numerical validation. From a suitable virtual or Conda environment:

```bash
python -m pip install -e '.[dev]' -c requirements-baseline.txt
export MPLCONFIGDIR=/tmp/matplotlib-cel
```

The constraints record direct dependency versions, not a complete environment
lock. Select an appropriate PyTorch CPU/CUDA build for your machine; production
SLURM scripts expect a Conda environment named `cel`. Spreadsheet regeneration
also needs `python -m pip install -e '.[excel]'`. Processed `.npy` inputs are
already present, so spreadsheet tools are unnecessary for ordinary runs.

Run research commands from the repository root. Data/results and specialized
figure scripts remain in the checkout; they are not bundled in the Python wheel.

## A small complete workflow

First exercise all four metrics on a deterministic eight-node synthetic graph:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tools.smoke_experiment --out-dir /tmp/cel-smoke
python -m pytest -q
```

The smoke command writes 54 rows (27 controls, two normalization modes), an
experiment CSV, and JSON containing settings, versions, seeds, and a trajectory.
It uses short time series for regression testing; it does not estimate the
manuscript capacities. Its two normalization modes should be analyzed separately.

For the processed connectome, run a small CPU experiment and aggregate it:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python inv_arc_test.py \
  --job real --out-dir /tmp/cel-example/raw \
  --ce-adj Connectome/ce_adj.npy --ce-ei "" \
  --unknown-sign-policy random_unknown_sign_matched \
  --ce-unknown-sign-weights Connectome/ce_unknown_sign_weights.npy \
  --rho-values 1.05 --neuron-biases 0.0 --n-repeats 1 \
  --seed 12345 --src-tag cpu-example --cuda -1

python util/merge.py \
  --glob-variants '/tmp/cel-example/raw/invariance_variants.csv' \
  --glob-shuf '/tmp/cel-example/no-shuffle-input/*.csv' \
  --out-dir /tmp/cel-example/merged

python graph_hist.py \
  --combined /tmp/cel-example/merged/combined.ALL.csv \
  --out-dir /tmp/cel-example/figures
```

The empty EI path is intentional for edge-signed arrays. This run retains the
three leak rates and four input scales at one spectral radius and bias value.
Add `--rank-only` for a faster KR/GR check; its output leaves MC/IPC blank.
Use fresh output directories: full runs can replace a raw experiment CSV, while
rank-only runs patch existing rank columns. This small example is a pipeline
check; production comparisons use many seeded repeats.

Installed command aliases are `cel-res-sweep`, `cel-res-plot`, and
`cel-res-smoke`. The original Python commands and imports remain supported.
Use `python inv_arc_test.py --help` and `python graph_hist.py --help` for options.

## Architecture

| Location | Responsibility |
|---|---|
| `cel_res_hypr/connectome.py` | Processed connectome loading and unknown-sign preprocessing |
| `cel_res_hypr/networks/` | Directed topology, weight/sign controls, construction, normalization |
| `network_stats/dynamics.py`, `streams.py` | Reservoir update and KR/GR input protocols |
| `network_stats/stats.py`, `stats_util.py`, `run_one.py` | Readouts, MC/IPC/KR/GR, evaluation orchestration |
| `cel_res_hypr/experiments/` | Simulation settings, sampling/grids, variants, evaluation, CSV persistence |
| `cel_res_hypr/analysis/` | Repeat grouping, hyperparameter averaging, dispersion and mean tables |
| `cel_res_hypr/plotting/` | Figure families, styles, table statistics, rendering, and plotting CLI parsing |
| `inv_arc_test.py`, `reservoir_variants.py`, `graph_hist.py`, `util/` | Compatible entry points and utility exports |
| `tools/`, `tests/`, `network_stats/test_stats.py` | Reproducibility commands and deterministic CPU tests |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes interfaces, shapes,
equations, controls, and grouping. [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
provides experiment, seed, environment, and figure instructions.

## Production experiments and manuscript figures

The committed launchers encode the production settings. Review cluster-specific
resources and output paths before submission:

```bash
sbatch r_data.sbatch
```

| Launchers | Experiment |
|---|---|
| `r_data.sbatch` | Main architecture comparison |
| `r_data_shuf.sbatch`, `r_data_shuf_baseline_rho.sbatch` | Own-radius and family-baseline-radius topology/weight controls |
| `r_data_frac.sbatch`, `r_sign_norm_ablation.sbatch` | Negative-edge-fraction sweeps and normalization ablation |
| `r_triad_sign_fraction.sbatch` | Triad activity across sign fractions |
| `r_mc_delay.sbatch`, `r_ipc_delay.sbatch`, `r_ipc_order.sbatch`, `r_ipc_methodology.sbatch` | Capacity methodology checks |
| `r_gr_common_tail.sbatch` | Fixed-prefix/shared-continuation GR sensitivity |

The main launchers partition 200 seeded repeats across 20 array tasks. Rank-only
reruns preserve existing MC/IPC. `Connectome/` contains processed inputs;
`new_cel.xls` and `old_cel.xls` retain source spreadsheets. `final_results/` holds
curated results and manuscript figures. Commit `1b2344a` removed the obsolete
`final_results_GRKR_200__relative_svd_rank/`, `final_results_GRKR_erank/`, and
`final_results_KR__1__relative_svd_rank/` trees (345 CSVs and eight PNGs).
Those historical versions remain recoverable from Git history. `logs/` and
`depreciated/` preserve historical execution evidence and older launchers. `Parameter_justification/`,
`metric_figures/`, and `architecture_variant_figures/` contain specialized workflows.

For exact manuscript inputs and plotting commands, use
[GRAPHING_README.md](GRAPHING_README.md). Prefer its rank-updated `_erank` CSVs;
do not assume similarly named historical result directories are interchangeable.
The refactor through `89466f0` preserved all archived data and figures; the later
cleanup left the curated `final_results/`, connectome inputs, and implementation
unchanged. The local `thesis_tex/` manuscript checkout is maintained separately,
ignored by this repository, and left in place with its contents untouched. It
is not included in a clone of this research repository.

The [changelog](CHANGELOG.md), [iteration log](docs/REFACTOR_LOG.md), and
[final Critic report](docs/CRITIC_REPORT.md) record changes, validation, and
remaining limitations.
