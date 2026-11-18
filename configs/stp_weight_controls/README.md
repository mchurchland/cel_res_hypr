# Continuous STP and weighted-control cluster sweep

This is a new follow-up experiment. Existing scripts, data and pilot results are
not changed. It extends the continuous-STP/tanh model used in the local pilots;
it does **not** implement c302, conductance-based neurons, or calibrated worm STP.

## Run on Orca

Use the existing GitHub checkout on Orca. From its repository root, use the
existing GPU-capable `cel` environment. Before submitting,
ensure it has the new dependencies (the jobs never install packages):

```bash
conda run -n cel python -m pip install -r requirements-stp-controls-cuda.txt
conda run -n cel python stp_weight_controls.py --backend cuda --dry-run
bash tools/submit_stp_weight_controls.sh
```

The helper creates `logs/`, submits a ten-task **GPU array** (one GPU per task),
and submits a dependent merge/analysis job that starts only if the whole array
succeeds. One process per GPU batches independent input/STP configurations.
Reservoir/STP dynamics run on CUDA; the existing readout fits remain on CPU.
The subsequent lightweight merge/PNG-analysis job uses a CPU and no GPU.
Default resources are requests, not a measured runtime forecast. No live GPU was
available on the development machine; execute the CUDA startup check inside a
GPU allocation before the full sweep:

```bash
srun --partition=normal --gres=gpu:1 --cpus-per-task=1 --mem=4G --time=00:10:00 \
  conda run -n cel python stp_weight_controls.py --check-cuda
```

The runner also compiles and runs that check automatically before writing results.
Unavailable CUDA fails explicitly; there is no CPU fallback. Never enable
`NUMBA_ENABLE_CUDASIM` in a production job. The Slurm runner rejects it.
If the cluster already supplies a compatible CUDA toolkit/Numba CUDA backend,
its existing installation can be used instead of reinstalling it. Installation
reference: https://nvidia.github.io/numba-cuda/user/installation.html

Defaults: **50 paired computational realizations, 168,000 MC/NARMA condition
pairs (336,000 task trajectories)**. This is a substantial follow-up, not an
eight-seed pilot. To start smaller:

```bash
TOTAL_REPEATS=20 bash tools/submit_stp_weight_controls.sh
```

`TOTAL_REPEATS`, `BASE_SEED`, `WASHOUT`, `TRAIN_LENGTH`, `VALIDATION_LENGTH`,
`TEST_LENGTH`, `MC_MAX_DELAY`, `NARMA_BURNIN`, `CONDA_ENV`, `PYTHON_BIN`, `OUTROOT`
`ANALYSIS_DIR`, `CUDA_DEVICE`, and `GPU_BATCH_SIZE` can be passed as environment
variables. Default GPU batch size is 64; reduce it if device memory is tight.
The helper exports a
unique `RUN_ID`. It prints the exact identifier and resume command. Repeat all
nondefault settings when resuming. A changed data/code hash, setting, or package
version is rejected; choose a new run ID for a new experiment.

To request fewer simultaneous array tasks, for example:

```bash
TOTAL_REPEATS=20 bash tools/submit_stp_weight_controls.sh --array=0-9%3
```

To submit the sweep directly:

```bash
mkdir -p logs
sbatch r_stp_weight_controls.sbatch
```

Its default results directory then uses the Slurm array job ID. After completion:

```bash
RUN_ID=YOUR_ARRAY_JOB_ID sbatch r_stp_weight_controls_analyze.sbatch
```

The helper defaults to 50 repeats because the eight-seed local pilots were
preliminary. This is not a power calculation; increase replication if the paired
intervals remain too wide. Do not count configurations as independent samples.
The primary uncertainty is over computational realizations/sign assignments,
not independently sampled biological connectomes.

## Experimental design

- E/I edge percentages: **0, 10, 20, 30, 40, 50**. No mirror symmetry is assumed
  or used to impute the omitted 60–100% sweep. Fractions are achieved to rounding
  at the edge-count level and actual fractions are recorded.
- MC and NARMA10 only. Same existing MC scoring and ReservoirPy **0.4.2** NARMA10
  generation, alignment, training, validation-only ridge selection and test split.
- Input scales **0.1 and 0.5**. Full input and sensory CE/random ER subsets are
  separate conditions. The 78 sensory labels are copied unchanged from the prior
  frozen WormAtlas-derived annotation table into `sensory_neurons.csv` here;
  identities are checked against `Connectome/ce_nodes.txt`. Restricted inputs have
  the full input vector's L2 norm. Every CE variant uses the same sensory nodes;
  every ER variant uses the same random subset. Inputs are paired across families.
- No spectral-radius, leak-rate, or bias grid. Initial radius reference **0.95**,
  leak **1**, bias **0**. No dynamic renormalization after adding STP.
- Unknown edges are restored using the known-edge inhibitory fraction, as in the
  existing crossover experiment. Every edge's sign is then randomized for the
  E/I sweep. This is not a fixed physiological sign map or a Dale-neuron model.

### Weighted controls

All seven models have the same node and edge counts. CE variants keep the CE
edge set; ER variants share a loop-free matched-edge-count random edge set.

| Family | Topology | Magnitudes and signs |
|---|---|---|
| `ce_empirical` | CE | Original CE magnitudes; randomized E/I edge signs |
| `ce_shuffled` | CE | Exact CE magnitudes permuted over CE edges; same CE sign mask |
| `ce_binary` | CE | Equal nonzero magnitudes; same CE sign mask |
| `er_empirical` | ER | Exact CE magnitude multiset, permuted without replacement |
| `er_signed_empirical` | ER | Exact signed CE weight multiset, matched separately at each fraction |
| `er_halfnormal` | ER | Absolute Gaussian draws, then assign signs for each E/I fraction |
| `er_binary` | ER | Equal nonzero magnitudes; legacy sign-sweep control type |

Raw mean nonzero magnitude is set to one in each family. ER half-normal is
explicitly a Gaussian **magnitude** control: a directly signed zero-mean Gaussian
would not support an independently prescribed E/I fraction. The exact multiset
controls do not sample with replacement. Magnitude matching does not preserve
node strengths or CE degree sequences.

Unsigned placements, edge sets and nested sign permutations are held fixed
across E/I fractions. The exception is `er_signed_empirical`: magnitudes are
reassigned separately at each fraction to match CE's positive AND negative weight
multisets. This also matches total inhibitory/excitatory weight before scaling.
The actual matrices are cached so the distinction is auditable.

### Two initial normalization modes, not a radius grid

1. `own_radius`: normalize each signed matrix to initial radius 0.95. This can
   change absolute weight distributions differently across families.
2. `shared_ce_scale`: at each fraction, compute the CE empirical multiplier that
   gives radius 0.95 and apply it to every family at that fraction. This preserves
   exact magnitude/signed-weight matching where specified. Other families need
   not end up at radius 0.95. This is **not** fixed-reference scaling across E/I:
   the shared multiplier can vary from one E/I fraction to the next.

The complete sign-sweep graphs are recurrent chemical-weight abstractions;
electrical gap junctions and biophysical voltages are not added in this experiment.

### STP conditions

Static baseline plus nine plastic conditions: three coupled parameter sets,
each run as all facilitating-dominant, all depressing-dominant, and approximately
50:50 mixed presynaptic populations. All plastic neurons have BOTH utilization
and depletion. The same randomized node assignment is used across all families,
fractions, inputs and tasks in each realization, independently of signs.

| Set | Facilitating `(U, tau_f, tau_d)` | Depressing `(U, tau_f, tau_d)` |
|---|---|---|
| fast | `(0.10, 30, 3)` | `(0.35, 3, 30)` |
| reference | `(0.15, 75, 5)` | `(0.45, 5, 75)` |
| slow | `(0.20, 150, 10)` | `(0.55, 10, 150)` |

These are generic parameter-sensitivity settings around the local pilot, not
measured biological ranges. Several parameters change together: do not infer
individual U/time-constant effects from this three-set design. Time units are
input samples, not calibrated milliseconds. STP is driven by
`abs(previous signed tanh activity)`, uses output gain `u*x/U`, and is integrated
with four RK4 substeps per reservoir step. The reference settings reproduce the
previous STP backend on synthetic compatibility tests. Only neuronal states enter
the readout; synaptic states are not directly exposed to it.

The Python CLI additionally supports, for example,
`--mixed-fractions 0.25 0.5 0.75`; this is not enabled in the default sbatch grid.

### Optional gain control

Omitted by default, following the user's stated priority. To include it:

```bash
GAIN_CONTROL=1 bash tools/submit_stp_weight_controls.sh
```

For each plastic condition and each task, average each neuron's gain over its
dynamic **training segment only**, freeze those coefficients, reset the neuronal
state, and rerun the same input with static transmission. Readouts are refitted.
Validation/test data never set the frozen coefficients. This is an additional
diagnostic, not a unique causal isolation of history; it does not match realized
activity in the subsequent static trajectory. It adds nine conditions per grid
point, for **319,200 MC/NARMA pairs** at 50 repeats.

## Outputs and what to send back

`results/stp_weight_controls/RUN_ID/` contains:

- `manifest.json`: exact settings, source/code hashes and package versions.
- `source_snapshot/`: copies of runner dependencies and input files for provenance.
- `repeats/repeat_XXXX/*.csv`: immutable, resumable result shards with task scores,
  MC delay curves, STP diagnostics, actual E/I fractions and normalization factors.
- `repeats/repeat_XXXX/inputs_and_networks.npz`: actual task inputs/targets, unsigned
  networks, sign masks, STP assignment, and input weights needed to reconstruct
  each condition. This does not cache every full state trajectory.
- `stp_weight_controls.ALL.csv` and `merge_audit.json`: complete validated results.

`analysis/stp_weight_controls/RUN_ID/` contains PNG figures, paired-bootstrap
comparison tables, changes from static, and `REPORT.md`. Both desirable and
undesirable STP outcomes are retained; no parameter setting is selected using
test scores. Intervals resample paired computational realizations, not individual
hyperparameter configurations. They are pointwise, not multiplicity-adjusted.

Send back the merged CSV, `manifest.json`, `merge_audit.json`, and preferably the
entire analysis directory. Keep the realization NPZs on the cluster for auditing.
If the merge fails, send the job logs and retain the shards: completed work resumes.

## Local/cluster smoke test

The full sweep is not launched locally. A short actual-connectome compatibility
check can be run before submitting, using a new output path:

```bash
conda run -n cel python stp_weight_controls.py \
  --out-dir /tmp/stp_controls_smoke --n-repeats 1 \
  --backend cuda --workers 1 \
  --ei-percent 20 --input-scales .1 --input-schemes full \
  --washout 100 --train-length 200 --validation-length 100 --test-length 100 \
  --mc-max-delay 10 --gain-control
```

Short segments and a single seed make this a software/runtime check, not an
experiment from which to draw performance conclusions.

```bash
conda run -n cel python -m unittest discover -s tests -p test_stp_weight_controls.py -v
```

Inside a GPU allocation, validate the actual CUDA kernel:

```bash
conda run -n cel python -m unittest discover -s tests -p test_stp_weight_controls_cuda.py -v
```

For local validation without a GPU, use `NUMBA_ENABLE_CUDASIM=1` with that test
command. Simulator agreement is not a hardware compilation/performance test.
CUDA and CPU results have separate protocol identities; do not pool them as
bitwise-identical realizations. Per-realization `gpu_devices/*.json` records
hardware for each attempt, including resumed jobs on a different GPU model.
