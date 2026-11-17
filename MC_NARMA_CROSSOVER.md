# Focused E/I crossover sweep: MC and NARMA10

This is a separate experiment. The existing simulation entry points, launchers,
and `final_results/` are unchanged. The launcher prepares new outputs under
`results/crossover_mc_narma10/`; it has not been submitted.

## Design

| Setting | Values |
| --- | --- |
| Requested negative-edge fraction | 20%, 24%, 28%, 32%, 36%, 40% |
| Models | CE with unknown-sign edges restored; matched ER |
| Spectral-radius target | 0.6, 0.95, 1.05 |
| Leak | 0.6, 1.0 |
| Input scale | 0.1, 0.5, 1.0, 1.5 |
| Reservoir bias | Fixed at zero; absent from the sweep |
| Normalization | Own-radius and fixed-reference |
| Metrics | MC and NARMA10 validation/test error |
| Repeats | 200, partitioned across 20 Slurm array tasks |

There are **24 hyperparameter configurations per model/fraction/normalization**,
576 evaluated configurations per repeat, and **115,200 rows** in the full run.
The intermediate 0.8 radius and 0.8 leak settings are omitted from this reduced grid.
Each row contains both tasks. IPC, KR and GR are never evaluated. There is no
automatically inserted biological-reference fraction.

## Reused implementations

- Network construction: existing `util.util.build_reservoir` sign-sweep models.
- Reservoir dynamics: existing `network_stats.run_one.run_reservoir_states`.
- MC: existing `network_stats.stats.compute_MC`, including its delay alignment,
  squared-correlation score and ridge epsilon. The readout uses float64;
  reservoir dynamics remain float32.
- NARMA: **`reservoirpy.datasets.narma` from ReservoirPy 0.4.2**, called directly
  with `order=10, a1=0.3, a2=0.05, b=1.5, c=0.1`. There is no locally implemented
  or copied NARMA recurrence. Upstream implementation and lag conventions are
  retained unchanged. [ReservoirPy documentation](https://reservoirpy.readthedocs.io/en/latest/api/generated/reservoirpy.datasets.narma.html).
- NARMA readout: **scikit-learn `Ridge`**, with its standard fitted intercept,
  float64 arrays and Cholesky solver. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

CE is the existing `random_unknown_sign_matched` variant: known signed edges
are retained while unknown edges are restored at the known edges' negative-sign
proportion (currently 0.2425287356), randomized once per repeat. The E/I sweep
then reassigns signs across **all** nonzero edges, preserving CE magnitudes.
It does not vary only the unknown edges or preserve known signs during the sweep.

ER is the existing **binary-weight** sign-sweep control, matched to the completed
CE edge count (currently 3,604 directed edges on 297 nodes). It is not the
Gaussian-weight ER variant from the main architecture experiment. Consequently,
the CE–ER contrast does not isolate topology from weight magnitudes.

Own-radius scaling uses `rho_target / raw_rho`. Fixed-reference scaling uses
`rho_target / ref_rho`: CE's completed signed matrix is its reference; ER uses
the same ER realization signed at the completed CE negative-edge fraction.
The reference is fixed across swept fractions within each realization.
The applied factor, raw/reference/post radii and achieved negative-edge
fraction are stored. The existing sign sampler floors the requested count,
so achieved percentages can differ slightly from the requested percentages.

## Task protocol

MC uses an independent uniform [-1, 1] input: 500 washout, 1,500 training and
500 test steps; delays 1–30; fixed ridge alpha 1e-4 plus the existing 1e-6
solver epsilon. The input is shared across all conditions within each repeat.

NARMA uses the package's uniform [0, 0.5] input without recentering. Generate
1,000 extra target burn-in steps, discard them, then simulate 500 reservoir
washout, 1,500 training, 500 validation and 500 test steps. ReservoirPy returns
inputs of length n+10 and targets `y[10:]` of length n. Pair those targets with
`u[9:-1]`: after consuming input at time t, predict the next output at t+1.
Burn-in is removed from both aligned arrays. Nonfinite targets fail the run;
no clipping or silent resampling is performed.
Zero reservoir bias means no added constant bias; the NARMA input itself has
a positive mean, as specified by the task.

The NARMA readout uses reservoir states only, with no direct input features,
target feedback or state standardization. Reservoir bias is zero; the fitted
readout intercept is retained. Ridge alpha is selected by validation NMSE
from {1e-8, 1e-6, 1e-4, 1e-2, 1}. Each candidate fits training data only; the
selected model is evaluated on the untouched test segment without refitting.
NMSE is scikit-learn MSE divided by population variance of the corresponding
target segment. **Lower NARMA error is better; higher MC is better.** Test NMSE,
NRMSE, RMSE, validation NMSE, chosen alpha and a training-mean baseline are saved.

Each condition uses identical numerical seeds and task signals within a repeat.
The graph/sign draw and input-weight draw remain fixed across hyperparameters;
the existing nested sign permutation couples neighboring fractions. Separate
seed offsets generate MC input, NARMA input and unknown-edge signs. Seeds are
computational realizations, not independent worms. Default repeat seed is
`20261117 + repeat_id * 100000`; network construction uses that seed plus one.

These runs establish a consistent new protocol. Their signal generation and
readout precision differ from historical runs, so retain the new 20% and 40%
anchors rather than mixing old and new observations. Interpret the crossover
as conditional on the tested input/radius/leak distribution and zero bias.
The requested 20–40% interval need not contain every input-specific or
fixed-reference crossover; absence of a crossing is a valid result.

## Install and inspect

From the repository root, in the existing `cel` environment:

```bash
conda run -n cel python -m pip install -r requirements-crossover.txt
conda run -n cel python mc_narma_crossover.py --dry-run --repeat-split 20 --repeat-rank 0
```

The dry-run validates inputs/dependencies and prints counts without running
reservoir trajectories or creating outputs. `requirements-crossover.txt` adds
only the pinned ReservoirPy dependency to the repository's existing environment.

## Submit

```bash
mkdir -p logs
sbatch r_mc_narma_crossover.sbatch
```

Run from the repository root. The launcher mirrors the existing `normal`
partition, one GPU, eight CPU cores, 32 GB RAM and 24-hour limit; GPU wall time
has not been benchmarked locally. Default results:

```text
results/crossover_mc_narma10/<array_job_id>/chunk_<task_id>/
    manifest.json
    repeat_0000__cel_matched__ei_20.csv
    repeat_0000__matched_er__ei_20.csv
    ...
```

Environment overrides include `CONDA_ENV`, `PYTHON_BIN`, `OUTROOT`, `RUN_ID`,
`TOTAL_REPEATS`, `BASE_SEED`, `CE_ADJ`, `CE_UNKNOWN_SIGN_WEIGHTS` and `CUDA_DEVICE`.
The Slurm array must be contiguous and start at zero. To resume an interrupted
run, use the same code, settings and array size, specifying its original ID:

```bash
sbatch --export=ALL,RUN_ID=12345 r_mc_narma_crossover.sbatch
```

Complete per-repeat/model/fraction CSVs are published atomically. Resume checks
the manifest, condition identities, grid completeness and finite scores, skips
finished files and recomputes an unfinished shard. It never overwrites a result.
Source/data hashes, package versions, upstream generator hash, seed protocol,
task settings and device information are recorded in each chunk manifest.

Merge after all array tasks finish:

```bash
conda run -n cel python mc_narma_crossover.py --merge --out-dir results/crossover_mc_narma10/12345
```

This writes `mc_narma_crossover.ALL.csv` and rejects incomplete, duplicate or
incompatible runs. It refuses to overwrite an existing merged file.
For subsequent tuning comparisons, select configurations using validation
scores, keeping test scores for reporting; a grid-averaged comparison is a
separate analysis.

## Validation

```bash
python -m unittest discover -s tests -p test_mc_narma_crossover.py -v
bash -n r_mc_narma_crossover.sbatch
```

Tests cover upstream target alignment, reproducible signals, test-label isolation
from ridge selection, fitted intercepts, invalid values, unknown-edge restoration,
matched edge counts, paired signs/input weights, both normalizations, disjoint
repeat partitions, dry-run counts, actual small runs, resume, merge and overwrite
protection. Full-length task evaluation was also checked locally on the actual
297-node CE and ER inputs at one configuration under both normalizations.
