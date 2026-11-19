# Pipeline audit: unified static architecture sweep

The new MC/NARMA10 pipeline is implemented and passes local checks. The audit
found a substantive NARMA task-definition discrepancy in the archived generator.
The new experiment has a separate protocol identifier and uses conventional
NARMA10. Actual CUDA verification remains an enforced Orca preflight step;
no production array was submitted from this machine.

## Findings that matter

**Archived NARMA is a different recurrence.** ReservoirPy 0.4.2 documents

\[
y_{t+1}=.3y_t+.05y_t\sum_{i=0}^{9}y_{t-i}+1.5u_{t-9}u_t+.1,
\]

but its implementation uses `sum(y[t-10:t])` and `u[t-10]*u[t]`.
The function source hash exactly matches the generator recorded in run 271800.
A direct numerical check gives zero residual against the implemented expression
and a maximum residual of 0.299 against the documented expression in a seeded
200-step example. This is a lag/history discrepancy, not evidence of target
leakage. The existing adapter aligns the consumed input to the next output of
that implemented recurrence. [ReservoirPy documentation](https://reservoirpy.readthedocs.io/en/stable/api/generated/reservoirpy.datasets.narma.html)

RCbench 1.0.10 was also checked. Its NARMA generator uses the same lag/history
indices and rescales input using extrema of the entire sequence. It was not
adopted. The new adapter reuses EchoTorch's NARMADataset **unchanged**, pinned to
commit `56c5d64084185c0bfe2b50229186cca20cc05181`, with the original license.
Its recurrence passes an independent residual check. No custom NARMA recurrence
was introduced. [Pinned upstream source](https://github.com/nschaetti/EchoTorch/blob/56c5d64084185c0bfe2b50229186cca20cc05181/echotorch/datasets/NARMADataset.py),
[RCbench repository](https://github.com/nanotechdave/RCbench)

**Conventional NARMA10 can diverge.** Screening all 100 planned default streams
found nonfinite targets for repeats 5, 12, 30 and 96. The explicit policy rejects
only nonfinite generated targets, before network evaluation, and tries seeds
`repeat_seed + 37000003 + 1009*attempt`, at most 100 attempts. One replacement
sufficed for each of these four repeats. Accepted/rejected seeds are stored;
all architectures and hyperparameters share the accepted stream. Finite large
errors are retained. This conditions the benchmark on usable finite target
sequences and must be disclosed, not hidden as ordinary seed sampling.

**Readout regularization and precision need control.** Archived MC uses an
absolute ridge penalty plus an added epsilon; changing state amplitude therefore
changes the effective penalty. That is a numerical/methodological dependence,
not proof that earlier performance differences were artifacts. The new protocol
uses float64 dynamics/readouts, a training-only common RMS scale for all state
features, and scikit-learn SVD Ridge. Separate validation data select penalties
for each MC delay and for NARMA. The readout is fit on training only; test targets
never affect selection. Multi-output Ridge shares one SVD over candidate
penalties and is tested against independent library fits. This removes arbitrary
global state-unit dependence; it does not remove real nonlinear input-drive
effects. [scikit-learn Ridge](https://sklearn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

A full-length precision check at 20% inhibitory edges, radius .95, leak 1,
inputs .01 and .5 compares float32 with float64 for CE and signed-weight-matched
ER. Scores and timings are in `precision_check.csv`. This is a one-realization
numerical check, not an architecture result or a GPU runtime estimate.

## Metric definitions and alignment

- The reservoir consumes `u[t]`, then predicts MC targets `u[t-d]` for delays
  1–60 or conventional NARMA target `y[t+1]`. Matrices use
  `W[source,target]`; row states update with `state @ W`.
- MC is held-out squared Pearson correlation summed over delays 1–30.
  Delays 31–60 and the extended sum are retained to expose truncation. Each
  estimated delay score lies in [0,1]; primary MC lies in [0,30]. This is a
  truncated finite-sample estimate, not the full theoretical capacity. The
  population bound by reservoir dimension requires the usual independent-input
  and readout assumptions. [Jaeger's memory-capacity report](https://www.ai.rug.nl/minds/uploads/STMEchoStatesTechRep.pdf)
- MC uses independent uniform [-1,1] input. NARMA uses uniform [0,.5], a 1,000-step
  target burn-in, and independent task streams. Constant predictions have MC
  zero; constant targets are rejected as undefined. No epsilon is added to
  correlations and no chance-floor subtraction is performed.
- NARMA NMSE = test MSE / test-target variance (population variance convention).
  Lower is better; it is unbounded above. NRMSE is its square root. A predictor
  using the training target mean is reported as a baseline. NMSE is not squared
  correlation or a universally bounded capacity.
- Both tasks use 500 washout, 1,500 training, 500 validation and 500 test steps.
  State continuity and earlier input history across split boundaries are
  intentional. Targets are never fed back. Only readout penalties are selected;
  the complete reservoir grid remains available for comparisons.

## Model controls and normalization

The verified input data have 297 nodes, 1,740 known-sign edges and 1,864
unknown-sign edges (3,604 completed edges). Masks are disjoint and loop-free.
All models impose randomized **edge** sign quotas using `floor(fraction*edges)`;
realized fractions and inhibitory magnitude fractions are saved. These are not
Dale-constrained signs or inferred biological E/I balances.

The ten models are specified in `network_stats/architecture_sweep.py`. Exact
signed-weight matching is verified for CE empirical versus CE within-sign
magnitude shuffle, degree rewiring, ER weighted and WS-like weighted controls.
CE/ER half-normal controls share the same signed magnitudes; CE/ER binary
controls have the same counts of each sign. Every family retains the same node
count. The removed-edge model retains original full-CE magnitude units but has
fewer edges; comparison with the full-density ER models is not a pure test of
unknown-edge removal. Shared-edge signs need not match the completed model
exactly at quota boundaries because its smaller mask has its own exact quota.

Directed degree rewiring uses NetworkX's established three-edge swap algorithm,
20 successful swaps per edge, checks both directed degree sequences, and retains
each source's signed outgoing-weight multiset. There is no silent partial-swap
fallback. The prior local helper makes a single pass with at most half as many
successful swaps as edges; the new experiment does not label that procedure a
fully mixed null. Even 20 swaps/edge is not proof of uniform sampling or mixing;
edge retention is recorded and connectivity is not constrained.
[NetworkX directed swaps](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.swap.directed_edge_swap.html)

ER is uniform directed G(n,m), sampled directly without edge-count repair.
WS is explicitly a directed WS-like ring/reconnection control, rewiring
probability .1 and random node relabeling, not textbook undirected WS.

Three scaling protocols are distinct:

| Protocol | Divisor for target radius | What changes across fractions |
|---|---|---|
| Own radius | Current radius of this family | Each multiplier changes |
| Fixed reference | This family's radius at CE's known-edge reference fraction | Multiplier is fixed for each target/family |
| Shared CE | Current empirical CE radius, applied to every family | Common multiplier changes with fraction |

Only common scaling preserves the exact signed-weight multiset after scaling.
The fixed-reference fraction is 422/1740 = 24.2529%, used as a scaling reference,
not an additional performance observation or biological prediction. Matrices
with zero/nilpotent radius fail rather than being silently replaced.

Actual scaled radii, leaky linearized radii, state variance, saturation and
finite-washout sensitivity to a second initial condition are recorded.
Spectral radius alone does not establish the echo-state property; a nonzero
initialization gap is a diagnostic, not an automatic reason to discard a row.
Changing several matrix properties together does not isolate an eigenmode
mechanism.

## Verification and execution

Local verification includes 20 unit/integration tests; graph, sign and weight
invariants; ideal delay-register recovery; constant/floor and high-variance
states; finite/nonfinite validation; deterministic stream rejection; independent
ridge comparisons; batch isolation; immutable hashes; tamper detection;
complete-grid merging; frozen execution outside the repository; and a fake-Slurm
submission test checking dependency and array arguments.

A separate short-trajectory CPU integration run exercised all ten models,
three normalization protocols, three fractions, eight settings and two
realizations: **1,440 paired evaluations**. Merge and all PNG/CSV analysis
outputs completed. Its short trajectories serve pipeline validation only;
they do not support scientific performance claims. They remain under `/tmp`,
not among archived scientific results.

The actual GPU is unavailable locally. On Orca, a GPU preflight reruns the tests,
compares double-precision CUDA recurrence and complete metric scoring against
CPU references, screens every task stream, and evaluates representative full
length conditions across all selected families/normalizations. Only a passed
preflight releases the GPU array. Merge/analysis depend on the whole array
succeeding. Frozen source/data hashes, package versions, cached spectra/streams
and per-shard checksums support reproducible resumption. Bootstrap uncertainty
resamples paired whole realizations; configurations are a fixed grid.

Launch instructions: `ARCHITECTURE_SWEEP.md`. Defaults are 100 realizations,
24 grid settings, six fractions, ten models and three protocols: 432,000 paired
MC/NARMA evaluations. Wall-time requests are scheduling limits; actual GPU runtime
is not yet measured. Existing task code and archived results were not overwritten.
The archived task/readout files still match their recorded source hashes.
