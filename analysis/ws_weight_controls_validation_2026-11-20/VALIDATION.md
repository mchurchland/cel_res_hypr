# WS binary and half-normal submission validation

The submission file is `r_ws_binary_halfnormal.sbatch`. No cluster jobs were
submitted during local validation.

- All 40 architecture pipeline tests passed, including five tests for the two
  new models, exact pairing, combined analysis, and submission arguments.
- Shell syntax and tracked-code whitespace checks passed.
- Preparing the default run against completed experiment 271871 passed all
  source, task, grid, IPC-basis and merged-data checksum checks: 158,400 new
  rows and 316,800 trajectories, only two new simulated families.
- Realization zero: both new models preserve all 3,604 WS edges and their signs
  at all eleven E/I fractions and the fixed-reference fraction. Their exact
  signed-weight multisets match the corresponding parent CE/ER controls.
- Full-length CPU scoring succeeded for both new models at 20% inhibitory
  edges, own-radius scaling, radius target 0.6, leak 1 and input scale 0.1.
- Recomputing the parent CE reference condition gave absolute differences
  below 1.2e-10 for MC, NARMA10 NMSE and IPC. These checks establish numerical
  compatibility for this condition; they are not performance comparisons.
- CUDA is unavailable locally. The submission retains mandatory GPU preflight
  and CPU/GPU numerical checks before launching the simulation array.

`real_parent_checks.json` records the actual cache checks and local scores;
`unit_tests.log` records the test run. `default_plan` is a local validation
snapshot, not a completed sweep. Production output uses a separate run folder.
The signed IPC estimator and readout are unchanged, including their previously
identified susceptibility to severe held-out reconstruction errors.
