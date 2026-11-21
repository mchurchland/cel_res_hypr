# ER weight-distribution submission validation

Submission file: `r_er_weight_distribution.sbatch`. No production jobs were
submitted during local validation.

- 8 new scientific/pipeline tests passed: finite-vector moments and positivity,
  upper-tail caps and mean restoration, both heterogeneity levels, paired masks
  and signs, scaling invariants, archived-score reuse, full small-run merge,
  strict resume, analysis estimands, and the three dependent Slurm submissions.
- 43 existing architecture/AUC tests passed after adding the optional explicit
  scaling and validation hooks to the existing scorer. The default scoring path
  is unchanged; task definitions and readout source files were not modified.
- Shell syntax, Python compilation and whitespace checks passed.
- The real default plan against run 271871 validated source/parent checksums:
  792,000 total rows; 158,400 direct parent baseline copies; 79,200 planned
  identical shared/Frobenius copies; 1,108,800 newly simulated trajectories.
- Realization zero: all 10 weight conditions passed topology/sign checks at all
  11 requested E/I fractions and the fixed-reference cache position. Gamma and
  lognormal realized moments match the original empirical vector at every point.
- Full-length CPU probes passed for gamma and cap95 under all three scaling
  protocols. A freshly scored empirical baseline at 20% inhibition reproduced
  the archived MC and NARMA scores to absolute differences below 9e-12.
- An end-to-end synthetic 12-node run validated/merged 180 rows and invoked the
  frozen analysis entry point from outside the repository. All 16 PNG outputs
  and their checksums passed; no PDFs were generated. Synthetic figures were
  temporary and were not published as experimental results.
- CUDA is unavailable locally. Mandatory GPU preflight checks recurrence and
  CPU/GPU score parity, fresh versus inherited baseline scores, every task
  stream, all models and all scaling protocols before the production array.

`real_parent_checks.json` records exact network diagnostics and CPU probes.
`plot_smoke.json`, `new_tests.log` and `existing_tests.log` retain verification
results. `default_plan` is a local validation snapshot, not a completed sweep.
