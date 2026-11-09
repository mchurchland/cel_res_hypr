# Final independent Critic report

**Decision: ACCEPT.** The independent Critic reviewed the final code, tests,
package boundaries, and documentation. All five iterations satisfy the
behavior-preserving refactor criteria within the validation scope below. No
unresolved software-refactor blocker remains. Historical scientific limitations
are explicitly retained and do not constitute approval of a scientific correction.

## Accepted changes and evidence

| Iteration | Decision | Independent evidence |
|---|---|---|
| Audit/baseline | ACCEPT | 16 tests; 54 original rows and schema verified |
| Connectome/network construction | ACCEPT | 32 tests; all 31 moved function ASTs preserved |
| Experiments/dynamics/CSV | ACCEPT after revision | 47 tests; dispatch/seed review; descriptions corrected |
| Plotting/aggregation | ACCEPT | 55 tests; all function exports retained; 23 artifacts identical |
| Package/documentation/final verification | ACCEPT | 55 tests, original unittest command, installed baseline, plotting driver, protected checksums |

The Actor's [iteration log](REFACTOR_LOG.md) records each change, test evidence,
Critic findings, and disposition. The Critic inspected actual code and tests
rather than relying on that log. Compatibility facades preserve the original
commands/imports. There are clear module boundaries for loading, graph controls,
dynamics, metrics, experiment settings, CSV storage, aggregation, and plotting.

Final verification evidence independently checked by the Critic:

- 55 deterministic CPU tests, plus the original 15-test unittest command.
- Exact equality of the original 54-row MC/IPC/KR/GR CPU baseline, including
  after isolated wheel installation outside the source tree.
- 23 original PNG/CSV/LaTeX artifacts matched byte-for-byte across eight plotting
  paths; a committed validation driver reproduces their hashes.
- A processed-connectome example executed all four metrics for 12 configurations,
  then merged its results and generated a histogram in temporary directories.
- Wheel build, isolated console-command checks, and SLURM shell syntax checks.
- SHA-256 verification of 8,651 archived data/result files and 607 manuscript
  files: none changed or disappeared; no manuscript files were added.

No metric algorithm, random-draw ordering, production grid, published result,
CSV contract, or statistical grouping was changed. The optional weight-performance
plot's disabled exporter was restored, correcting a software crash without
changing plotted data. Documentation changes correct solver/SVD/sign definitions.

## Remaining historical issues

| Severity | Issue | Disposition / proposed separate correction |
|---|---|---|
| High | Main CLI validates but ignores `--unknown-sign-inhibitory-frac` | Regression retained. Forward the parameter only in a separately approved scientific correction with labeled comparisons. |
| Medium | ER sign-sweep graph generation fixes probability at 0.1 | Regression retained. Parameterizing it changes seeded realizations even at matched edge count. |
| Medium | Preactivation trajectory path uses default-dtype buffers | Float64 failure retained/tested; production float32 preserved. A dtype fix needs comparison of all affected paths. |
| Medium | Unused `_shuffle_ce_weights_except_1` shuffles rows through incorrect indexing | Historical output retained/tested and helper explicitly documented as unsuitable for production edge shuffling. |
| Medium | Absolute rank cutoff can retain float32 centering residuals | Regression retained. A scale-aware cutoff would change historical rank values and needs a protocol decision. |
| Medium | Optional MC/GR scatter uses IPC background points and an obsolete Δstate label | Data behavior retained/tested. Correct both plotting choices separately; canonical manuscript paths do not use this optional scatter. |
| Low | Array rank aliases are ambiguous for one-based callers | Actual zero-based precedence documented/tested; production launchers use zero-based ranks. |

Restricted single-delay IPC and Shannon effective rank are deliberate historical
metric definitions, not claims to evaluate a full mixed-delay IPC basis or an
integer matrix rank. They are now documented accurately. Methodology scripts and
specialized rendering paths retain justified local calculations to avoid a
numerically risky generalized abstraction.

## Validation limits

Production GPU/SLURM sweeps and every full-resolution manuscript figure were not
rerun. The plotting comparison caps saved DPI at 50; hash equality is tied to the
recorded plotting environment. Dependency constraints cover direct packages,
not a complete transitive/hardware lock. Alternative Python versions are allowed
by metadata but have not received the recorded Python 3.13.9 validation.

The archived `depreciated/` launchers retain historical paths/settings and are
not presented as current reproduction recipes. Some figure scripts retain
manuscript-specific paths; `GRAPHING_README.md` documents their canonical use.
The `thesis_tex/` manuscript tree remains untracked, unmoved, and content-identical.

The accepted refactor preserves these historical limitations explicitly. Any
scientific correction requires its own protocol decision and labeled numerical
comparison; this acceptance does not authorize one.
