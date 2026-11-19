# IPC upper-half extension validation

30 unit/integration tests passed, including complete polynomial basis, cross-delay alignment, orthonormality, ideal recovery, negative held-out estimates, constant-state baseline, preserved MC/NARMA scores, signed IPC win direction, parent-setting compatibility, and sbatch forwarding.

A short-trajectory CPU run exercised all ten models, three normalizations, 50/80/100% inhibition, two realizations and the default 285-target IPC basis. All 180 rows merged and the analysis produced MC/NARMA/IPC outputs. These are implementation checks, not new scientific results.

Actual GPU checks and the parent 50% overlap check run in the Orca preflight before the array can start. No production jobs were submitted locally. See ARCHITECTURE_IPC_EXTENSION.md for launch details and estimator scope.
