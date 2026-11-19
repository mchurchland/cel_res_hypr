# Full-range IPC backfill validation

The full launcher covers IPC at all eleven 0–100% fractions, copies validated MC/NARMA scores for 0–50% from run 271846, and computes fresh MC/NARMA scores only for 60–100%.

35 unit/integration tests passed. Tests specifically prove that the backfill runs only the MC trajectory, never refits the existing MC/NARMA readouts, preserves their score values, verifies cached parent input streams and raw matrices, and streams a mixed reused/new dataset through immutable merge and resume.

A separate CPU check used the actual stored run 271846, all ten models and all three normalizations at 0%, 50%, 100%, one stable setting and production sequence lengths. All 90 rows completed: 60 inherited MC/NARMA scores with new IPC, 30 all-new scores. The recorded CSV is a validation sample, not a grid-level scientific result.

GPU preflight on Orca remains mandatory and includes fresh 50% overlap and the reuse branch. No production jobs were submitted locally. See ARCHITECTURE_IPC_EXTENSION.md for launch details.
