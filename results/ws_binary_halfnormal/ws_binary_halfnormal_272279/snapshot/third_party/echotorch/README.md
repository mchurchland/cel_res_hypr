# Reused EchoTorch NARMA generator

`NARMADataset.py` is copied **without modification** from Nils Schaetti's EchoTorch:

- Repository: https://github.com/nschaetti/EchoTorch
- Commit: `56c5d64084185c0bfe2b50229186cca20cc05181`
- File: `echotorch/datasets/NARMADataset.py`
- SHA256: `07a9a4edf2ae6b7b70a2e97590172159b6fac516eab94462293747dbdf71256f`
- Retrieved: 2026-11-19.

The file's original copyright notice and GPL version 2 header are retained;
`COPYING` supplies that license. This is reused upstream code, not a local NARMA
recurrence. The adapter in `network_stats/audited_temporal_tasks.py` controls RNG,
precision, burn-in and one-step alignment outside this unmodified class.

For system_order=10, upstream uses the conventional ten-output history including
the current output and the input product u[t-9]*u[t]. The audit contrasts this
with ReservoirPy 0.4.2 and RCbench 1.0.10. Do not silently substitute generators:
their task definitions differ, even when all are labeled NARMA10.
