# Orca storage cleanup — 21 November 2026

The failure was a personal home-directory byte quota, not filesystem exhaustion or an inode quota. Job `274063_0` failed while flushing a result file with `OSError: [Errno 122] Disk quota exceeded`. No Slurm jobs were active during cleanup.

## Space recovered

| Quota measurement | Before | After |
|---|---:|---:|
| Home storage used (MiB, as reported by `quota -s`) | 101,017 | 82,771 |
| Soft quota | 92,160 | 92,160 |
| Hard limit | 102,400 | 102,400 |

Recovered **18,246 MiB = 17.82 GiB**. Remaining headroom is **9.17 GiB below the soft quota** and **19.17 GiB below the hard limit**.

Only disposable download caches were removed:

- `python3 -m pip cache purge`: 2,028 cached files removed successfully.
- `conda clean --tarballs --index-cache --logfiles --yes --json`: successful; removed 6,091,687,897 bytes of downloaded Conda archives plus the repodata index cache.

Installed environments, research results, experiment snapshots, parent-run caches, source code, and Git history were retained. The entire Conda package directory was not removed: it contains extracted packages that may share files with installed environments.

After cleanup, the `cel` environment successfully imported NumPy 2.2.6, SciPy 1.15.3, and PyTorch 2.8.0+cu128 on the login node. This checks imports, not GPU execution.

## Local backup coverage

The three completed merged result files below have local SHA-256 hashes matching their recorded Orca checksum sidecars:

| Run | SHA-256 |
|---|---|
| `architecture_mc_narma_271846` | `c5939609492cd4b7ba791d21579b3048e3d1ea8c2885692a01e7938b99cf6d8b` |
| `architecture_full_ipc_271871` | `f567d454822e0ed7afe4cf4d13ea2c67e44fb45fda2eb9532c52a017c07fa896` |
| `ws_binary_halfnormal_272279` | `742a240cd3a8dab2caaec4860b8168be28cd56de11b069606bb4e4bc141365ee` |

The WS merged CSV is in the nested local download directory `results/architecture_mc_narma_ipc_full/ws_binary_halfnormal/ws_binary_halfnormal_272279/`.

The new partial run `results/er_weight_distribution/er_weights_274061/` was not present locally at the inventory snapshot. It was retained on Orca. A size/path comparison is not a checksum verification of every historical shard; no scientific files were deleted on the strength of that comparison.

## Evidence

Read-only remote and local inventories, quota output, and local merged-file hash checks were captured in `/tmp/orca_results_inventory_2026-11-21.json`, `/tmp/local_results_inventory_2026-11-21.json`, `/tmp/orca_storage_audit_2026-11-21.txt`, and `/tmp/orca_local_merged_hash_checks_2026-11-21.json`. These temporary inventories may disappear after reboot. The values and completed-result hashes needed to interpret this cleanup are recorded above.

No jobs were submitted or restarted.
