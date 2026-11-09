#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge chunk CSVs and write combined, mean, and dispersion tables.

Source CSVs are read-only. Existing outputs receive version suffixes.
Pass --glob-variants explicitly for current invariance_variants.csv files;
the historical default glob targets ce_real_reps.csv. Use graph_hist.py on
combined.ALL.csv to generate figures. Legacy plot flags remain accepted.
"""

import os
import argparse
import sys
from pathlib import Path

# Support both the historical script command and python -m util.merge.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
from util.graph_utils import (
    _safe_path,
    _read_glob,
    _build_combined,
    _compute_dispersion_table,
    _compute_mean_table,
)

# ---------------------------- CLI ----------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Merge CE real/shuffle + variants and plot all together."
    )
    ap.add_argument(
        "--glob-shuf",
        default="experiment_full_chunks/chunk_*/bio_vs_shuffle_invariance.csv",
        help="Glob for CEL+bioW vs shuffled CSVs.",
    )
    ap.add_argument(
        "--glob-variants",
        default="experiment_full_chunks/chunk_*/ce_real_reps.csv",
        help="Glob for per-chunk variants CSVs.",
    )
    ap.add_argument(
        "--out-dir",
        default="experiment_full_merged",
        help="Directory to write merged CSVs and figures.",
    )
    ap.add_argument(
        "--bins", type=int, default=40, help="Histogram bins for dispersion plots."
    )
    ap.add_argument(
        "--scatter-alpha", type=float, default=0.55, help="Alpha for MC-vs-GR scatter."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df_shuf = _read_glob(args.glob_shuf)
    df_var = _read_glob(args.glob_variants)

    combined = _build_combined(df_shuf, df_var)
    out_comb = _safe_path(os.path.join(args.out_dir, "combined.ALL.csv"))
    combined.to_csv(out_comb, index=False)
    print(f"[saved] {out_comb}  (rows={len(combined)})")

    mean_tbl = _compute_mean_table(combined)
    out_mean = _safe_path(os.path.join(args.out_dir, "mean_by_group.ALL.csv"))
    mean_tbl.to_csv(out_mean, index=False)
    print(f"[saved] {out_mean}  (rows={len(mean_tbl)})")

    disp = _compute_dispersion_table(combined, mode="cv")
    out_disp = _safe_path(os.path.join(args.out_dir, "dispersion_by_group.ALL.csv"))
    disp.to_csv(out_disp, index=False)
    print(f"[saved] {out_disp}  (rows={len(disp)})")


if __name__ == "__main__":
    main()
