"""Io for reservoir research figures."""

from cel_res_hypr.analysis.aggregation import (
    _compute_dispersion_table,
    _compute_mean_table,
)
import csv
import numpy as np
import os
import pandas as pd


def _read_combined_csv(path: str) -> pd.DataFrame:
    """Read combined CSV and trim malformed extra columns per row if needed."""
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError as exc:
        print(f"[warn] {exc}")
        print(f"[warn] Retrying by trimming malformed extra column(s) from {path}.")

        with open(path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise

            expected_fields = len(header)
            rows = []
            trimmed_rows = 0
            padded_rows = 0
            first_trimmed_line = None
            first_padded_line = None

            for line_no, row in enumerate(reader, start=2):
                n_fields = len(row)
                if n_fields > expected_fields:
                    if first_trimmed_line is None:
                        first_trimmed_line = line_no
                    rows.append(row[:expected_fields])
                    trimmed_rows += 1
                elif n_fields < expected_fields:
                    if first_padded_line is None:
                        first_padded_line = line_no
                    rows.append(row + [""] * (expected_fields - n_fields))
                    padded_rows += 1
                else:
                    rows.append(row)

        if trimmed_rows == 0 and padded_rows == 0:
            raise

        print(
            f"[warn] Trimmed extra trailing column(s) on {trimmed_rows} row(s). "
            f"First trimmed line: {first_trimmed_line}."
        )
        if padded_rows:
            print(
                f"[warn] Padded missing trailing column(s) on {padded_rows} row(s). "
                f"First padded line: {first_padded_line}."
            )
        df = pd.DataFrame(rows, columns=header)
        numeric_cols = [
            "shuffle_id",
            "rho_target",
            "leak",
            "input_scale",
            "neuron_bias",
            "MC",
            "IPC",
            "KR",
            "GR",
            "raw_rho",
            "ref_rho",
            "post_rho",
            "scale_factor",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


def _read_compare_table(path: str, value_col: str = "auto"):
    df = pd.read_csv(path)
    required = {"mode", "metric"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Comparison table {path} is missing required column(s): {sorted(missing)}"
        )

    col = (value_col or "auto").strip().lower()
    if col == "auto":
        if "mean" in df.columns:
            col = "mean"
        elif "dispersion" in df.columns:
            col = "dispersion"
        else:
            raise ValueError(
                f"Could not infer comparison value column for {path}. "
                "Expected one of: 'mean', 'dispersion'."
            )
    elif col not in ("mean", "dispersion"):
        raise ValueError("--compare-value-col must be one of: auto, mean, dispersion")

    if col not in df.columns:
        raise ValueError(
            f"Requested comparison column '{col}' not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[col] = pd.to_numeric(df[col], errors="coerce")
    if "n_hparams" in df.columns:
        df["n_hparams"] = pd.to_numeric(df["n_hparams"], errors="coerce")
    return df, col


def _load_local_sign_binary_overlay(local_csv: str):
    if not local_csv:
        return None
    if not os.path.isfile(local_csv):
        print(f"[warn] real-wall overlay CSV not found: {local_csv}")
        return None

    extra = _read_combined_csv(local_csv)
    extra = _ensure_columns(extra)
    extra = extra[extra["mode"].astype(str) == "real"].copy()
    if extra.empty:
        print(f"[warn] real-wall overlay: no mode='real' rows in {local_csv}")
        return None

    disp_extra = _compute_dispersion_table(extra, mode="cv")
    mean_extra = _compute_mean_table(extra)
    out = {
        "cv_lookup": disp_extra.groupby(["mode", "metric"])["dispersion"].mean(),
        "mean_lookup": mean_extra.groupby(["mode", "metric"])["mean"].mean(),
        "kl_value": np.nan,
    }
    if "kl_to_gaussian" in extra.columns:
        kl_tbl = _compute_mean_table(extra, metrics=["kl_to_gaussian"])
        kl_lookup = kl_tbl.groupby("mode")["mean"].mean()
        out["kl_value"] = float(kl_lookup.get("real", np.nan))
    else:
        print(
            "[warn] real-wall overlay missing 'kl_to_gaussian'; using KL baseline z=0.0 for overlay wall."
        )
    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "mode",
        "shuffle_id",
        "rho_target",
        "leak",
        "input_scale",
        "neuron_bias",
        "MC",
        "IPC",
        "KR",
        "GR",
        "src",
    ]
    df = df.copy()
    for c in needed:
        if c not in df.columns:
            if c in ("MC", "IPC", "KR", "GR"):
                df[c] = np.nan
            elif c == "neuron_bias":
                df[c] = 0.0
            elif c == "shuffle_id":
                df[c] = -1
            elif c == "mode":
                df[c] = "unknown"
            elif c == "src":
                df[c] = "unknown"
            else:
                raise ValueError(f"Missing required column: {c}")
    # types
    df["mode"] = df["mode"].astype(str).str.strip()
    extras = [c for c in df.columns if c not in needed]
    return df[needed + extras].copy()
