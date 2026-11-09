"""Stable 20-column experiment CSV schema and selective rank updates.

Rank-only writes preserve MC/IPC; KR-only writes also preserve existing GR.
Row identity includes architecture, normalization, repeat, grid, seed, source.
"""

from pathlib import Path

CSV_HEADER = [
    "mode",
    "normalization",
    "shuffle_id",
    "rho_target",
    "leak",
    "input_scale",
    "neuron_bias",
    "MC",
    "IPC",
    "KR",
    "GR",
    "wt_mean",
    "cosine_similarity",
    "kl_to_gaussian",
    "seed",
    "src",
    "raw_rho",
    "ref_rho",
    "post_rho",
    "scale_factor",
]


RANK_ROW_KEY_COLUMNS = (
    "mode",
    "normalization",
    "shuffle_id",
    "rho_target",
    "leak",
    "input_scale",
    "neuron_bias",
    "seed",
    "src",
)


def _rank_row_key(row: dict[str, str]) -> tuple[str, ...]:
    """Identify one architecture/hyperparameter/repeat row in a chunk CSV."""
    return tuple(row[column] for column in RANK_ROW_KEY_COLUMNS)


def _update_rank_columns(
    out_csv: str,
    rows: list[tuple],
    *,
    update_gr: bool = True,
) -> None:
    """Patch KR and optionally GR without touching other metric columns."""
    import csv
    import os
    import tempfile

    path = Path(out_csv)
    new_rows = [dict(zip(CSV_HEADER, map(str, row), strict=True)) for row in rows]

    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in new_rows:
                row["MC"] = ""
                row["IPC"] = ""
                if not update_gr:
                    row["GR"] = ""
                writer.writerow(row)
        return

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(
                f"Cannot apply rank-only update to {out_csv}: unexpected CSV schema."
            )
        existing_rows = list(reader)

    updates = {_rank_row_key(row): row for row in new_rows}
    matched: set[tuple[str, ...]] = set()
    for row in existing_rows:
        key = _rank_row_key(row)
        if key in updates:
            row["KR"] = updates[key]["KR"]
            if update_gr:
                row["GR"] = updates[key]["GR"]
            matched.add(key)

    for row in new_rows:
        if _rank_row_key(row) not in matched:
            row["MC"] = ""
            row["IPC"] = ""
            if not update_gr:
                row["GR"] = ""
            existing_rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        newline="",
        delete=False,
    ) as f:
        temp_name = f.name
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(existing_rows)
    os.replace(temp_name, path)


def save_rows(
    out_csv: str,
    rows: list[tuple],
    *,
    append: bool = False,
    rank_only: bool = False,
    kr_only: bool = False,
) -> None:
    """Write/append full rows or patch selected rank columns in an existing CSV.

    The caller creates the output directory. Rank patches require exact schema
    agreement and replace the file atomically when updating existing rows.
    """
    if rank_only and kr_only:
        raise ValueError("rank_only and kr_only are mutually exclusive.")
    if kr_only:
        _update_rank_columns(out_csv, rows, update_gr=False)
        return
    if rank_only:
        _update_rank_columns(out_csv, rows)
        return

    mode = "a" if append and Path(out_csv).exists() else "w"
    with open(out_csv, mode, newline="") as f:
        import csv

        w = csv.writer(f)
        if mode == "w":
            w.writerow(CSV_HEADER)
        w.writerows(rows)
