"""Output-path policies retained by plotting and aggregation commands."""

import os


def _safe_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{root}.v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def _replace_path(path: str) -> str:
    """Use a stable output path, replacing an older plot/table if it exists."""
    return path
