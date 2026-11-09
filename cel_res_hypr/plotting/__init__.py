"""Manuscript plotting with the legacy headless/interactive backend policy."""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
import matplotlib as mpl

if not os.environ.get("MPLBACKEND"):
    interactive_3d_requested = "--show-cv-performance-3d" in sys.argv
    if not interactive_3d_requested or not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        mpl.use("Agg")
