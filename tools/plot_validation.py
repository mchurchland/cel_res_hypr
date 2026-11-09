"""Generate deterministic figures/tables and optionally verify recorded hashes.

Run from the repository root with --out-dir pointing to an empty directory.
--source-root selects another checkout's graph_hist implementation, allowing
original and refactored code to be exercised with the same plotting inputs.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def plotting_fixture(repeats: int = 6) -> pd.DataFrame:
    """Return the seed-1603, ten-architecture synthetic plotting fixture."""
    rng = np.random.default_rng(1603)
    rows = []
    modes = [
        "real",
        "shuffle",
        "celW+connShuf",
        "conn_shuf_only",
        "local_sign+binary",
        "global_sign_pres",
        "binary+shuffle",
        "binary+conshuffle+wshuffle",
        "binary_base",
        "binary_base_topology_shuffle",
    ]
    for m, mode in enumerate(modes):
        for rep in range(repeats):
            offset = rng.uniform(0.5, 1.5)
            for i in range(8):
                rows.append(
                    dict(
                        mode=mode,
                        shuffle_id=rep,
                        src="fixture",
                        rho_target=[0.8, 1.05][i % 2],
                        leak=[0.6, 0.8][i // 2 % 2],
                        input_scale=[0.1, 0.5][i // 4],
                        neuron_bias=0.0,
                        MC=offset * (2 + 0.2 * m + 0.08 * i),
                        IPC=offset * (1 + 0.1 * m + 0.2 * i),
                        KR=offset * (3 + 0.3 * m + 0.1 * i),
                        GR=offset * (1 + 0.2 * m + 0.1 * i),
                        raw_rho=2 + 0.1 * m,
                        ref_rho=2.0,
                        post_rho=0.8,
                        scale_factor=0.4,
                        normalization="spectral_radius",
                        wt_mean=0.01 * m,
                        kl_to_gaussian=0.1 * m,
                        cosine_similarity=0.9,
                    )
                )
    return pd.DataFrame(rows)


def generate_outputs(out_dir: Path, source_root: Path, dpi: int = 50) -> dict:
    """Run eight plotting paths once, recording PNG/CSV/LaTeX content hashes.

    DPI is capped only at save time to keep validation memory use modest.
    Call in a fresh process for each source checkout: Python caches imports.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Plot validation requires an empty output directory.")
    if not (source_root / "graph_hist.py").is_file():
        raise FileNotFoundError(f"No graph_hist.py in {source_root}")
    sys.path.insert(0, str(source_root.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    graph = importlib.import_module("graph_hist")

    out_dir.mkdir(parents=True, exist_ok=True)
    original_save = Figure.savefig

    def small_save(self, *args, **kwargs):
        kwargs["dpi"] = dpi
        return original_save(self, *args, **kwargs)

    def run(name, call):
        plt.rcdefaults()
        destination = out_dir / name
        destination.mkdir()
        call(destination)
        plt.close("all")

    Figure.savefig = small_save
    try:
        with (
            (out_dir / "validation.log").open("w") as log,
            contextlib.redirect_stdout(log),
        ):
            combined = plotting_fixture()
            disp = graph._compute_dispersion_table(combined)
            mean = graph._compute_mean_table(combined)
            run(
                "contours",
                lambda d: graph.plot_cv_performance_contours_2d(
                    disp, mean, str(d), show=False, bins=12
                ),
            )
            run(
                "triptych",
                lambda d: graph.plot_cv_performance_contour_triptych(
                    disp, mean, combined, str(d), show=False, bins=12
                ),
            )
            run(
                "differences",
                lambda d: graph.plot_cv_performance_differences_3d(
                    graph.print_cv_difference_tables(disp, str(d)),
                    mean,
                    str(d),
                    show=False,
                ),
            )
            run("tables", lambda d: graph.print_cv_difference_tables(disp, str(d)))
            sign = []
            for norm in ("spectral_radius", "original_radius"):
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                    part = combined[combined["mode"] == "real"].copy()
                    part["mode"] = f"sign_test_og_cel{fraction}"
                    part["normalization"] = norm
                    for metric in ("MC", "IPC", "KR", "GR"):
                        part[metric] *= 1 + fraction * (
                            1 if norm == "spectral_radius" else 0.5
                        )
                    part["raw_rho"] *= 1 + fraction
                    sign.append(part)
            sign = pd.concat(sign, ignore_index=True)
            sign_disp = graph._compute_dispersion_table(sign)
            run(
                "signnorm",
                lambda d: graph.plot_sign_norm_ablation(
                    sign, str(d), cv_include_rho=True
                ),
            )
            run(
                "signfrac",
                lambda d: graph.plot_frac_cv_meanline(
                    sign_disp, sign, str(d), show=False
                ),
            )
            weight = combined.copy()
            modes = combined["mode"].unique()
            weight["mode"] = weight["mode"].map(
                {mode: f"weight_test{i * 0.1}" for i, mode in enumerate(modes)}
            )
            weight_disp = graph._compute_dispersion_table(weight)
            run(
                "weights",
                lambda d: graph.plot_weight_gauss_cv_metric_2d(
                    weight_disp, weight, str(d)
                ),
            )
            run("rho", lambda d: graph.plot_rho_cv_other_perf(combined, str(d)))
    finally:
        Figure.savefig = original_save
        plt.close("all")
    artifacts = [
        {
            "file": str(path.relative_to(out_dir)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(out_dir.rglob("*"))
        if path.suffix in {".png", ".csv", ".tex"}
    ]
    result = {"seed": 1603, "save_dpi": dpi, "files": artifacts}
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--reference-manifest", type=Path)
    args = parser.parse_args()
    result = generate_outputs(args.out_dir, args.source_root)
    if args.reference_manifest:
        reference = json.loads(args.reference_manifest.read_text())
        expected = {row["file"]: row["sha256"] for row in reference["files"]}
        actual = {row["file"]: row["sha256"] for row in result["files"]}
        differences = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        if differences:
            raise SystemExit(f"Plot artifact differences: {differences}")
        print(f"All {len(actual)} plot/table hashes match {args.reference_manifest}")
    else:
        print(f"Wrote {len(result['files'])} plot/table artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
