"""Data-driven report for complete MC/NARMA crossover grids of different sizes."""
import json

import numpy as np


NORM_NAMES = {"spectral_radius": "Own-radius", "original_radius": "Fixed-reference"}
METRIC_NAMES = {"MC": "MC", "NARMA10_test_nmse": "NARMA10 test NMSE"}


def write_report(tables, allocation, old, spec, validation, out):
    grid_size = len(spec["rho_values"]) * len(spec["leak_values"]) * len(spec["input_scales"])
    fractions = sorted(np.round(np.asarray(spec["sign_fracs"]) * 100, 8))
    steps = sorted(set(np.round(np.diff(fractions), 8)))
    inputs = sorted(spec["input_scales"])
    n = validation["repeats"]
    lines = ["# When does the connectome-derived reservoir outperform matched ER?", "",
             f"Stored run **{validation.get('run_id', out.name)}**: **{n} paired computational realizations, "
             f"{validation['rows']:,} rows, {grid_size} settings per architecture/normalization/E/I condition**. "
             "This analysis uses stored static-reservoir results; no simulations were launched.", "",
             f"E/I edge percentages: {', '.join(f'{x:g}' for x in fractions)}. "
             f"Input scales: {', '.join(f'{x:g}' for x in inputs)}. "
             f"Radius targets: {spec['rho_values']}; leak rates: {spec['leak_values']}; bias: zero.", "",
             "## Main results", "",
             "Uniform grid averages, fixed-input comparisons, and validation-selected performance answer "
             "different questions. Changing the sampled input levels also changes the grid average.", "",
             "| Normalization | Metric | CE higher mean / sampled fractions | CE-favoring intervals | ER-favoring intervals |",
             "| --- | --- | ---: | ---: | ---: |"]
    overall = tables["contrasts"].query("scope == 'grid_mean'")
    for (norm, metric), g in overall.groupby(["normalization", "metric"], sort=False):
        lines.append(f"| {NORM_NAMES[norm]} | {METRIC_NAMES[metric]} | {(g.ce_advantage > 0).sum()}/{len(g)} | {(g.ci_low > 0).sum()} | {(g.ci_high < 0).sum()} |")
    lines += ["", "![Mean performance](figures/01_overall_performance.png)", "",
              "![Paired architecture advantage](figures/02_paired_advantage.png)", "",
              "Positive CE advantage means CE minus ER MC, or ER minus CE NARMA error. "
              "Configurations are averaged within realization before resampling complete paired realizations. "
              "All intervals are pointwise 95% percentile seed-bootstrap intervals, not simultaneous or multiplicity-adjusted intervals.", "",
              "## Dependence on input drive", "",
              "![Input-dependent advantages](figures/03_advantage_by_input.png)", "",
              "![Absolute performance by input](figures/08_performance_by_input.png)", "",
              "The smallest input is not assumed to be optimal. Input scaling changes signal amplitude "
              "relative to the fixed MC ridge and the candidate NARMA ridge penalties, as well as nonlinear dynamics. "
              "These mechanisms are not isolated by a performance curve.", "",
              "| Input scale | Normalization | Metric | Fractions with CE-favoring intervals (%) | Fractions with ER-favoring intervals (%) |",
              "| ---: | --- | --- | --- | --- |"]
    for scale in inputs:
        scope = f"input_{scale:g}"
        for norm in spec["normalizations"]:
            for metric in METRIC_NAMES:
                g = tables["contrasts"]
                g = g[(g.scope == scope) & (g.normalization == norm) & (g.metric == metric)]
                ce = ", ".join(f"{v:g}" for v in g.loc[g.ci_low > 0, "ei_percent"])
                er = ", ".join(f"{v:g}" for v in g.loc[g.ci_high < 0, "ei_percent"])
                lines.append(f"| {scale:g} | {NORM_NAMES[norm]} | {METRIC_NAMES[metric]} | {ce or '—'} | {er or '—'} |")
    lines += ["", "## Conditional crossover locations", "",
              "| Metric | Normalization | Scope | Mean sign-change bracket(s), % | Interpolated root(s), % | Conditional bootstrap interval, % | Unique-root bootstrap fraction |",
              "| --- | --- | --- | --- | --- | --- | ---: |"]
    for row in tables["crossovers"].itertuples():
        if not row.n_mean_crossings:
            continue
        roots = json.loads(row.mean_crossings)
        bracket = "; ".join(f"{r['bracket_low']:g}–{r['bracket_high']:g}" for r in roots)
        points = "; ".join(f"{r['root']:.2f} ({r['direction']})" for r in roots)
        interval = f"[{row.ci_low:.2f}, {row.ci_high:.2f}]" if len(roots) == 1 else "Not summarized across multiple roots"
        lines.append(f"| {METRIC_NAMES[row.metric]} | {NORM_NAMES[row.normalization]} | {row.scope} | {bracket} | {points} | {interval} | {row.bootstrap_unique_fraction:.3f} |")
    lines += ["", f"Grid spacings are {', '.join(f'{v:g}' for v in steps)} percentage points. "
              "Roots are piecewise-linear interpolations within sampled sign-change brackets; no root is extrapolated. "
              "Bootstrap intervals describe seed uncertainty conditional on exactly one crossing, not the unknown "
              "shape between sampled fractions. Narrow intervals do not imply sub-grid experimental resolution. "
              "Multiple/no-crossing outcomes remain in `tables/crossovers.csv` and `configuration_crossovers.csv`.", "",
              "The approximately 24.25% known-edge inhibitory fraction already enters unknown-sign completion "
              "and fixed-reference construction. Proximity to a crossover is not an independent prediction of "
              "biological E/I balance, an optimum, or physiological current balance.", "",
              "## Matched configurations and NARMA validation selection", "",
              "![Configuration comparisons](figures/07_matched_configuration_advantages.png)", "",
              f"All {grid_size} matched settings are retained, including counterexamples. "
              "Favorable-setting counts are descriptive; settings are not independent biological replicates.", "",
              "![Validation-selected NARMA performance](figures/04_validation_selected_narma.png)", "",
              f"For each realization, architecture, normalization and E/I fraction, choose among {grid_size} settings "
              "using validation NMSE only; report the untouched test score. Within each setting, validation also "
              "selects ridge alpha. Neither selection uses test performance.", "",
              "| Normalization | E/I (%) | CE test NMSE | ER test NMSE | CE advantage [pointwise 95% interval] |",
              "| --- | ---: | ---: | ---: | --- |"]
    selected = tables["validation_selected_narma"]
    for row in selected.itertuples():
        lines.append(f"| {NORM_NAMES[row.normalization]} | {row.ei_percent:g} | {row.cel_matched_test_nmse:.4f} | {row.matched_er_test_nmse:.4f} | {row.ce_advantage:+.4f} [{row.ci_low:+.4f}, {row.ci_high:+.4f}] |")
    lines += ["", f"Validation-selected comparisons: {int((selected.ci_low > 0).sum())} intervals favor CE, "
              f"{int((selected.ci_high < 0).sum())} favor ER, and "
              f"{int(((selected.ci_low <= 0) & (selected.ci_high >= 0)).sum())} include zero.", "",
              "## NARMA tail and influence diagnostics", "",
              "`tables/largest_narma_test_errors.csv` retains the largest errors, with validation error, "
              "selected ridge and configuration. `narma_seed_tail_diagnostics.csv` compares means with "
              "medians after averaging configurations within each realization, and reports the fraction of "
              "paired seeds favoring CE. It also gives the range of means after omitting one whole paired "
              "seed at a time. These are descriptive influence checks; no errors or seeds are removed "
              "from the primary means, bootstrap intervals or crossover estimates.", "",
              "## Hyperparameter contributions", "",
              "![Allocated variance](figures/05_parameter_contributions.png)", "",
              "Exact finite-grid factorial decomposition includes all radius, leak and input main effects and "
              "interactions. Each interaction is divided equally among its participating parameters. Shares "
              "are ratios of summed within-E/I component variances to summed within-E/I total variances, "
              "excluding between-E/I mean changes. Bootstrap draws recompute the seed-averaged surfaces and "
              "decomposition. These shares depend on the sampled grid, not a universal sensitivity measure.", "",
              "| Architecture | Normalization | Metric | Radius target (%) | Leak (%) | Input scale (%) |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for (family, norm, metric), g in allocation.groupby(["family", "normalization", "metric"], sort=False):
        a = g.set_index("parameter").allocated_pct
        lines.append(f"| {'CE' if family == 'cel_matched' else 'ER'} | {NORM_NAMES[norm]} | {METRIC_NAMES[metric]} | {a.rho_target:.1f} | {a.leak:.1f} | {a.input_scale:.1f} |")
    lines += ["", "## Normalization, provenance and limits", "",
              "![Spectral diagnostics](figures/06_spectral_diagnostics.png)", "",
              "Own-radius and fixed-reference normalization are distinct protocols, paired on the same raw "
              "realization. Radius correlations do not isolate causal effects of recurrent gain or topology.", "",
              "CE retains empirical magnitudes; the matched ER sign-sweep control is binary-weighted. "
              "This run is not the newer weighted-control/STP experiment, and cannot separate topology from "
              "weight heterogeneity or weight placement. E/I sweeps randomize all edge signs after unknown-sign "
              "completion; the sign structure at each fraction is not the biological sign map.", "",
              f"MC uses delays 1–{spec['tasks']['mc_max_delay']} and a fixed ridge penalty "
              f"{spec['tasks']['mc_ridge_alpha']:g}; high scores can approach this truncated ceiling. "
              "NARMA10 uses ReservoirPy 0.4.2 with its upstream lag convention and an intercept-fitted "
              "scikit-learn Ridge readout. Inputs and all segment lengths are recorded in the manifest. "
              "Computational seeds are realizations of one connectome, not independent animals.", "",
              f"Recorded devices: {', '.join(validation['devices'])}. Device-stratified descriptive summaries "
              "are exported. Device assignment and seed IDs are confounded, so device-group differences "
              "cannot be attributed to hardware alone.", ""]
    if (out / "tables/reference_run_overlap.csv").exists():
        lines += ["The previous crossover run shares seed IDs and overlapping configurations with this run. "
                  "It is not an independent replication. `tables/reference_run_overlap.csv` compares exactly "
                  "matched settings; `reference_run_overlap_audit.json` records protocol compatibility, "
                  "pair counts and numerical differences. The datasets are not pooled.", ""]
    if len(old):
        lines += ["`tables/old_grid_comparison.csv` additionally retains descriptive legacy MC grid averages. "
                  "Different input grids and historical protocols must not be interpreted as matched contrasts.", ""]
    lines += ["## Validation", "",
              f"All {validation['shards']:,} shards are complete, unique and consistent with the merged "
              "table. All scores are finite, MC respects its documented bounds, NRMSE squared matches NMSE, "
              "and no rows were filtered or clipped. Source-data and simulation-code hash checks are recorded "
              "individually in `VALIDATION.json`; analysis provenance is in `manifest.json`. "
              "Figures are PNGs; no PDFs were generated."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
