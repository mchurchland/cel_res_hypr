# Performance versus E/I balance, with hyperparameter dots

Each dot is a single hyperparameter combination averaged over seeds, with no CV axis or connecting lines.

- Color: spectral-radius target (0.6, 0.8, 0.95, 1.05).
- Shape: leak rate (circle 0.6, square 0.8, triangle 1).
- Area: input scale (0.1, 0.5, 1, 1.5).
- Outline: neuron-bias range (none for 0; black outline for 0.1).

Horizontal jitter is deterministic and bounded by ±1.2 percentage points, matched across metrics and normalization modes. Performance is never jittered. The x-axis is percentage of negative **edges**, rather than inhibitory neurons. Points just outside 0–100% are display jitter only.

Updated sign-sweep figures average 200 seeds per combination. Separate normalization-control figures average 100 seeds per combination and retain their archival KR protocol; compare normalization rows within those figures. Dot colors encode hyperparameters, with architecture families separated into panels. The seed means do not display seed uncertainty.

Open `cel_matched_performance_dots.png` for the primary connectome view, `all_families_performance_dots.png` for all families, or the normalization figures for own-radius versus fixed-reference scaling. All figures are 300 dpi PNGs. `plotted_points.csv.gz` preserves exact unjittered coordinates, displayed coordinates, encodings and seed counts.

Reproduce with `python tools/plot_ei_hyperparameter_dots.py --out-dir analysis/new_dot_figures`. Exact sources, hashes and settings are in `manifest.json`.
