# NARMA10 win fractions on the existing smaller grid

Source run 271800: 100 paired realizations, fractions 0/10/20/30/40/50%, radius targets 0.6/0.95/1.05, leaks 0.6/1.0, inputs 0.001/0.01/0.05/0.1/0.5, bias fixed to zero: 30 settings. Only the completed-unknown-sign CE variant and binary ER were evaluated. Own-radius and fixed-reference protocols remain separate.

A configuration is a CE win when mean CE test NMSE across the 100 realizations is strictly lower than mean ER test NMSE at that identical configuration. Exact ties count as neither. It is not a significance-filtered count. Readout ridge is selected on validation data, but no reservoir configuration is selected for these win-fraction or grid-mean plots.

| Normalization | Inhibitory edges | CE wins | ER wins |
|---|---:|---:|---:|
| Own-radius scaling | 0% | 100.0% | 0.0% |
| Own-radius scaling | 10% | 93.3% | 6.7% |
| Own-radius scaling | 20% | 83.3% | 16.7% |
| Own-radius scaling | 30% | 20.0% | 80.0% |
| Own-radius scaling | 40% | 10.0% | 90.0% |
| Own-radius scaling | 50% | 16.7% | 83.3% |
| Fixed-reference scaling | 0% | 100.0% | 0.0% |
| Fixed-reference scaling | 10% | 70.0% | 30.0% |
| Fixed-reference scaling | 20% | 60.0% | 40.0% |
| Fixed-reference scaling | 30% | 6.7% | 93.3% |
| Fixed-reference scaling | 40% | 20.0% | 80.0% |
| Fixed-reference scaling | 50% | 26.7% | 73.3% |

## Scope and validation

- All mean surfaces and matched grid coordinates agree with the independently reconstructed raw-data panel. All 12 overall win counts match the existing completed-run analysis. Bootstrap surfaces are reused from that analysis (2,000 whole-paired-realization draws, seed 271601), retaining all conditions together.
- Input-conditional curves have six settings each. Normalized AUC integrates the win fraction over the observed 0–50% range, divided by that range. This is not the full 0–100% archived AUC, and the grid differs. No mirroring, extrapolation or pooling with the 96-setting experiments.
- All test errors are retained. A CE own-radius observation at 40% inhibition has NMSE about 604 and strongly influences some mean comparisons. The mean-error figure is untrimmed; see the original run's narma_seed_tail_diagnostics.csv for median and leave-one-seed-out diagnostics.
- Win frequency measures breadth of relative advantage, not its size or absolute usefulness. Seeds are computational realizations, not biological animals. E/I here is an assigned edge-sign fraction.
- Existing source and analysis files are preserved; all new figures are PNGs. No new simulations.
