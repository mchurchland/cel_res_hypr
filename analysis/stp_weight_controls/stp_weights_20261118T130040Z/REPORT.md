# STP and weighted-control follow-up

Source: `/home/milesc/Thesis/results/stp_weight_controls/stp_weights_20261118T130040Z/stp_weight_controls.ALL.csv`; SHA256 `009be66364c3e0e4989fb699e7bc54139b888cc4d754275347f5c1b51b8d913f`.
Experiment `d62c4049cbc4959bc560a6774e238351826a8cfe279e6b915361846d3d8b7f40`. 67,200 paired MC/NARMA evaluations from
20 computational realizations. E/I percentages: [0.0, 10.0, 20.0, 30.0, 40.0, 50.0].

## How to assess the proposed CE advantage

1. Compare CE empirical against ER signed empirical: their exact signed weight multiset matches.
   ER empirical additionally matches the unsigned magnitude multiset, with unconstrained
   positive/negative magnitude allocation. Signed matching redistributes magnitudes
   across fractions to preserve the signed multiset, while the other controls keep
   their unsigned placements fixed throughout the sweep.
2. Compare CE shuffled against ER empirical: both randomize magnitude placement.
3. Compare CE binary against ER binary: removes magnitude heterogeneity in both.
4. Read absolute performance alongside paired differences and changes from static.
   A relative CE advantage can reflect ER deterioration, not CE improvement.
5. Check each STP setting, input scale and input scheme separately. Do not select
   a winning setting on test scores or pool configurations as independent samples.

`paired_architecture_comparisons.csv` contains paired realization differences.
Positive values favor the left family (MC: left minus right; NARMA: right error
minus left error). Intervals are pointwise 95% percentile bootstrap intervals
over paired seeds (2000 draws), not multiplicity-adjusted discoveries.
`stp_changes_from_static.csv` distinguishes improvement from control degradation.
The PNG heatmaps show all prespecified parameter sets, including unfavorable ones.

## Scope and limitations

- This extends the original continuous-STP/tanh reservoir, not c302 or BAAIWorm.
- Simulation backend: cuda; readout fits remain on CPU. GPU runs batch
  independent trajectories, retain float64 STP variables and float32 neural states,
  and record device metadata for each realization attempt.
- Both utilization and depletion operate in every plastic neuron. Parameters
  are generic sensitivity settings in model-step units, not measured worm fits.
- STP uses abs(previous signed tanh state), one state per presynaptic neuron,
  and recovered-rest gain normalization u*x/U. Signs are independently randomized
  at the edge level; these are neither fixed biological signs nor Dale neurons.
- Unknown-sign edges are restored before the E/I sweep. The sweep randomizes
  ALL signs, not only unknown edges.
- The 0–50% restriction is a sampling choice. No symmetry is imposed or inferred;
  biased NARMA inputs, nonlinear dynamics and STP can break apparent symmetry.
- Own-radius scaling targets 0.95 only at initialization. Shared-CE scaling uses
  the CE empirical multiplier at EACH fraction for every family; it does not
  fix scale across fractions. Exact magnitude matching survives shared scaling,
  while own-radius scaling can change magnitude distributions differently.
- No radius/leak/bias grid and no time-dependent renormalization. Leak=1, bias=0.
- Full input and CE sensory/ER random input subsets are separate experiments;
  restricted input matches the full input vector L2 norm, not driven activity.
- Scalar input and all-neuron linear readouts; no claim about biological memory
  behavior follows directly from MC/NARMA performance.
- Weight-shuffled CE does not preserve node strengths; ER does not preserve CE
  degree sequences. These controls narrow, but do not isolate, all topology effects.
- Optional frozen gain controls enabled: False. When present,
  they freeze each neuron's average transmission coefficient measured on training
  samples only. They do not match subsequent realized activity, nor constitute a
  unique causal isolation of synaptic history.
