# Exploratory follow-up: candidate findings and counterexamples

This is an exploratory search of stored final_results only. No simulations were run. Results are descriptive, conditional on the sampled grid and protocols, and selected after inspecting the data. Bootstrap intervals are marginal, not multiplicity-adjusted confirmation.

## 1. The CE–ER crossover is strongly conditional on input scale

At an input scale of 0.1, the updated sign sweep gives CE a mean MC and IPC advantage at 0%, 20%, 80% and 100% inhibitory edges. At 40% and 60%, ER is higher. At larger input scales, CE’s MC advantage near 20% and 80% becomes negligible or reverses, and its IPC advantage often reverses even at 0%. A single pooled crossover therefore reflects a chosen distribution over hyperparameters.

| Metric | Inhibitory edges | Input scale | CE − ER mean | 95% seed interval |
| --- | ---: | ---: | ---: | --- |
| MC | 0% | 0.1 | +1.311 | [+1.277, +1.343] |
| IPC | 0% | 0.1 | +1.446 | [+1.399, +1.492] |
| MC | 0% | 1.5 | +0.073 | [+0.042, +0.101] |
| IPC | 0% | 1.5 | -0.061 | [-0.126, -0.004] |
| MC | 20% | 0.1 | +1.383 | [+1.250, +1.521] |
| IPC | 20% | 0.1 | +1.470 | [+1.292, +1.653] |
| MC | 20% | 1.5 | -0.034 | [-0.072, +0.006] |
| IPC | 20% | 1.5 | -0.341 | [-0.405, -0.279] |
| MC | 40% | 0.1 | -6.145 | [-6.336, -5.968] |
| IPC | 40% | 0.1 | -7.182 | [-7.426, -6.952] |
| MC | 40% | 1.5 | -2.794 | [-2.844, -2.742] |
| IPC | 40% | 1.5 | -5.602 | [-5.682, -5.521] |
| MC | 80% | 0.1 | +1.027 | [+0.894, +1.173] |
| IPC | 80% | 0.1 | +1.121 | [+0.941, +1.304] |
| MC | 80% | 1.5 | -0.045 | [-0.086, -0.004] |
| IPC | 80% | 1.5 | -0.307 | [-0.372, -0.240] |
| MC | 100% | 0.1 | +1.173 | [+1.145, +1.204] |
| IPC | 100% | 0.1 | +1.308 | [+1.270, +1.343] |
| MC | 100% | 1.5 | +0.124 | [+0.089, +0.161] |
| IPC | 100% | 1.5 | +0.068 | [+0.009, +0.131] |

**Potential use:** define the crossover conditional on input drive, or state exactly which grid average it estimates. This can guide a small, targeted follow-up rather than a full multidimensional simulation campaign. Normalization controls and their archived KR remain separate; see the figure and full CSV.

## 2. Bias exchanges linear memory for IPC and state-space separation

Changing bias range from 0 to 0.1 usually lowers MC, raises IPC, and raises KR. The following estimates first average matched high-minus-low differences within each seed over the other parameters and the six shared E/I levels, then average across seeds.

| Model | Metric | Mean change from bias | 95% seed interval | Positive matched grid contrasts |
| --- | --- | ---: | --- | ---: |
| C. elegans: unknown signs matched | MC | -0.357 | [-0.367, -0.348] | 3/288 |
| C. elegans: unknown signs matched | IPC | +1.346 | [+1.325, +1.369] | 283/288 |
| C. elegans: unknown signs matched | KR | +0.737 | [+0.725, +0.750] | 288/288 |
| C. elegans: unknown signs removed | MC | -0.343 | [-0.354, -0.333] | 0/288 |
| C. elegans: unknown signs removed | IPC | +1.323 | [+1.302, +1.345] | 287/288 |
| C. elegans: unknown signs removed | KR | +0.584 | [+0.576, +0.591] | 288/288 |
| Matched ER | MC | -0.292 | [-0.302, -0.283] | 26/288 |
| Matched ER | IPC | +1.975 | [+1.957, +1.994] | 281/288 |
| Matched ER | KR | +1.156 | [+1.146, +1.166] | 288/288 |

**Potential use:** parameter importance hides the direction of its effect. Bias is a small contributor to total MC variance but consistently shifts which kind of computation the model supports. A plausible explanation is symmetry breaking in the tanh reservoir; the stored full-grid results do not decompose IPC by order at both bias settings, so this explanation is not established. The existing order-validation experiment contains bias=0 only.

## 3. Input drive creates a clear memory–separation tension

For both CE variants, changing input scale from 0.1 to 1.5 lowers MC and IPC and raises KR in every matched seed-averaged configuration contrast over the six shared E/I levels. ER also shows a universally negative MC contrast and positive KR contrast; its IPC response has exceptions. These signs describe 144 finite-design contrasts per model, not 144 independent biological samples.

| Model | Metric | Mean change with stronger input | 95% seed interval |
| --- | --- | ---: | --- |
| C. elegans: unknown signs matched | MC | -5.143 | [-5.198, -5.088] |
| C. elegans: unknown signs matched | IPC | -5.268 | [-5.348, -5.185] |
| C. elegans: unknown signs matched | KR | +3.183 | [+3.130, +3.238] |
| C. elegans: unknown signs removed | MC | -4.335 | [-4.384, -4.282] |
| C. elegans: unknown signs removed | IPC | -4.164 | [-4.242, -4.078] |
| C. elegans: unknown signs removed | KR | +2.310 | [+2.269, +2.350] |
| Matched ER | MC | -5.484 | [-5.515, -5.451] |
| Matched ER | IPC | -4.843 | [-4.900, -4.782] |
| Matched ER | KR | +7.285 | [+7.235, +7.338] |

**Potential use:** do not describe larger KR as uniformly better computational performance. The strongest tested input drive increases separation while reducing linear memory. The best tested input setting for MC is at the lower boundary of the grid; these data do not locate an unconstrained optimum below 0.1.

## 4. MC and IPC are not independent pieces of evidence

The inspected implementation computes MC and IPC from the same reservoir states, training/testing signals, ridge readout and delay range (30). IPC includes order 1, whose Legendre target is the original input, alongside higher orders. Consequently its linear contribution overlaps MC. The diagnostic residual IPC − MC is appropriate only conditional on those documented settings applying to the archived run; batched numerical solves can differ slightly. It is not an independently validated nonlinear task score.

In the updated own-radius CE-versus-ER sweep, CE’s overall IPC advantage at the two E/I endpoints is smaller than its MC advantage. The IPC−MC residual is lower in CE at every shared E/I level. Thus the endpoint MC and IPC advantages should not be presented as two independent demonstrations of a broad computational advantage. The residual pattern changes in the normalization controls, which further limits a universal interpretation.

**Audit finding:** 3 archived configuration/seed rows have IPC below MC, with a minimum residual of -0.385. Their exact identities are in tables/ipc_below_mc_anomalies.csv. These values are incompatible with an exact nonnegative decomposition if archived MC and IPC use the same linear target/readout protocol and exact arithmetic. Historical protocol differences or numerical solve differences need checking; these data do not identify the cause. No values were removed or clipped. Until resolved, IPC−MC should remain a diagnostic residual rather than a verified nonlinear-capacity measure.

**Potential use:** explicitly separate linear-memory contribution from higher-order capacity before interpreting IPC. This is a useful qualification even if the paper remains centered on locating a crossover.

## A practical way to reduce the crossover experiment

Input scale 0.1 gives the highest grid-averaged MC at every shared E/I level for all three model families. Holding radius/leak/bias fixed, it is also the best tested MC input in all 144 CE profiles, all 144 unknown-sign-removed CE profiles, and 129/144 ER profiles; ER prefers 0.5 in the other 15. These counts are over seed-averaged surfaces. The best input sits at the tested lower boundary, so it is not an estimate of the unconstrained optimum.

For an explicitly conditional MC crossover study, fixing input scale at 0.1 would reduce the grid from 96 to 24 settings while retaining all radii, leaks and bias levels. That is a 75% reduction in configurations per E/I point. It would estimate the crossover at a shared, generally favorable tested MC input, not the crossover averaged over the original full grid, and it would not optimize input separately for every ER profile. The existing data support checking this restricted estimand before any new simulations. Relevant per-profile and per-E/I optima are exported.

## Additional check: endpoint polarity symmetry

The all-negative minus all-positive contrasts, split by leak and bias, are exported in tables/polarity_asymmetry.csv. Their dependence on leak can motivate an explanation involving self-memory and recurrent sign. These were inspected as secondary exploratory diagnostics, not used to claim a new biological mechanism.

## Methods and limitations

All effects hold the other hyperparameters fixed before averaging. Each seed retains its full matched grid. Mean differences and 95% percentile intervals use 1,000 seed bootstrap draws; configurations and E/I levels are not resampled as biological replicates. Across-model paired numerical seeds preserve the experiment’s common-random-number design; they do not make CE and ER the same network. All summary effects across E/I use exactly the six shared fractions 0, 20, 40, 60, 80 and 100%, avoiding the extra CE-only reference point.

Main sign sweeps use updated KR over 200 seeds; archival normalization controls use 100 seeds and are kept separate. CE and ER differ in topology and weight structure, so no contrast isolates topology by itself. The models use assigned edge signs, not a fitted functional E/I balance. An unexplained correlation or crossover cannot independently estimate biological sign fraction.

The complete tables retain all examined strata, including counterexamples and weak effects. They are not confirmatory tests of hypotheses specified before seeing these data. No publication novelty or acceptance claim is made.

## Files

- crossover_by_input_scale.png
- bias_effects.png
- input_drive_effects.png
- ipc_memory_component_comparison.png
- tables/conditional_ce_minus_er.csv: all four metrics, all individual-parameter strata and complete-grid averages.
- tables/configuration_ce_minus_er.csv: paired seed-averaged differences at each full parameter combination.
- tables/parameter_manipulation_effects.csv: matched bias/input contrasts per E/I level and pooled over shared levels.
- tables/configuration_sign_consistency.csv: finite-grid counts for positive and negative manipulation effects.
- tables/polarity_asymmetry.csv: all-inhibitory minus all-excitatory contrasts.
- manifest.json: exact sources, cached matrices, code hashes, settings and validation.
