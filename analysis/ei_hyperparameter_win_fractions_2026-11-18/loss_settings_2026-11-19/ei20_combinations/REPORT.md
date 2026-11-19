# Exact hyperparameter combinations at 20% inhibitory edges

All statements below describe the sign of paired mean differences, not significance-filtered outcomes. Exact differences and existing uncertainty/support fields are in exact_comparisons_ei20.csv. Archived normalization controls and both CE variants are included in the CSV, while the figures emphasize the main archived sweep and newer MC/NARMA experiment.

## Archived completed-sign CE: MC

- Leak 0.6: wins all 32 combinations of radius, input and bias.
- Leak 0.8: wins 31/32. The only loss is radius 1.05, input 1.5, bias 0.1.
- Leak 1: wins 10/32. Wins every radius at input 0.1, either bias; also wins radius 0.6 at input 0.5, either bias. All other leak-1 settings lose.
- IPC exceptions are shown explicitly in the heatmap: leak 0.6 wins 30/32, leak 0.8 wins 26/32, leak 1 wins 7/32. KR favors CE for all 96 settings at this fraction.

## Newer MC/NARMA run

- The two metrics have identical win/loss classifications across all 30 settings at 20% within each normalization protocol.
- Own-radius: inputs 0.001, 0.01, 0.05 and 0.1 win for every radius/leak combination. At input 0.5 only radius 0.6 + leak 0.6 wins; the other five combinations lose.
- Fixed-reference: radius 0.6 loses at every input and both leaks; radius 0.95 wins at inputs <=0.1 and loses at 0.5; radius 1.05 wins at every input and both leaks.

## Matched changes

matched_parameter_flips_at_ei20.csv changes one parameter between its lowest and highest tested levels while holding all other parameters fixed. It records CE-win→ER-win and reverse transitions. This is not an ANOVA attribution, nor does the number of transitions identify a unique dynamical mechanism. The counts depend on the tested ranges; the two experiments have different grids.

No new simulations, source changes, test-score optimization or pooling across grids. Binary ER remains the comparison; this is not the newer weight-matched ER study. Larger KR denotes separation rather than general task superiority.
