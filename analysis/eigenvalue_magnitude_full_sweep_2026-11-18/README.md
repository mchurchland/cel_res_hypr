# Eigenvalue-magnitude distributions across the full broad E/I sweep

Uses all six saved broad-sweep fractions: 0, 20, 40, 60, 80 and 100% inhibitory edges. These are the same CE/ER matrix pairs used in the network video; no new trajectories or simulations are run.

All figures use identical axes across architectures and fractions. Histograms use common width-0.025 bins and show fractions of each matrix's 297 eigenvalues. ECDFs avoid bin choices. Ranked curves use a log rank axis to make the leading modes visible while retaining the full spectrum. The dotted reference marks radius 0.95.

These are post-normalization spectra at an own-radius target of 0.95. The largest magnitude is fixed by normalization; differences concern the remaining spectrum. Input scale 0.1, leak 1 and bias 0 describe the associated video dynamics, but the recurrent-weight spectra themselves do not depend on input scale or bias.

CE retains empirical magnitudes with unknown edges restored; ER is the existing binary-weight matched sign-sweep control. One illustrative realization (base seed 20261117), not uncertainty over seeds or evidence of an MC association. The finer 20–40% crossover experiment is a separate grid.

Exact sorted values: `eigenvalue_magnitudes.csv`. Descriptive quantities: `spectral_summary.csv`. Sources, hashes and settings: `manifest.json`. PNG only; existing outputs are unchanged.
