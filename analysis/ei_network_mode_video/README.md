# CE/ER E/I network and activity video

Open `ce_er_ei_activity_modes.mp4`. Six 8-second conditions show 0, 20, 40, 60, 80 and 100% negative edges. Both models have 297 nodes and 3,604 directed edges. CE restores unknown edges at the known-edge negative proportion; ER is the existing binary-weight matched sign-sweep control.

Edges: orange positive, blue negative. Thickness is proportional to absolute post-scaling weight, divided by that model/condition's maximum. All connections are drawn; arrowheads are omitted for readability. Positions are weighted Kamada–Kawai layouts from the existing repository helper, fixed within each architecture across conditions.

**Width normalization cancels global spectral-radius scaling.** Consequently, CE's relative widths stay fixed across fractions and all ER widths are equal. The panels do not compare absolute recurrent strength. Exact post-scaled matrices, maximum weights and scale factors are saved.

Nodes show actual driven state, on a single shared color scale. Every condition starts at zero, receives the same U[-1,1] sequence and undergoes 500 washout steps. No task readout is fitted. PCA summaries use all 2,000 post-washout samples; the video displays the first 240 of those at two model steps per frame. Network/input realization seed is 20261117; target radius 0.95, leak 1, input scale 0.1 and reservoir bias 0. The default movie uses own-radius normalization.

The eigenvalue plots show the normalized recurrent matrix, with its largest-modulus eigenvalue highlighted. PCA uses mean-centered activity, not the recurrent-weight eigenvectors. The participation ratio measures variance dimensionality; it does not establish how many modes are computationally useful.

This is a **single illustrative realization**, not a statistical test or evidence of biological optimization. The displayed data support greater CE activity dimensionality at 0%, 20% and 100%, and greater ER dimensionality at 40%, 60% and 80%. The 80% case is a counterexample to a universal CE advantage at sign-imbalanced fractions.

Reproduce: `python tools/animate_ei_network_modes.py --out-dir analysis/ei_network_mode_video_new`. Settings, hashes and library versions are in `manifest.json`; diagnostics are also exported as PNG and CSV.
