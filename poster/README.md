# NeurReps poster

`poster.pdf` is the rendered, single-page poster. `poster_preview.png` is a screen preview. `poster.tex` and `assets/` are sufficient to compile the poster without access to the parent thesis directory.

## Format

61 cm wide × 85 cm high, portrait, following the **workshop** dimensions on the [official NeurIPS 2026 poster instructions](https://neurips.cc/Conferences/2026/PosterInstructions), checked November 2, 2026. Print at 100% / actual size.

This is a custom LaTeX/TikZ poster layout, not an official branded template. The [NeurReps website](https://neurreps.org/) supplies a paper template; no downloadable poster TeX template was found there or in the [Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook). Main-conference landscape dimensions do not apply to this workshop poster.

## Build

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error poster.tex
pdftoppm -scale-to 1800 -png -singlefile poster.pdf poster_preview
```

To regenerate the connectome image using the parent thesis code and data:

```sh
python render_network.py
```

## Content and figure provenance

- Scientific content: `neurreps_abstract.tex` in this directory, checked against `../thesis_tex/thesis.tex` and `../thesis_tex/neurreps_2026/paper.tex`.
- Poster title: the full paper’s “Determinants of hyperparameter robustness in connectome reservoir computing,” used after removing generalization-rank content at the user’s request.
- Authors, ordering, and affiliation mapping: the full thesis TeX. The supplied abstract is anonymized, so its author list was inferred from that full-paper source.
- `assets/connectome.pdf` and `.png`: rendered by `render_network.py`, reusing `../architecture_variant_figures/make_weighted_graph_examples.py` (project loader and unknown-sign assignment) and `../Connectome/ce_adj.npy` / `ce_unknown_sign_weights.npy`. The illustration shows only the largest weakly connected component: 278 nodes and 3,477 edges, of which 854 are negative. The full experimental network remains 297 nodes and 3,604 edges; only this illustration is filtered. Unknown signs use seed 7 and the project’s empirical sign ratio. Node positions use the original weighted Kamada–Kawai layout, rigidly rotated with equal axis scaling. These are connectivity coordinates, not anatomical positions. All component edges are retained. Orange/blue indicate positive/negative weights; line width and shading use the original renderer. Labels and arrowheads are omitted for clarity.
- `assets/architecture_tradeoff.png`: unchanged copy of `../thesis_tex/neurreps_2026/figures/architecture_tradeoff.png`. Its abstract caption specifies highest-density **50%** contours. This is distinct from the 70% contour variants described in the extended-abstract results directory README.
- `assets/performance_robustness_curves.png`: unchanged copy of `../thesis_tex/thesis/sign_fraction/cel_matched/meanpoint_frac_cv_lines.png`, the full paper’s E/I balance sweep. It includes the IPC/KR/MC performance–CV curves and the associated raw-radius/mean-CV panel.
- The current poster contains no generalization-rank analysis, claims, or schematic. The initial version’s radius plotting script and assets are unused by the current TeX.
- No simulations, statistical fits, or new experimental results were generated.  The original abstract and parent-directory files were not modified.

## Validation

Compiled with pdfLaTeX through latexmk; verified one page at 1729.13 × 2409.45 PDF points (61 × 85 cm); visually inspected the rendered PNG for overlap, clipping, and legibility. No overfull boxes or compilation errors remain. Benign underfull paragraph warnings may remain.

## PDF annotation fixes

Original marked-up document preserved as `poster_annotated.pdf`. Six comments were addressed: redraw the network; rename section 1 to “Connectome as a reservoir”; move the CV block downward; use “CE” for the empirical fraction in the architecture caption; define pm1 as ±1 weights; use “E/I edge balance” in the tradeoff caption.
