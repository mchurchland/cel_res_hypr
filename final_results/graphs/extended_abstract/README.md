# Extended-abstract figure variants

These figures are deliberately distinct from the thesis/journal versions.
They use the `extended-abstract` colour scheme and 70% highest-density
contours instead of the 50% contours used by the thesis figures.
The raw-spectral-radius summary includes GR as a fourth row; lower GR means
greater convergence after the shared input tail. The thesis version retains
the separate GR synthesis figure.

All plotted analyses use only these canonical rank-updated effective-rank
inputs:

- `final_results/main/combined.ALL.GRKR_erank.rank_updated.csv`
- `final_results/shuf/combined.ALL.GRKR_erank.rank_updated.csv`
- `final_results/sign_frac/cel_matched/combined.ALL.GRKR_erank.rank_updated.csv`
- `final_results/sign_frac/cel_removed/combined.ALL.GRKR_erank.rank_updated.csv`
- `final_results/sign_frac/matched_er/combined.ALL.GRKR_erank.rank_updated.csv`

Regenerate the complete set from the repository root with:

```bash
python make_extended_abstract_figures.py
```

The generator does not overwrite anything under `final_results/graphs/thesis`.
