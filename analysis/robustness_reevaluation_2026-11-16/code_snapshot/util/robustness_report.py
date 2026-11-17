"""Reproducible report with numerical findings drawn from analysis tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
from util.robustness import bootstrap_mean


def markdown(frame, columns=None, digits=3):
    frame=frame if columns is None else frame[columns]
    def cell(x):
        if isinstance(x,(float,np.floating)):
            return f"{x:.{digits}f}" if np.isfinite(x) else "—"
        return str(x)
    header="| "+" | ".join(map(str,frame.columns))+" |\n"
    line="| "+" | ".join("---" for _ in frame.columns)+" |\n"
    return header+line+"\n".join("| "+" | ".join(cell(x) for x in row)+" |" for row in frame.to_numpy())+"\n"


def write_report(t,settings,out):
    # Preserve distinct inclusive maximum and zero-attainment endpoints at export.
    t["threshold_curves"].to_csv(out/"tables/threshold_curves.csv",index=False,float_format="%.17g")
    # Paired contrasts preserve common numerical seeds across architectures.
    contrast_rows=[]
    rng=np.random.default_rng(settings["seed"]+101)
    measures=["mean","cv","sd","iqr","mad","bottom10","attain_q50","retention90","sensitivity_mean"]
    for dataset in ("main","shuf"):
        d=t["per_realization"].query("dataset == @dataset")
        for metric,g in d.groupby("metric"):
            reference=g[g.architecture=="real"].set_index("seed")
            for arch,h in g[g.architecture!="real"].groupby("architecture"):
                alternative=h.set_index("seed")
                shared=reference.index.intersection(alternative.index)
                delta=reference.loc[shared,measures].to_numpy()-alternative.loc[shared,measures].to_numpy()
                mean,ci=bootstrap_mean(delta,settings["bootstrap"],rng)
                contrast_rows.extend(dict(dataset=dataset,metric=metric,alternative=arch,measure=measure,
                    ce_minus_alternative=value,ci_low=lo,ci_high=hi,n_paired_seeds=len(shared))
                    for measure,value,lo,hi in zip(measures,mean,ci[0],ci[1]))
    pd.DataFrame(contrast_rows).to_csv(out/"tables/architecture_contrasts.csv",index=False)
    a=t["associations"]
    pooled=a[(a.scope=="pooled_realizations") & a.dataset.isin(["main","shuf"]) & a.measure.isin(["cv","sd","iqr","mad","bottom10","attain_q50","sensitivity_mean"])]
    c=t["comparison"]
    main=c[c.dataset=="main"]
    audit=t["data_audit"]
    text=["# Reevaluate robustness in connectome reservoir computing\n",
          "This analysis attempts to falsify a general performance–robustness tradeoff. All performance observations come from `final_results/`. No reservoir simulations were run. Existing results and figures were read only; their SHA256 hashes were verified before and after the analysis.\n",
          "## Findings\n"]
    for dataset in ("main","shuf"):
        sentences=[]
        for metric in ("MC","IPC","KR"):
            g=pooled[(pooled.dataset==dataset)&(pooled.metric==metric)].set_index("measure")
            values=[f"{name.upper()} {g.loc[name,'rho']:.2f} [{g.loc[name,'ci_low']:.2f}, {g.loc[name,'ci_high']:.2f}]" for name in ("cv","sd","iqr","mad")]
            sentences.append(f"- **{metric}:** "+"; ".join(values)+".")
        text.append(f"### Does the association survive replacing CV? {dataset}\n\nSpearman ρ and 95% seed-cluster bootstrap intervals:\n\n"+"\n".join(sentences)+"\n")
    text.append("Positive associations with SD, IQR, or MAD describe greater *absolute dispersion*, not lower useful capability. CV divides SD by mean; an algebraic denominator alone does not predict the observed sign. These correlations cannot identify a causal performance cost. The lower-tail and threshold analyses below test a different, more direct meaning of reliable computation.\n")
    text.append("### Are low performers called robust because they are uniformly poor?\n")
    for metric in ("MC","IPC","KR"):
        g=main[main.metric==metric]
        low=g.loc[g["mean"].idxmin()]
        high=g.loc[g["mean"].idxmax()]
        threshold=settings["thresholds"][metric]["anchors"]["attain_q50"]
        text.append(f"For {metric}, the lowest-mean architecture is `{low.architecture}`: mean {low['mean']:.3f}, SD {low.sd:.3f}, CV {low.cv:.3f}, bottom-decile mean {low.bottom10:.3f}, and attainment {low.attain_q50:.3f} at the common exploratory threshold {threshold:.3f}. The highest-mean architecture is `{high.architecture}`: mean {high['mean']:.3f}, SD {high.sd:.3f}, bottom-decile mean {high.bottom10:.3f}, attainment {high.attain_q50:.3f}. Their 90% retention fractions are {low.retention90:.3f} and {high.retention90:.3f}, respectively.\n")
    text.append("A small absolute spread can therefore be consistent with consistently weak computation. A high retention fraction likewise requires a useful optimum before it can support a capability claim. No independently specified usefulness threshold was supplied; threshold attainment here is exploratory. A floor does **not** necessarily reduce CV: SD/mean can grow as the mean approaches zero.\n")
    text.append("### Which architecture comparisons remain consistent?\n\nThe following table gives C. elegans minus each main-experiment alternative. Positive mean, lower-tail and threshold differences favor C. elegans; negative dispersion or sensitivity differences favor C. elegans. These are differences of seed means; full metric uncertainty intervals are in `tables/comparison.csv`.\n")
    differences=[]
    for metric in ("MC","IPC","KR"):
        g=main[main.metric==metric]
        base=g[g.architecture=="real"].iloc[0]
        for _,r in g[g.architecture!="real"].iterrows():
            differences.append(dict(metric=metric,alternative=r.architecture,mean=base["mean"]-r["mean"],sd=base.sd-r.sd,iqr=base.iqr-r.iqr,mad=base.mad-r.mad,bottom10=base.bottom10-r.bottom10,attain_q50=base.attain_q50-r.attain_q50))
    text.append(markdown(pd.DataFrame(differences)))
    text.append("### Comparable mean performance\n\nPairwise comparisons first restrict to the intersection of the two empirical mean ranges, then greedily match the closest means without replacement, with a caliper of 5% of the combined mean span. This is exploratory conditioning, not causal adjustment. Matches can change seed composition. All architecture pairs, counts and matched differences are in `tables/mean_matched.csv`; exact matched seeds are in `tables/mean_matched_pairs.csv`. A small match count prevents a general comparison.\n")
    m=t["mean_matched"]
    selected=m[(m.architecture_a=="real")|(m.architecture_b=="real")]
    counts=selected[selected.measure=="mean"]
    text.append(markdown(counts,["dataset","metric","architecture_a","architecture_b","n_overlap_a","n_overlap_b","n_matched","matched_difference_a_minus_b"]))
    for measure in ("sd","iqr","mad","bottom10"):
        g=m[(m.measure==measure)&(m.n_matched>=20)]
        same=((np.sign(g.matched_difference_a_minus_b)==np.sign(g.full_difference_a_minus_b))).sum()
        text.append(f"For {measure}, {same}/{len(g)} architecture pairs with at least 20 matches retain their full-sample difference sign. This count includes both main and shuffle panels and does not establish significance.\n")
    text.append("### Floors and finite bounds\n\nThe code sums 30 clamped squared correlations for MC and 30 delays × five single-order Legendre targets for IPC. Their implementation-based bounds are [0,30] and [0,150]. KR is effective rank of a state matrix with 297 node features, hence [0,297]; nonzero effective rank is at least one but a zero state matrix returns zero. The 297-node adjacency shape was checked during inspection. The current launcher/protocol supports these ceilings, but the archival CSVs do not embed delay/order/stream counts, so historical upper-bound conclusions are explicitly conditional. IPC here is the implemented restricted target set, not a full polynomial-basis IPC measurement.\n")
    text.append("For a bounded score a ≤ P ≤ b, nonnegativity of E[(P−a)(b−P)] gives Var(P) ≤ (μ−a)(b−μ). We report variance divided by this ceiling, and plots versus distances from both bounds. The ceiling constrains possibilities; it is not expected null variance or a correction that establishes intrinsic sensitivity. Zero ceilings yield an undefined ratio. Main Q25 threshold fractions, SD restricted to configurations above Q25 (minimum 20 retained configurations), and descriptive partial rank correlations controlling the low-performance fraction provide additional floor diagnostics. The truncated subsets vary between surfaces and should not be treated as the original fixed design.\n")
    text.append(f"Observed nonfinite values across selected metric rows: {audit.n_nonfinite.sum()}; negative values: {audit.n_negative.sum()}; values outside the conditional bounds: {audit.n_outside_conditional_bounds.sum()}. No clipping or imputation was applied.\n")
    text.append("### Hyperparameter variation versus seed variation\n\nThe explicit `(architecture, normalization, seed, hyperparameters)` identity is unique. Each seeded surface contains the same 96 configurations. The main, own-radius shuffle, fixed-reference shuffle and sign-only experiments have 200 repeats; normalization ablations have 100. `src` records chunks; it is not an independent biological sample. `shuffle_id` identifies the repeat but seeds provide the checked cross-configuration identity. Building a reservoir resets its RNG at the same seed for each grid point, keeping the underlying randomized network/input-weight realization paired across the grid. Bias uses a separate fixed seed. Updated KR input streams use explicit per-repeat offsets; MC/IPC and network variation remain bundled in the recorded seed.\n")
    text.append("For each architecture, metric and normalization, the balanced matrix Y(seed,configuration) is decomposed into grand mean, configuration main effect, seed main effect, and residual. Population variances sum exactly. Between-configuration variance is variance of configuration means; within-configuration variance is the mean variance across seeds, equal to seed plus residual components. One observation per cell means the residual combines seed×configuration interaction with evaluation noise. These data cannot identify those two sources separately, or separate biological wiring uncertainty, randomized input weights and stimulus sampling.\n")
    v=t["variance_components"].query("dataset == 'main'").copy()
    for x in ("configuration_variance","seed_variance","interaction_noise_variance"):
        v[x]=v[x]/v.total_variance
    text.append(markdown(v,["architecture","metric","configuration_variance","seed_variance","interaction_noise_variance"]))
    text.append("`tables/per_realization.csv` computes robustness before averaging; `tables/averaged_surface.csv` instead summarizes the performance surface averaged across seeds. These are distinct estimands: averaging can smooth an individually sensitive surface. Bootstrap intervals resample seed blocks, keeping all architectures/sign conditions that share a seed together for pooled correlations; no hyperparameter is treated as a biological replicate. Architecture mean intervals use independent seed draws within each marginal architecture summary. Correlation intervals condition on the selected architecture panel and grid; architecture-centroid and partial-rank correlations are descriptive without sampling intervals.\n")
    text.append("### How much can spectral normalization explain?\n\nThe stored sign-normalization experiments provide own-radius and fixed-reference scaling on the full grid. Own-radius scaling uses target/ρ(W); reference scaling uses target/ρ(W_reference), allowing actual post-scaling radius to vary. The parameter called sign-flip fraction is the requested **negative-edge fraction**: the implementation takes absolute weights and assigns random edge signs. It does not selectively perturb identified inhibitory neurons. At the empirical fraction it randomizes sign locations, so that point is not the unperturbed connectome.\n")
    text.append("We compare modes **within each archival normalization experiment**, retaining its KR vintage. We do not substitute updated sign-only KR values into these contrasts. The updated main/shuffle KR and archival control KR are not proven protocol-compatible; cross-run shuffle normalization contrasts therefore use MC/IPC only. Even MC/IPC can differ across runs at unchanged scaling, so these contrasts include evaluation/run effects. Correlations with raw radius, scale factor or E/I fraction do not isolate any one mechanism.\n")
    norm=t["normalization_differences"]
    n=norm[norm.measure.isin(["mean","sd","bottom10"])].copy()
    # Show representative endpoints and reference-near middle conditions; all conditions in CSV.
    n=n[n.architecture.str.endswith(("0.0","0.4","1.0")) & n.dataset.str.startswith("norm_")]
    text.append(markdown(n,["dataset","architecture","metric","measure","own_minus_fixed","ci_low","ci_high","n_paired_seeds"]))
    maxerr=t["spectral_per_seed"].max_scaling_identity_error.max()
    text.append(f"Maximum stored normalization-factor identity error is {maxerr:.3g}. Full raw radius, factor, post-radius and robustness diagnostics appear in the spectral figures and tables. No truly unnormalized performance control was identified among the curated merged tables. Fixed-reference controls exist and suffice to test whether the original conclusion depends on changing rescaling. The data support reporting the magnitude and direction of this dependence, not a causal percentage of the total performance–robustness association explained.\n")
    text.append("### What remains supported about C. elegans?\n\nThe stored experiments support architecture-specific differences in simulated MC, the implemented IPC and KR under the chosen dynamics, input/readout protocol and grid. The comparison table identifies whether C. elegans improves absolute mean, lower-tail capability or threshold attainment against each null model. A claim of a universal biological performance–robustness tradeoff requires more than a positive CV correlation, particularly if higher-performing models retain better lower-tail performance. These are randomized computational realizations of one anatomical connectome, not independent worms or direct measurements of in vivo computation. Effective-rank differences are not by themselves task accuracy or biological adaptation.\n")
    text.append("## Definitions and reproducibility\n\nAll configurations receive equal weight. SD uses ddof=0; IQR is Q75−Q25; MAD is the unscaled median absolute deviation. Quantiles use linear interpolation. Worst-decile mean uses the lowest ceil(0.1n) values: 10 of 96, an explicit finite-grid convention. Minimum, p10, maximum and range are also reported. CV is undefined for a nonpositive mean; 80%/90% retention is undefined for a nonpositive optimum. Constant positive surfaces have zero spread and full retention; all-zero surfaces have zero absolute spread and undefined relative metrics. Nonfinite scores are excluded and counted, negative scores are retained in absolute metrics and flagged. Duplicate identities fail rather than being silently averaged. Variance partitioning is omitted if cells are missing.\n")
    text.append("Local sensitivity uses only adjacent design levels with all other coordinates unchanged. Radius and input scale use log coordinates; leak and bias use linear coordinates, including zero bias. Each dimension is normalized to its full grid span [0,1]. We report mean and p90 slopes per dimension, and an edge-weighted mean across dimensions for the overview scatterplots. Values are performance units per normalized coordinate unit; they are not comparable across MC/IPC/KR scales. A missing intermediate level is not bridged within a realization. Performance range and minima are descriptive and sensitive to extreme observations. Main and shuffle panels share some architectures but are analyzed separately, never pooled as independent repeats.\n")
    text.append("Thresholds are identical across all architectures/conditions within a metric. Anchor thresholds are main pooled Q25/Q50/Q75, with the main grid equally represented for each architecture. Curves cover zero through the largest stored finite score on 81 equally spaced levels plus anchors and an upper endpoint. All are exploratory; none is claimed to be independently useful. Raw thresholds are never pooled across metrics.\n")
    threshold_rows=[dict(metric=metric,**data["anchors"]) for metric,data in settings["thresholds"].items()]
    text.append(markdown(pd.DataFrame(threshold_rows)))
    text.append(f"The analysis uses {settings['bootstrap']} percentile-bootstrap replicates at 95% confidence and RNG seed {settings['seed']}. Numerous exploratory associations are reported without multiplicity-adjusted hypothesis tests. Do not read their intervals as confirmatory evidence selected independently of the data.\n")
    text.append("## Files\n\n- `manifest.json`: exact source paths, SHA256 hashes, code hashes, environment, filtering, coordinates, thresholds and invocation.\n- `tables/comparison.csv`: every robustness metric with seed-bootstrap mean intervals.\n- `tables/per_realization.csv`, `averaged_surface.csv`, `variance_components.csv`: separate estimands and finite-design decomposition.\n- `tables/associations.csv`: pooled, within-architecture, architecture-centroid and floor-controlled associations.\n- `tables/mean_matched.csv` and `mean_matched_pairs.csv`: overlap, calipers, observation counts and selected seeds.\n- `tables/threshold_curves.csv`: absolute threshold curves with intervals.\n- `tables/spectral_per_seed.csv`, `normalization_differences.csv`, `spectral_associations.csv`: normalization diagnostics.\n- `tables/data_audit.csv`: completeness, bounds, missing/negative values by architecture and metric.\n- `figures/`: publication-resolution PNG and vector PDF figures, preserving thesis sign-balance groups/colors.\n- `cache/`: fingerprinted extraction tables and compressed seed×configuration matrices.\n\nRegenerate in a new directory with `python tools/reevaluate_robustness.py --out-dir analysis/robustness_new`; add `--resume` only for this analysis's own directory. Run synthetic checks with `python -m unittest discover -s tests -p 'test_robustness.py'`.\n")
    text.append("## Methodological references\n\nThe implemented IPC should be distinguished from the broader capacity framework in [Dambre et al. (2012)](https://www.nature.com/articles/srep00514). The analysis uses Spearman rank association and seed-block percentile bootstrap intervals; see the official [SciPy Spearman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) and [bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for the underlying statistical definitions. The cluster resampling and finite-design variance partition are implemented explicitly here. Bounds follow directly from the inspected repository score definitions and the inequality derived above.\n")
    (out/"METHODS_AND_RESULTS.md").write_text("\n".join(text))
    # Compact, readable companion to the full machine-readable comparison.
    brief=main[["architecture","metric","mean","cv","sd","iqr","mad","minimum","p10","bottom10","attain_q25","attain_q50","attain_q75","retention80","retention90","range","sensitivity_mean"]]
    (out/"COMPARISON.md").write_text("# Main experiment robustness comparison\n\nValues are means of per-realization metrics; see comparison.csv for bootstrap intervals.\n\n"+markdown(brief))
    concise_report(t,settings,out)


def concise_report(t,settings,out):
    """Keep the primary scientific conclusions short; full detail is separate."""
    a=t["associations"]
    c=t["comparison"]
    main=c[c.dataset=="main"]
    pooled=a[(a.dataset=="main")&(a.scope=="pooled_realizations")]
    corr=[]
    for metric in ("MC","IPC","KR"):
        g=pooled[pooled.metric==metric].set_index("measure")
        corr.append(dict(metric=metric,**{measure:f"{g.loc[measure,'rho']:.3f} [{g.loc[measure,'ci_low']:.3f}, {g.loc[measure,'ci_high']:.3f}]"
                        for measure in ("cv","sd","iqr","mad","bottom10","attain_q50")}))
    lines=["# Reevaluate the performance–robustness tradeoff\n",
        "**The positive performance–dispersion association survives removing CV. The broader claim that higher-performing reservoirs have less reliable computational capability does not follow:** higher-performing reservoirs also have better lower-tail performance and attain common absolute thresholds more often. Normalization substantially changes the sign-sweep results.\n",
        "All experiment data came from nine curated merged tables in `final_results/`. No simulations were launched. New outputs are isolated here, and all input SHA256 hashes were verified unchanged.\n",
        "## Does the association persist without CV?\n\nMain experiment Spearman correlations, with 95% seed-cluster bootstrap intervals:\n",
        markdown(pd.DataFrame(corr)),
        "The own-radius shuffle panel gives the same qualitative result; its SD correlations with mean are "+", ".join(f"{metric} {a[(a.dataset=='shuf')&(a.metric==metric)&(a.measure=='sd')&(a.scope=='pooled_realizations')].iloc[0].rho:.3f}" for metric in ("MC","IPC","KR"))+". Thus CV's denominator alone cannot explain the observed positive association. SD restricted to configurations above the exploratory main Q25 threshold also remains positively associated with mean (minimum 20 surviving configurations). This diagnostic changes the configuration subset and does not isolate a floor mechanism.\n",
        "## Are weak reservoirs being called robust?\n\nYes, a small spread can reward consistently weak performance. In the main experiment, the binary architecture has the lowest mean and SD for all three metrics. Its bottom-decile means, compared with the best mean-performing architecture (`ws_p0.1+randN`), are:\n"]
    within=a[(a.dataset=="main")&(a.scope=="within_architecture")&(a.architecture=="real")&(a.measure=="sd")]
    lines.insert(-1,"Within the C. elegans architecture, the mean–SD correlations across seeds are "+"; ".join(f"{r.metric} {r.rho:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]" for _,r in within.iterrows())+". The very strong pooled associations therefore also reflect differences between architectures; they are not a universal within-architecture law.\n")
    rows=[]
    for metric in ("MC","IPC","KR"):
        g=main[main.metric==metric].set_index("architecture")
        low,high=g.loc["binary_base"],g.loc["ws_p0.1+randN"]
        rows.append(dict(metric=metric,binary_bottom10=low.bottom10,WS_bottom10=high.bottom10,
                         binary_threshold_fraction=low.attain_q50,WS_threshold_fraction=high.attain_q50))
    lines.append(markdown(pd.DataFrame(rows)))
    lines.append("Threshold fractions use an identical main pooled median threshold within each metric. All thresholds are exploratory; no independently justified useful-performance cutoffs were available. Retention at 80% and 90% of each realization's maximum is reported alongside absolute performance and cannot establish useful capability on its own. A floor need not lower CV, since its denominator also shrinks.\n")
    lines.append("## Which C. elegans comparisons remain supported?\n\nC. elegans exceeds the binary architecture in mean, bottom-decile performance and median-threshold attainment for MC, IPC and KR. Gaussian-weight CE, ER and WS models exceed C. elegans on all three of those capability summaries, while showing greater SD/IQR/MAD. The results therefore support neither universal C. elegans superiority nor a general loss of reliable capability in the higher-performing models. C. elegans has better lower-tail MC and IPC than several sign-preserving controls despite a lower mean; the complete comparison is in [COMPARISON.md](COMPARISON.md) and `tables/comparison.csv`.\n")
    lines.append("In the shuffle panel, weight shuffle, connection shuffle and joint weight/connection shuffle all have higher estimated mean and bottom-decile MC, IPC and KR than the intact weighted connectome, alongside greater absolute dispersion. Thus preserving the biological weight/topology arrangement is not uniformly advantageous under this protocol. Paired seed-bootstrap differences for C. elegans versus each alternative, including all these comparisons, are in `tables/architecture_contrasts.csv`.\n")
    m=t["mean_matched"]
    counts=m[m.measure=="mean"]
    lines.append(f"Mean matching is limited by overlap: {int((counts.n_matched==0).sum())}/{len(counts)} architecture-pair/metric comparisons have no matches, and {int((counts.n_matched>=20).sum())}/{len(counts)} have at least 20. We restrict to empirical overlapping mean ranges, then match without replacement within a 5%-of-combined-span caliper. Counts and residual mean differences are recorded for every pair. Conditioning on mean is descriptive and does not establish causality.\n")
    details=[]
    for metric in ("MC","IPC","KR"):
        for dataset in ("main","shuf"):
            for measure in ("sd","iqr","mad","bottom10"):
                g=m[(m.dataset==dataset)&(m.metric==metric)&(m.measure==measure)&(m.n_matched>=20)]
                details.append(dict(dataset=dataset,metric=metric,measure=measure,n_pairs=len(g),
                    same_difference_sign=int((np.sign(g.matched_difference_a_minus_b)==np.sign(g.full_difference_a_minus_b)).sum())))
    pd.DataFrame(details).to_csv(out/"tables/mean_matched_sign_consistency.csv",index=False)
    lines.append("Mean-matched differences can change sign; consult `tables/mean_matched.csv` and `mean_matched_sign_consistency.csv` before generalizing an ordering. A pooled positive correlation does not imply a universal comparison at equal mean performance.\n")
    examples=[]
    for metric in ("MC","IPC","KR"):
        g=m[(m.dataset=="main")&(m.metric==metric)&(m.architecture_a=="local_sign+binary")&(m.architecture_b=="real")]
        if len(g):
            row=g[g.measure=="mean"].iloc[0]
            examples.append(f"{metric}: {int(row.n_matched)} matched pairs")
    lines.append("One comparison is consistent at comparable means: C. elegans has lower estimated SD, IQR and MAD, and a higher bottom-decile mean, than the sign-preserving ±1 architecture ("+"; ".join(examples)+"). These are conditional descriptive comparisons, with sparse or absent overlap preventing many other architecture comparisons. C. elegans and the Gaussian CE/ER/WS architectures have no overlapping mean ranges for any metric.\n")
    lines.append("## How much depends on spectral normalization?\n\nFixed-reference controls already exist. For the sign-matched connectome at negative-edge fraction 0.4, the following are own-radius minus fixed-reference differences from paired seeds within the same archival experiment:\n")
    n=t["normalization_differences"]
    g=n[(n.dataset=="norm_cel_matched")&(n.architecture=="sign_test_og_cel0.4")&n.measure.isin(["mean","sd","bottom10"])]
    lines.append(markdown(g,["metric","measure","own_minus_fixed","ci_low","ci_high","n_paired_seeds"]))
    lines.append("Own-radius scaling amplifies changes near balanced signs because it divides by the altered raw radius; fixed-reference scaling changes that result substantially. This establishes dependence on the normalization protocol, not a causal percentage explained by raw radius. Sign structure, effective recurrent gain and other dynamics change together. The sweep assigns a negative-edge fraction after taking absolute weights, rather than selectively flipping known inhibitory neurons. At the empirical fraction it is still a randomized sign assignment.\n")
    lines.append("KR normalization contrasts stay within the archival control tables: their exact historical rank protocol is not embedded in the CSVs and differs in provenance from the updated main/sign-only KR runs. Cross-run shuffle normalization contrasts therefore use MC/IPC only. Unchanged-scaling baselines also exhibit run variation. These limitations prevent a clean attribution of the total association to spectral normalization. No curated unnormalized performance control was identified.\n")
    lines.append("## Seed variability, bounds and limits\n\nThe full design is 4 radii × 3 leaks × 4 input scales × 2 bias ranges = 96 configurations. There are 200 seeds per architecture in main, shuffle and sign-only tables, and 100 per sign-normalization condition. Robustness is computed per seed before averaging; robustness of the configuration-averaged surface is a separate table. Exact balanced-design variance partitioning separates configuration effects, seed effects and residual interaction plus noise. One observation per seed/configuration prevents separating interaction from evaluation noise or separating network, input and stimulus randomness. Seeds are computational repeats of one connectome, not independent biological networks.\n")
    v=t["variance_components"].query("dataset == 'main'")
    ranges=[]
    for metric in ("MC","IPC","KR"):
        h=v[v.metric==metric]
        frac=h.configuration_variance/h.total_variance
        ranges.append(f"{metric} {frac.min():.1%}–{frac.max():.1%}")
    lines.append("Configuration main effects account for "+"; ".join(ranges)+" of finite-design total variance across main architectures. Residual and seed components remain architecture-dependent. These are descriptive partitions, not identified biological variance components.\n")
    lines.append("Bounds follow from the inspected implementation: MC ≤30, restricted IPC ≤150, and KR ≤297. Historical delay/order/stream metadata are absent, so archived upper-bound diagnostics are conditional on the documented protocol. No selected finite observation violates these bounds. Variance is compared with the feasible ceiling (μ−a)(b−μ), not an assumed expected variance. The ceiling-normalized dispersion remains associated with mean, but cannot establish intrinsic sensitivity.\n")
    lines.append("Local sensitivity uses adjacent configurations differing in one dimension, with log radius/input scale and linear leak/bias coordinates scaled to [0,1]. Both mean and p90 sensitivity are reported. Threshold curves, all dispersion definitions, lower tails, retention, range, missing-value counts and exact seed matches are retained in the tables.\n")
    lines.append("## Scientific conclusion\n\nReplace the unqualified performance–robustness tradeoff with a narrower statement: **higher mean performance is associated with larger absolute hyperparameter variation on this grid, while useful lower-tail capability often improves at the same time.** Topology, weights and sign structure affect the simulated computation, and the size and direction of sign-sweep effects depend on normalization. These observations do not establish evolutionary optimization, in vivo computational superiority, or an intrinsic causal cost of performance.\n")
    lines.append(f"## Reproduce and inspect\n\n- [Full methods, numerical comparisons and limitations](METHODS_AND_RESULTS.md)\n- [Compact main comparison](COMPARISON.md)\n- [Compact performance–dispersion–lower-tail comparison](figures/main_key_comparison.pdf)\n- [All robustness definitions](figures/main_mean_vs_robustness.pdf)\n- [Absolute threshold curves](figures/main_threshold_attainment.pdf)\n- [Local sensitivity](figures/main_local_sensitivity.pdf)\n- [Normalization diagnostics](figures/norm_spectral_balance_diagnostics.pdf)\n\n`manifest.json` records all sources, hashes, settings and package versions. Intervals use {settings['bootstrap']} seed-cluster bootstrap draws, RNG seed {settings['seed']}, conditional on the chosen architecture panel/grid. All analyses are exploratory. Regenerate with `python tools/reevaluate_robustness.py --out-dir analysis/robustness_new`; synthetic tests: `python -m unittest discover -s tests -p 'test_robustness.py'`.\n")
    (out/"REPORT.md").write_text("\n".join(lines))
