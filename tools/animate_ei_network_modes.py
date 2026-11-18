#!/usr/bin/env python3
"""Side-by-side signed networks, driven activity, spectra and PCA diagnostics.

Uses the repository's reservoir construction/dynamics/layout and standard
scikit-learn PCA. This is an illustrative trajectory, not a performance sweep.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reservoir-video")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/reservoir-video-xdg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
import torch

from architecture_variant_figures.make_weighted_graph_examples import compute_ce_kamada_layout
from mc_narma_crossover import build_condition, load_completed_connectome
from network_stats.run_one import run_reservoir_states

FAMILIES = ("cel_matched", "matched_er")
NAMES = ("C. elegans · unknown signs matched", "Matched ER · binary weights")
FRACTIONS = (0., .2, .4, .6, .8, 1.)
EDGE_POSITIVE = "#d87535"
EDGE_NEGATIVE = "#2876b4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_edge_widths(weights, max_width=2.4):
    """Widths proportional to |W_post| / max_condition,model(|W_post|).

    No additive minimum, quantile clipping or cross-panel rescaling is applied.
    Any positive global matrix-scaling factor cancels exactly in this display.
    """
    a = np.abs(np.asarray(weights, dtype=float))
    if not np.isfinite(a).all() or not np.isfinite(max_width) or max_width <= 0:
        raise ValueError("Require finite weights and positive maximum width.")
    largest = a.max() if a.size else 0.
    return a * (max_width / largest) if largest > 0 else np.zeros_like(a)


def activity_modes(states):
    """Centered variance spectrum and participation ratio, using sklearn PCA."""
    x = np.asarray(states, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Require a finite time-by-node activity array.")
    if np.max(np.var(x, axis=0)) == 0:
        return np.zeros(min(x.shape)), 0.
    pca = PCA(svd_solver="full").fit(x)
    fractions = pca.explained_variance_ratio_
    return fractions, float(1. / np.sum(fractions ** 2))


def generate(args):
    torch.set_num_threads(1)
    known, unknown = ROOT / "Connectome/ce_adj.npy", ROOT / "Connectome/ce_unknown_sign_weights.npy"
    before = {str(p.relative_to(ROOT)): digest(p) for p in (known, unknown)}
    completed, fraction_unknown = load_completed_connectome(known, unknown, args.seed)
    input_seed = args.seed + 31_000_003
    inputs = np.random.default_rng(input_seed).uniform(-1, 1, args.washout + args.analysis_steps).astype(np.float32)[:, None]
    u = torch.from_numpy(inputs)
    networks, states, eigenvalues, variances, records = [], [], [], [], []
    for family in FAMILIES:
        model_networks, model_states, model_eigenvalues, model_variances = [], [], [], []
        for negative_fraction in FRACTIONS:
            base, win, info = build_condition(completed, family, negative_fraction, args.normalization, args.seed, torch.device("cpu"))
            w = base * args.rho
            win = win * args.input_scale
            trajectory = run_reservoir_states(w, win, u, args.leak)[args.washout:].numpy()
            matrix = w.numpy()
            eig = np.linalg.eigvals(matrix.astype(np.float64))
            ratios, participation = activity_modes(trajectory)
            moduli = np.sort(np.abs(eig))[::-1]
            model_networks.append(matrix)
            model_states.append(trajectory)
            model_eigenvalues.append(eig)
            model_variances.append(ratios)
            records.append(dict(
                family=family, ei_percent=int(round(negative_fraction * 100)), seed=args.seed,
                normalization=args.normalization, rho_target=args.rho, leak=args.leak, input_scale=args.input_scale,
                raw_rho=info["raw_rho"], post_rho=float(moduli[0]), scale_factor=float(info["scale_factor"]) * args.rho,
                n_edges=int(np.count_nonzero(matrix)), actual_negative_fraction=float(np.mean(matrix[matrix != 0] < 0)),
                max_post_abs_weight=float(np.max(abs(matrix))), median_post_abs_weight=float(np.median(abs(matrix[matrix != 0]))),
                pc1_variance_fraction=float(ratios[0]), participation_ratio=participation,
                second_to_first_eigenvalue_modulus=float(moduli[1] / moduli[0]),
                fraction_state_abs_above_09=float(np.mean(abs(trajectory) > .9)),
            ))
            print(f"[condition] {family} {negative_fraction:.0%}: PC1={ratios[0]:.3f}; PR={participation:.2f}", flush=True)
        networks.append(model_networks)
        states.append(model_states)
        eigenvalues.append(model_eigenvalues)
        variances.append(model_variances)
    arrays = dict(weights=np.asarray(networks), states=np.asarray(states), eigenvalues=np.asarray(eigenvalues),
                  variance_fractions=np.asarray(variances), inputs=inputs)
    layouts = []
    for f in range(2):
        # Same positive-magnitude layout across all E/I conditions in each panel.
        pos = compute_ce_kamada_layout(arrays["weights"][f, 0], seed=args.seed,
                                      fallback_iters=150, layout_scale=1.)
        layouts.append(np.asarray([pos[i] for i in range(len(completed))]))
    arrays["layouts"] = np.asarray(layouts)
    for path, value in before.items():
        if digest(ROOT / path) != value:
            raise ValueError("Connectome source changed during rendering.")
    return arrays, pd.DataFrame(records), before, fraction_unknown


def render(args, arrays, diagnostics):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor="#fafbfd")
    title = fig.text(.5, .971, "", ha="center", va="top", fontsize=22, weight="semibold", color="#182c46")
    normalization = "Own-radius" if args.normalization == "spectral_radius" else "Fixed-reference"
    fig.text(.5, .923, f"{normalization} scaling · target {args.rho:g} · leak {args.leak:g} · input scale {args.input_scale:g} · bias 0",
             ha="center", fontsize=12, color="#4c5b70")
    fig.text(.5, .892, "Same input sequence and input weights · fixed layout within each architecture · all 3,604 directed connections",
             ha="center", fontsize=10, color="#4c5b70")
    graph_axes = [fig.add_axes([.025, .345, .455, .50]), fig.add_axes([.52, .345, .455, .50])]
    spectrum_axes = [fig.add_axes([.055, .119, .17, .17]), fig.add_axes([.55, .119, .17, .17])]
    pca_axes = [fig.add_axes([.285, .119, .19, .17]), fig.add_axes([.78, .119, .19, .17])]
    summary_text = []
    for f, ax in enumerate(graph_axes):
        ax.set_title(NAMES[f], fontsize=15, color="#332288" if f == 0 else "#0077bb", pad=12)
        ax.axis("off")
        pos = arrays["layouts"][f]
        span = np.ptp(pos, axis=0)
        ax.set_xlim(pos[:, 0].min() - span[0]*.04, pos[:, 0].max() + span[0]*.04)
        ax.set_ylim(pos[:, 1].min() - span[1]*.04, pos[:, 1].max() + span[1]*.04)
        summary_text.append(fig.text(.252 if f == 0 else .747, .315, "", ha="center", fontsize=11, color="#26364d"))
    edge_artists, node_artists = [], []
    edge_indices = []
    # One shared activity color scale across both models and all six conditions.
    vmax = max(.1, np.ceil(np.max(abs(arrays["states"])) * 10) / 10)
    for f, ax in enumerate(graph_axes):
        indices = np.argwhere(arrays["weights"][f, 0] != 0)
        edge_indices.append(indices)
        pos = arrays["layouts"][f]
        segments = pos[indices]
        lines = LineCollection(segments, linewidths=.5, alpha=.12, zorder=1)
        ax.add_collection(lines)
        nodes = ax.scatter(pos[:, 0], pos[:, 1], s=23, c=np.zeros(len(pos)),
                           cmap="PRGn", vmin=-vmax, vmax=vmax, edgecolors="#424b58", linewidths=.25, zorder=3)
        edge_artists.append(lines)
        node_artists.append(nodes)
    cax = fig.add_axes([.484, .405, .011, .37])
    bar = fig.colorbar(node_artists[0], cax=cax)
    bar.set_label("Node state z(t) · fixed shared scale", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    input_text = fig.text(.5, .066, "", ha="center", fontsize=11, color="#26364d")
    fig.legend(handles=[Line2D([], [], color=EDGE_POSITIVE, lw=2, label="Positive edge"),
                        Line2D([], [], color=EDGE_NEGATIVE, lw=2, label="Negative edge")],
               loc="lower center", bbox_to_anchor=(.5, .014), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .008, "Width = 2.4 pt × |post-scaled weight| / panel maximum · each model/EI panel normalized separately · illustrative realization, not a task score",
             ha="center", fontsize=8, color="#526176")
    frames_per_condition = int(round(args.seconds_per_condition * args.fps))
    if frames_per_condition * args.steps_per_frame > args.analysis_steps:
        raise ValueError("Video asks for more displayed states than the analysis trajectory contains.")
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                         metadata={"title": "CE versus ER: E/I, normalized weights and activity modes", "artist": "Miles Churchland"},
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-preset", "fast", "-movflags", "+faststart"])
    filename = args.out_dir / "ce_er_ei_activity_modes.mp4"
    snapshot_paths = []
    with writer.saving(fig, str(filename), dpi=100):
        for c, fraction in enumerate(FRACTIONS):
            percent = int(round(fraction * 100))
            title.set_text(f"{percent}% inhibitory edges")
            for f in range(2):
                weights = arrays["weights"][f, c]
                indices = edge_indices[f]
                selected_weights = weights[indices[:, 0], indices[:, 1]]
                edge_artists[f].set_color(np.where(selected_weights[:, None] >= 0,
                                                   matplotlib.colors.to_rgb(EDGE_POSITIVE), matplotlib.colors.to_rgb(EDGE_NEGATIVE)))
                edge_artists[f].set_linewidth(relative_edge_widths(weights, args.max_edge_width)[indices[:, 0], indices[:, 1]])
                row = diagnostics[(diagnostics.family == FAMILIES[f]) & (diagnostics.ei_percent == percent)].iloc[0]
                summary_text[f].set_text(f"PC1: {row.pc1_variance_fraction:.0%} of activity variance   |   effective activity dimension: {row.participation_ratio:.2f}")
                ax = spectrum_axes[f]
                ax.clear()
                e = arrays["eigenvalues"][f, c]
                ax.scatter(e.real, e.imag, s=5, color="#677b91", alpha=.6)
                leading = np.argmax(abs(e))
                ax.scatter([e[leading].real], [e[leading].imag], s=28, color="#d87535", zorder=3)
                ax.add_patch(plt.Circle((0, 0), args.rho, fill=False, ls="--", lw=.7, color="#aeb8c4"))
                ax.axhline(0, color="#d9dfe6", lw=.5)
                ax.axvline(0, color="#d9dfe6", lw=.5)
                limit = max(1.05, float(abs(e).max()) * 1.12)
                # Fixed axes for all conditions under the default own-radius protocol.
                if args.normalization == "spectral_radius":
                    limit = max(1.05, args.rho * 1.12)
                ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Re(λ)", ylabel="Im(λ)",
                       title=f"Eigenvalues · raw ρ = {row.raw_rho:.1f}")
                ax.set_aspect("equal")
                ax.tick_params(labelsize=8)
                ax.title.set_fontsize(9)
                ax.xaxis.label.set_fontsize(8)
                ax.yaxis.label.set_fontsize(8)
                ax = pca_axes[f]
                ax.clear()
                v = arrays["variance_fractions"][f, c]
                shown = np.r_[v[:8], v[8:].sum()]
                ax.bar(np.arange(9), shown * 100, color=["#332288" if f == 0 else "#0077bb"] * 8 + ["#c2cbd6"])
                ax.set(ylim=(0, 100), xticks=np.arange(9), xticklabels=[str(i) for i in range(1, 9)] + ["rest"],
                       ylabel="Variance (%)", xlabel="Principal component", title="Centered activity · 2,000-step summary")
                ax.title.set_fontsize(9)
                ax.tick_params(labelsize=8)
                ax.xaxis.label.set_fontsize(8)
                ax.yaxis.label.set_fontsize(8)
            for frame in range(frames_per_condition):
                step = frame * args.steps_per_frame
                for f in range(2):
                    node_artists[f].set_array(arrays["states"][f, c, step])
                input_text.set_text(f"Shared input U[−1, 1]: u(t) = {arrays['inputs'][args.washout + step, 0]:+.3f}   |   t = {args.washout + step}   |   condition {c+1}/6")
                if frame == frames_per_condition // 2:
                    snapshot = args.out_dir / f"ei_{percent:03d}_snapshot.png"
                    fig.savefig(snapshot, dpi=120, facecolor=fig.get_facecolor())
                    snapshot_paths.append(snapshot)
                writer.grab_frame(facecolor=fig.get_facecolor())
            print(f"[rendered] {percent}% ({frames_per_condition} frames)", flush=True)
    plt.close(fig)
    # A compact companion plot makes the activity-mode comparison easy to read.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for f, family in enumerate(FAMILIES):
        g = diagnostics[diagnostics.family == family]
        color = "#332288" if f == 0 else "#0077bb"
        axes[0].plot(g.ei_percent, g.pc1_variance_fraction*100, "o-", color=color, label="CE" if f == 0 else "ER")
        axes[1].plot(g.ei_percent, g.participation_ratio, "o-", color=color)
    for ax in axes:
        ax.set_xlabel("Inhibitory edges (%)")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#e6e9ee")
    axes[0].set_ylabel("Variance in PC1 (%)")
    axes[1].set_ylabel("Activity participation ratio")
    axes[0].legend(frameon=False)
    fig.suptitle("One paired realization · centered driven activity · not a task-performance score", fontsize=11)
    fig.savefig(args.out_dir / "mode_diagnostics.png", dpi=220)
    plt.close(fig)
    return filename


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis/ei_network_mode_video")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--rho", type=float, default=.95)
    p.add_argument("--leak", type=float, default=1.)
    p.add_argument("--input-scale", type=float, default=.1)
    p.add_argument("--normalization", choices=["spectral_radius", "original_radius"], default="spectral_radius")
    p.add_argument("--washout", type=int, default=500)
    p.add_argument("--analysis-steps", type=int, default=2000)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seconds-per-condition", type=float, default=8.)
    p.add_argument("--steps-per-frame", type=int, default=2)
    p.add_argument("--max-edge-width", type=float, default=2.4)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    arrays, diagnostics, sources, unknown_fraction = generate(args)
    diagnostics.to_csv(args.out_dir / "mode_diagnostics.csv", index=False)
    np.savez_compressed(args.out_dir / "networks_and_trajectories.npz", **arrays)
    filename = render(args, arrays, diagnostics)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = dict(settings=settings, fractions=FRACTIONS, sources=sources,
                    unknown_sign_inhibitory_fraction=unknown_fraction,
                    code={name: digest(ROOT/name) for name in ["tools/animate_ei_network_modes.py", "mc_narma_crossover.py", "util/util.py", "network_stats/run_one.py", "architecture_variant_figures/make_weighted_graph_examples.py"]},
                    packages=dict(numpy=np.__version__, torch=torch.__version__, matplotlib=matplotlib.__version__, sklearn=sklearn.__version__, networkx=nx.__version__),
                    width_rule="max_width * abs(W_post) / max(abs(W_post)), separately per model and fraction; no minimum width",
                    activity_color="one shared fixed scale across models and fractions",
                    pca="sklearn PCA on mean-centered post-washout activity; PR=1/sum(explained_variance_ratio**2)",
                    inference="One illustrative realization at one parameter setting; no uncertainty interval or task score")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "README.md").write_text(
        "# CE/ER E/I network and activity video\n\n"
        "Open `ce_er_ei_activity_modes.mp4`. Six 8-second conditions show 0, 20, 40, 60, 80 and 100% negative edges. "
        "Both models have 297 nodes and 3,604 directed edges. CE restores unknown edges at the known-edge negative proportion; ER is the existing binary-weight matched sign-sweep control.\n\n"
        "Edges: orange positive, blue negative. Thickness is proportional to absolute post-scaling weight, divided by that model/condition's maximum. "
        "All connections are drawn; arrowheads are omitted for readability. Positions are weighted Kamada–Kawai layouts from the existing repository helper, fixed within each architecture across conditions.\n\n"
        "**Width normalization cancels global spectral-radius scaling.** Consequently, CE's relative widths stay fixed across fractions and all ER widths are equal. "
        "The panels do not compare absolute recurrent strength. Exact post-scaled matrices, maximum weights and scale factors are saved.\n\n"
        "Nodes show actual driven state, on a single shared color scale. Every condition starts at zero, receives the same U[-1,1] sequence and undergoes 500 washout steps. "
        "No task readout is fitted. PCA summaries use all 2,000 post-washout samples; the video displays the first 240 of those at two model steps per frame. "
        "Network/input realization seed is 20261117; target radius 0.95, leak 1, input scale 0.1 and reservoir bias 0. "
        "The default movie uses own-radius normalization.\n\n"
        "The eigenvalue plots show the normalized recurrent matrix, with its largest-modulus eigenvalue highlighted. "
        "PCA uses mean-centered activity, not the recurrent-weight eigenvectors. The participation ratio measures variance dimensionality; it does not establish how many modes are computationally useful.\n\n"
        "This is a **single illustrative realization**, not a statistical test or evidence of biological optimization. "
        "The displayed data support greater CE activity dimensionality at 0%, 20% and 100%, and greater ER dimensionality at 40%, 60% and 80%. "
        "The 80% case is a counterexample to a universal CE advantage at sign-imbalanced fractions.\n\n"
        "Reproduce: `python tools/animate_ei_network_modes.py --out-dir analysis/ei_network_mode_video_new`. "
        "Settings, hashes and library versions are in `manifest.json`; diagnostics are also exported as PNG and CSV.\n"
    )
    print("Saved", filename, flush=True)


if __name__ == "__main__":
    main()
