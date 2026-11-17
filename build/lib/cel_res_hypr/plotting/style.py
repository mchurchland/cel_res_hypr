"""Style for reservoir research figures."""

import matplotlib as mpl
import numpy as np
import re


PERFORMANCE_METRICS = ("IPC", "KR", "MC")


SHORT_THESIS_NAMES = {
    "real": "C. elegans",
    "shuffle": "Wt. shuffle",
    "shuffle_weights": "Wt. shuffle",
    "celW+connShuf": "Conn. + wt. shuf.",
    "conn_shuf": "Conn. + wt. shuf.",
    "conn_shuf_only": "Conn. shuffle",
    "local_sign+binary": "Sign-pres. pm1",
    "global_sign_pres": "pm1 + wt. shuf.",
    "global_sign_pres_real_w": "Sign-pres. real wt.",
    "global_sign_pres_real_weight": "Sign-pres. real wt.",
    "binary+conshuffle+wshuffle": "pm1 sign/conn/wt shuf.",
    "binary_base": "Binary wt.",
    "binary_base_topology_shuffle": "Binary wt. shuf.",
    "binary+shuffle": "pm1 + conn. shuf.",
    "cel_randN": "CE Gaussian wt.",
    "er_randN": "ER Gaussian wt.",
    "ws_p0.1+randN": "WS Gaussian wt.",
    "ws_p01_randN": "WS Gaussian wt.",
    "local_sign": "Sign-pres. Gaussian",
    "local_sign+flat": "Sign-pres. uniform",
}


_SHORT_THESIS_NAME_ALIASES = {
    "shuffle": "shuffle_weights",
    "cel+randN": "cel_randN",
    "er+randN": "er_randN",
    "ws_p0.1+randN": "ws_p01_randN",
    "celW+connShuf": "conn_shuf",
}


def _short_thesis_name(mode):
    if not isinstance(mode, str):
        return mode
    canonical = _SHORT_THESIS_NAME_ALIASES.get(mode, mode)
    return SHORT_THESIS_NAMES.get(canonical, SHORT_THESIS_NAMES.get(mode, mode))


def _short_legend_name(mode):
    if not isinstance(mode, str):
        return mode
    legend_only = {
        "shuffle_weights": "Shuffle",
        "conn_shuf": "Conn. + shuf.",
        "global_sign_pres": "pm1 + shuf.",
        "global_sign_pres_real_w": "Sign shuf.",
        "global_sign_pres_real_weight": "Sign shuf.",
        "binary+conshuffle+wshuffle": "pm1 sign/conn/shuf.",
        "binary_base": "Binary",
        "binary_base_topology_shuffle": "Binary shuf.",
        "cel_randN": "CE Gaussian",
        "er_randN": "ER Gaussian",
        "ws_p01_randN": "WS Gaussian",
        "cel_sample": "Sampled",
        "local_sign+sample": "Sign-pres. sampled",
    }
    canonical = _SHORT_THESIS_NAME_ALIASES.get(mode, mode)
    return legend_only.get(canonical, legend_only.get(mode, _short_thesis_name(mode)))


def _sign_fraction_from_mode(mode) -> float:
    """Parse sign-balance sweep fractions such as sign_test_og_cel0.2205."""
    mode_str = str(mode)
    match = re.search(r"sign_test", mode_str)
    if not match:
        return np.nan
    nums = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        mode_str[match.end() :],
    )
    if not nums:
        return np.nan
    try:
        return float(nums[0])
    except Exception:
        return np.nan


def _is_grid_tenth(value: float, atol: float = 1e-9) -> bool:
    if not np.isfinite(value):
        return False
    return abs(value * 10.0 - round(value * 10.0)) <= atol


def _detect_ce_sign_fraction(fractions) -> float | None:
    values = sorted({round(float(v), 15) for v in fractions if np.isfinite(v)})
    if not values:
        return None

    # The CE-balance point is usually inserted as an extra non-decile sample
    # within the sign sweep; prefer that data-driven marker when present.
    non_grid = [v for v in values if 0.0 < v < 0.5 and not _is_grid_tenth(v, atol=1e-7)]
    if non_grid:
        return min(non_grid, key=lambda v: abs(v - 0.22))

    # Known CE-balance points used by the current thesis runs.
    known = (0.2425287356321839, 0.06113256113256113)
    for target in known:
        for v in values:
            if abs(v - target) <= 1e-7:
                return v
    return None


_SIGN_BALANCE_COLOR_MAX = 0.5


COLOR_SCHEMES = ("thesis", "extended-abstract")


def _validate_color_scheme(color_scheme: str) -> str:
    if color_scheme not in COLOR_SCHEMES:
        raise ValueError(
            f"color_scheme must be one of {COLOR_SCHEMES}; got {color_scheme!r}."
        )
    return color_scheme


def _rho_delta_colormap(color_scheme: str):
    """Return the diverging map used for raw-spectral-radius differences."""
    _validate_color_scheme(color_scheme)
    cmap_name = "PRGn" if color_scheme == "extended-abstract" else "coolwarm"
    return mpl.colormaps[cmap_name]


def _sign_balance_color(
    frac: float,
    ce_frac: float | None,
    color_scheme: str = "thesis",
) -> str:
    """Color sign-balance modes only by negative-sign fraction."""
    if not np.isfinite(frac):
        return "#8a8a8a"
    frac = min(max(float(frac), 0.0), _SIGN_BALANCE_COLOR_MAX)
    return mpl.colors.to_hex(
        _sign_balance_colormap(ce_frac, color_scheme)(frac / _SIGN_BALANCE_COLOR_MAX)
    )


def _sign_balance_colormap(
    ce_frac: float | None = None,
    color_scheme: str = "thesis",
):
    _validate_color_scheme(color_scheme)
    ce = float(ce_frac) if ce_frac is not None and np.isfinite(ce_frac) else 0.22
    ce_pos = min(max(ce / _SIGN_BALANCE_COLOR_MAX, 0.08), 0.92)
    if color_scheme == "extended-abstract":
        anchors = [
            (0.0, "#00204C"),
            (ce_pos, "#2A9D8F"),
            (1.0, "#F6C85F"),
        ]
    else:
        anchors = [
            (0.0, "#1557D4"),
            (ce_pos, "#B72FE3"),
            (1.0, "#F2B705"),
        ]
    return mpl.colors.LinearSegmentedColormap.from_list(
        "sign_balance_0_to_50",
        anchors,
    )


def _add_sign_balance_colorbar(
    fig,
    ce_frac: float | None,
    bottom: float = 0.078,
    *,
    left: float = 0.34,
    width: float = 0.32,
    height: float = 0.012,
    labelsize: float = 8.0,
    orientation: str = "horizontal",
    color_scheme: str = "thesis",
):
    ce = float(ce_frac) if ce_frac is not None and np.isfinite(ce_frac) else 0.22
    ce = min(max(ce, 1e-6), 0.499999)
    cax = fig.add_axes([left, bottom, width, height])
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0.0, vmax=_SIGN_BALANCE_COLOR_MAX),
        cmap=_sign_balance_colormap(ce, color_scheme),
    )
    scalar.set_array([])
    cbar = fig.colorbar(scalar, cax=cax, orientation=orientation)
    cbar.set_ticks([0.0, ce, _SIGN_BALANCE_COLOR_MAX])
    ce_pct = f"{100.0 * ce:.1f}".rstrip("0").rstrip(".")
    cbar.ax.tick_params(labelsize=labelsize, length=2.4, width=0.6, pad=1)
    if orientation == "horizontal":
        cbar.set_ticklabels(["0", f"CE {ce_pct}%", "50%"])
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.set_title("E/I Edge Balance", fontsize=labelsize + 0.5, pad=12)
    else:
        cbar.set_ticklabels(["0", "CE", "50%"])
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label(
            "E/I Edge Balance", fontsize=labelsize + 2, rotation=270, labelpad=9
        )
    cbar.outline.set_linewidth(0.45)
    cbar.outline.set_edgecolor("#333333")
    return cbar


def _mode_sign_fraction(mode, ce_frac: float | None = None) -> float:
    parsed = _sign_fraction_from_mode(mode)
    if np.isfinite(parsed):
        return parsed
    ce = (
        float(ce_frac)
        if ce_frac is not None and np.isfinite(ce_frac)
        else 0.2425287356321839
    )
    group = _architecture_sign_balance_group(mode)
    if group == "zero":
        return 0.0
    if group == "ce":
        return ce
    if group == "half":
        return 0.5
    return np.nan


def _mode_marker_map(modes):
    preferred = {
        "real": "*",
        "CE-real": "*",
    }
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "8", "p", "H"]
    marker_map = {}
    marker_idx = 0
    for mode in modes:
        marker = preferred.get(str(mode))
        if marker is None:
            marker = markers[marker_idx % len(markers)]
            marker_idx += 1
        marker_map[mode] = marker
    return marker_map


def _sign_balance_label(mode, ce_frac: float | None = None) -> str:
    frac = _sign_fraction_from_mode(mode)
    if not np.isfinite(frac):
        return _short_legend_name(mode)
    pct = f"{100.0 * frac:.1f}".rstrip("0").rstrip(".") + "%"
    if abs(frac - 0.0) <= 1e-9:
        return "0% inhib."
    if ce_frac is not None and np.isfinite(ce_frac) and abs(frac - ce_frac) <= 1e-7:
        return f"CE balance ({pct})"
    if abs(frac - 0.5) <= 1e-9:
        return "50% balance"
    return f"{pct} inhib."


def _architecture_sign_balance_group(mode) -> str | None:
    canonical = _SHORT_THESIS_NAME_ALIASES.get(str(mode), str(mode))
    zero_balance = {
        "binary_base",
        "binary_base_topology_shuffle",
    }
    ce_balance = {
        "real",
        "CE-real",
        "local_sign",
        "local_sign+flat",
        "local_sign+binary",
        "local_sign+sample",
        "global_sign_pres",
        "global_sign_pres_real_w",
        "global_sign_pres_real_weight",
        "binary+conshuffle+wshuffle",
        "binary+shuffle",
        "shuffle_weights",
        "shuffle",
        "conn_shuf",
        "conn_shuf_only",
        "celW+connShuf",
        "cel_sample",
    }
    half_balance = {
        "cel+randN",
        "cel_randN",
        "er+randN",
        "er_randN",
        "ws_p0.1+randN",
        "ws_p01_randN",
    }

    if canonical in zero_balance:
        return "zero"
    if canonical in ce_balance:
        return "ce"
    if canonical in half_balance:
        return "half"
    return None


def _architecture_sign_balance_color_map(modes):
    groups = {
        "zero": ["#1100FF"],
        "ce": ["#DD00FF"],
        "half": ["#FF0000"],
    }
    grouped_modes = {"zero": [], "ce": [], "half": []}
    for mode in modes:
        group = _architecture_sign_balance_group(mode)
        if group in grouped_modes:
            grouped_modes[group].append(mode)

    if not any(grouped_modes.values()):
        return {}

    color_map = {}
    for group, members in grouped_modes.items():
        palette = groups[group]
        for idx, mode in enumerate(members):
            color_map[mode] = palette[idx % len(palette)]
    return color_map


def _mode_styles_for_cv_performance(
    present_modes,
    color_scheme: str = "thesis",
):
    _validate_color_scheme(color_scheme)
    present_modes = list(dict.fromkeys(str(mode) for mode in present_modes))
    sign_fracs = {mode: _sign_fraction_from_mode(mode) for mode in present_modes}
    sign_modes = [mode for mode, frac in sign_fracs.items() if np.isfinite(frac)]
    sign_sweep = len(sign_modes) >= 3 and len(sign_modes) >= 0.6 * max(
        len(present_modes), 1
    )

    if sign_sweep:
        ce_frac = _detect_ce_sign_fraction(sign_fracs.values())
        modes = sorted(sign_modes, key=lambda mode: (sign_fracs[mode], mode))
        modes.extend(mode for mode in present_modes if mode not in modes)
        color_map = {
            mode: _sign_balance_color(
                sign_fracs.get(mode, np.nan), ce_frac, color_scheme
            )
            for mode in modes
            if np.isfinite(sign_fracs.get(mode, np.nan))
        }
        fallback_palette = mpl.colormaps[
            "Dark2" if color_scheme == "extended-abstract" else "tab20"
        ]
        fallback_idx = 0
        for mode in modes:
            if mode not in color_map:
                color_map[mode] = fallback_palette(fallback_idx % fallback_palette.N)
                fallback_idx += 1
        label_map = {mode: _sign_balance_label(mode, ce_frac) for mode in modes}
        marker_map = _mode_marker_map(modes)
        return (
            modes,
            color_map,
            label_map,
            {
                "sign_balance_colorbar": True,
                "ce_frac": ce_frac,
                "marker_map": marker_map,
            },
        )

    preferred_order = [
        "real",
        "cel+randN",
        "er+randN",
        "ws_p0.1+randN",
        "local_sign",
        "local_sign+flat",
        "local_sign+binary",
        "global_sign_pres",
        "global_sign_pres_real_w",
        "global_sign_pres_real_weight",
        "binary_base",
    ]
    modes = [mode for mode in preferred_order if mode in present_modes]
    modes.extend(mode for mode in present_modes if mode not in modes)
    ce_frac = 0.2425287356321839
    mode_fracs = {mode: _mode_sign_fraction(mode, ce_frac) for mode in modes}
    fallback_palette = mpl.colormaps[
        "Dark2" if color_scheme == "extended-abstract" else "tab20"
    ]
    color_map = {}
    palette_idx = 0
    for mode in modes:
        frac = mode_fracs.get(mode, np.nan)
        if np.isfinite(frac):
            color_map[mode] = _sign_balance_color(frac, ce_frac, color_scheme)
        else:
            color_map[mode] = fallback_palette(palette_idx % fallback_palette.N)
            palette_idx += 1
    label_map = {mode: _short_legend_name(mode) for mode in modes}
    marker_map = _mode_marker_map(modes)
    return (
        modes,
        color_map,
        label_map,
        {
            "sign_balance_colorbar": any(np.isfinite(v) for v in mode_fracs.values()),
            "ce_frac": ce_frac,
            "marker_map": marker_map,
        },
    )


def _mode_numeric_value(mode: str) -> float:
    nums = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(mode))
    for tok in nums:
        try:
            return float(tok)
        except Exception:
            continue
    return np.nan
