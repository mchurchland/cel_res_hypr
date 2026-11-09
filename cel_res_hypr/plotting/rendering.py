"""Rendering for reservoir research figures."""

from cel_res_hypr.paths import _replace_path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings


def _save_publication_figure(fig, out_path: str, dpi: int = 600):
    """Save figure using paper-friendly defaults."""
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="white",
    )


def _tight_layout_quiet(fig):
    """Apply tight layout while suppressing benign layout warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Tight layout not applied.*", category=UserWarning
        )
        fig.tight_layout()


def _style_3d_axis(
    ax,
    tick_labelsize: int = 11,
    tick_pad: int = 2,
    z_tick_pad: int = 2,
    pane_fill: bool = True,
    pane_facecolor=(1.0, 1.0, 1.0, 1.0),
    pane_edgecolor=(0.85, 0.85, 0.85, 1.0),
):
    """Apply a cleaner, publication-friendly 3D style."""
    ax.grid(True, which="major", linestyle=":", alpha=0.28)
    ax.tick_params(axis="x", which="major", labelsize=tick_labelsize, pad=tick_pad)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize, pad=tick_pad)
    ax.tick_params(axis="z", which="major", labelsize=tick_labelsize, pad=z_tick_pad)
    y_formatter = ax.yaxis.get_major_formatter()
    if isinstance(y_formatter, mpl.ticker.ScalarFormatter):
        y_formatter.set_useOffset(False)
        y_formatter.set_scientific(False)
    ax.yaxis.get_offset_text().set_visible(False)

    try:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = pane_fill
            axis.pane.set_facecolor(pane_facecolor)
            axis.pane.set_edgecolor(pane_edgecolor)
    except Exception:
        pass


def _clear_3d_axis_background(ax):
    """Make the 3D axis background transparent while keeping guide lines visible."""
    try:
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax.patch.set_alpha(0.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    except Exception:
        pass


def _format_wt_mean_axis_as_offset(ax, wt_vals_scaled):
    """
    Compact Wt-mean tick labels by subtracting a shared 2-decimal baseline.
    Axis values are already scaled by 10**z_power.
    """
    if not wt_vals_scaled:
        return None
    vals = np.asarray(wt_vals_scaled, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    base = np.floor(np.nanmin(vals) * 100.0) / 100.0
    # Show only the fine variation around the shared base; values are in micro-units.
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _pos: f"{(v - base) * 1000.0:.1f}")
    )
    ax.yaxis.get_offset_text().set_visible(False)
    return base


def _capture_figure_rgba(fig):
    """Render a figure and return an RGBA image array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return (
        np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4).copy()
    )


def _trim_white_border(img: np.ndarray, tol: int = 250, pad: int = 20) -> np.ndarray:
    """Trim mostly-white border from an RGBA image."""
    if img.ndim != 3 or img.shape[2] < 3:
        return img
    rgb = img[..., :3]
    alpha = (
        img[..., 3]
        if img.shape[2] >= 4
        else np.full(img.shape[:2], 255, dtype=np.uint8)
    )
    mask = (np.any(rgb < tol, axis=2)) & (alpha > 0)

    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0:
        return img

    y0 = max(int(ys[0]) - pad, 0)
    y1 = min(int(ys[-1]) + pad + 1, img.shape[0])
    x0 = max(int(xs[0]) - pad, 0)
    x1 = min(int(xs[-1]) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def _pad_height_center(img: np.ndarray, target_h: int) -> np.ndarray:
    """Pad image to target height with white background, centered vertically."""
    h, w = img.shape[:2]
    if h >= target_h:
        return img
    out = np.full((target_h, w, 4), 255, dtype=np.uint8)
    top = (target_h - h) // 2
    out[top : top + h, :w] = img
    return out


def _pad_width_center(img: np.ndarray, target_w: int) -> np.ndarray:
    """Pad image to target width with white background, centered horizontally."""
    h, w = img.shape[:2]
    if w >= target_w:
        return img
    out = np.full((h, target_w, 4), 255, dtype=np.uint8)
    left = (target_w - w) // 2
    out[:, left : left + w] = img
    return out


def _save_3d_front_back(
    fig,
    axes,
    out_png: str,
    front_view=(22, -35),
    back_view=(22, 145),
    dpi: int = 600,
    tick_labelsize: int = 11,
    tick_pad: int = 2,
    z_tick_pad: int = 2,
    pane_fill: bool = True,
    pane_facecolor=(1.0, 1.0, 1.0, 1.0),
    pane_edgecolor=(0.85, 0.85, 0.85, 1.0),
):
    """Save one publication PNG with front/back views side-by-side."""
    if isinstance(axes, (list, tuple, np.ndarray)):
        axes_list = [ax for ax in axes if ax is not None]
    else:
        axes_list = [axes] if axes is not None else []

    for ax in axes_list:
        _style_3d_axis(
            ax,
            tick_labelsize=tick_labelsize,
            tick_pad=tick_pad,
            z_tick_pad=z_tick_pad,
            pane_fill=pane_fill,
            pane_facecolor=pane_facecolor,
            pane_edgecolor=pane_edgecolor,
        )

    front_png = _replace_path(out_png)
    front_root, front_ext = os.path.splitext(front_png)
    if not front_ext:
        front_ext = ".png"
        front_png = f"{front_png}{front_ext}"
        front_root, _ = os.path.splitext(front_png)

    original_views = []
    for ax in axes_list:
        original_views.append((ax.elev, ax.azim))

    for ax in axes_list:
        ax.view_init(elev=front_view[0], azim=front_view[1])
    img_front = _trim_white_border(_capture_figure_rgba(fig))

    for ax in axes_list:
        ax.view_init(elev=back_view[0], azim=back_view[1])
    img_back = _trim_white_border(_capture_figure_rgba(fig))

    target_h = max(img_front.shape[0], img_back.shape[0])
    img_front = _pad_height_center(img_front, target_h)
    img_back = _pad_height_center(img_back, target_h)

    gap_px = 0
    canvas_w = img_front.shape[1] + gap_px + img_back.shape[1]
    canvas = np.full((target_h, canvas_w, 4), 255, dtype=np.uint8)
    canvas[:, : img_front.shape[1]] = img_front
    canvas[:, img_front.shape[1] + gap_px :] = img_back

    panel_fig = plt.figure(
        figsize=(canvas.shape[1] / float(dpi), canvas.shape[0] / float(dpi)),
        dpi=dpi,
    )
    ax_panel = panel_fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_panel.imshow(canvas)
    ax_panel.axis("off")

    _save_publication_figure(panel_fig, front_png, dpi=dpi)
    plt.close(panel_fig)

    for ax, (elev, azim) in zip(axes_list, original_views):
        ax.view_init(elev=elev, azim=azim)

    return [front_png]
