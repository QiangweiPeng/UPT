"""
Nature Methods-style dose interpolation panel (point-refined version)

Refinement goal
---------------
Reduce the "AI-ish" look of the markers by:
1. Making predicted median markers smaller and more restrained.
2. Removing the strong white-edge / glossy look.
3. Making observed markers lighter, thinner, and less button-like.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Optional, Sequence, Tuple


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _pick_available_font(
    preferred=("Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"),
):
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            return font
    return "DejaVu Sans"


def _to_plot_x(x, logx: bool = True):
    x = np.asarray(x, dtype=float)
    return np.log10(x) if logx else x


def _format_dose_tick_nm(x):
    x = float(x)
    if not np.isfinite(x):
        return ""
    if x < 1000:
        return f"{int(x)}" if abs(x - int(x)) < 1e-9 else f"{x:g}"
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    if abs(mant - 1) < 1e-8:
        return rf"$10^{{{exp}}}$"
    if abs(mant - int(mant)) < 1e-8:
        mant = int(mant)
    return rf"${mant}\times10^{{{exp}}}$"


def _format_log_integer_tick(log_value):
    log_value = float(log_value)
    if not np.isfinite(log_value):
        return ""
    nearest = int(round(log_value))
    if abs(log_value - nearest) < 1e-8:
        return f"{nearest}"
    return ""


def _gaussian_kde_1d(vals, grid, bandwidth: Optional[float] = None):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    grid = np.asarray(grid, dtype=float)

    if len(vals) < 2:
        return np.zeros_like(grid)

    std = np.std(vals, ddof=1)

    if bandwidth is None:
        bw = 1.06 * std * len(vals) ** (-1 / 5)
        if not np.isfinite(bw) or bw <= 0:
            bw = max(abs(np.median(vals)) * 0.03, 1e-3)
    else:
        bw = float(bandwidth)

    z = (grid[:, None] - vals[None, :]) / bw
    density = np.exp(-0.5 * z**2).sum(axis=1)
    density /= len(vals) * bw * np.sqrt(2 * np.pi)

    return density


def summarize_particle_df(
    particle_df: pd.DataFrame,
    x_key: str = "dose_value",
    y_key: str = "mass_ratio",
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    fixed_values: Optional[dict] = None,
):
    df = particle_df.copy()

    if fixed_values is not None:
        for k, v in fixed_values.items():
            if k in df.columns:
                df = df[df[k].astype(str) == str(v)]

    df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
    df[y_key] = pd.to_numeric(df[y_key], errors="coerce")
    df = df.dropna(subset=[perturb_key, cellline_key, x_key, y_key])

    summary = (
        df.groupby([perturb_key, cellline_key, x_key], as_index=False)[y_key]
        .agg(
            pred_mean="mean",
            pred_median="median",
            pred_q05=lambda x: np.quantile(x, 0.05),
            pred_q10=lambda x: np.quantile(x, 0.10),
            pred_q25=lambda x: np.quantile(x, 0.25),
            pred_q75=lambda x: np.quantile(x, 0.75),
            pred_q90=lambda x: np.quantile(x, 0.90),
            pred_q95=lambda x: np.quantile(x, 0.95),
            n="size",
        )
    )
    return summary


# ---------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------
def plot_dose_violin_panel_nature_methods_refined(
    particle_df: pd.DataFrame,
    x_key: str = "dose_value",
    y_key: str = "mass_ratio",
    perturb_key: str = "perturbation",
    cellline_key: str = "cell_line",
    fixed_values: Optional[dict] = None,
    true_df: Optional[pd.DataFrame] = None,
    true_y_key: str = "true_mass_ratio",

    # Transforms
    logx: bool = True,
    logy: bool = False,

    # Summary statistic
    center: str = "median",

    # Background violin
    show_violin: bool = True,
    violin_scale: str = "area",
    violin_width: float = 0.24,
    violin_area: Optional[float] = None,
    max_violin_width_factor: float = 0.72,
    min_points_per_violin: int = 20,
    y_grid_size: int = 280,
    bandwidth: Optional[float] = None,
    trim_quantile: Tuple[float, float] = (0.005, 0.995),
    density_alpha: float = 0.14,
    density_outline_alpha: float = 0.22,
    density_outline_width: float = 0.35,

    # Quantile ribbons
    show_outer_ribbon: bool = True,
    outer_interval: Tuple[float, float] = (0.10, 0.90),
    outer_ribbon_alpha: float = 0.14,

    show_inner_ribbon: bool = True,
    inner_interval: Tuple[float, float] = (0.25, 0.75),
    inner_ribbon_alpha: float = 0.26,

    # Predicted median line / markers
    prediction_linewidth: float = 1.70,
    prediction_marker_size: float = 3.35,
    prediction_marker_edge_width: float = 0.35,

    # Observed markers
    observed_marker: str = "o",
    observed_size: float = 28,
    observed_linewidth: float = 1.10,

    # Figure layout
    baseline: Optional[float] = 1.0,
    figsize: Tuple[float, float] = (7.20, 2.55),
    ncols: int = 3,
    sharey: bool = True,
    y_limits: Optional[Tuple[float, float]] = None,
    title_prefix: Optional[str] = None,
    xlabel: str = r"$\log_{10}(\mathrm{dose})$",
    ylabel: str = "Mass ratio",
    panel_title_mode: str = "cellline",
    cellline_order: Optional[Sequence[str]] = None,
    order: Optional[Sequence[Tuple[str, str]]] = None,

    # Ticks
    x_tick_mode: str = "log_integer",
    major_doses: Optional[Sequence[float]] = None,
    major_log_ticks: Optional[Sequence[float]] = None,
    minor_x_ticks: bool = False,
    rotate_xticks: float = 0,

    # Legend labels
    observed_label: str = "Observed",
    prediction_label: str = "Predicted median",
    violin_label: str = "Predicted cells",
    inner_ribbon_label: str = "Predicted IQR",
    outer_ribbon_label: str = "Predicted 10–90%",

    # Panel labels
    show_panel_labels: bool = True,
    panel_labels: Optional[Sequence[str]] = None,
    panel_label_fontsize: float = 10.0,
    panel_label_weight: str = "bold",
    panel_label_offset: Tuple[float, float] = (-0.15, 1.105),

    # Typography
    font_family: Optional[str] = None,
    base_fontsize: float = 7.2,
    title_fontsize: float = 8.4,
    axis_label_fontsize: float = 7.8,
    suptitle_fontsize: float = 9.0,
    legend_fontsize: float = 6.7,

    # Grid / axes
    show_grid: bool = True,
    grid_alpha: float = 0.80,

    # Legend / output
    show_legend: bool = True,
    legend_loc: str = "lower center",
    show: bool = True,
):
    # -----------------------------------------------------------------
    # Palette
    # -----------------------------------------------------------------
    pred_line_color = "#5B4F9A"
    pred_fill_color = "#6A61AC"
    pred_fill_light = "#AAA6CF"
    density_face = "#CBD9E5"
    density_edge = "#90A9BF"

    # Point palette refined
    pred_marker_face = "#5B4F9A"
    pred_marker_edge = "#ECEAF4"   # not pure white; much less "glossy"
    obs_color = "#BB5B70"
    obs_face = "#FCF8F9"           # warm off-white instead of pure white

    baseline_color = "#BCC9D4"
    grid_color = "#EDEDED"
    text_color = "#222222"
    spine_color = "#383838"

    if font_family is None:
        font_family = _pick_available_font()

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.linewidth": 0.6,
        "axes.edgecolor": spine_color,
        "axes.labelcolor": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,
        "text.color": text_color,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    })

    # -----------------------------------------------------------------
    # Prepare prediction data
    # -----------------------------------------------------------------
    df = particle_df.copy()

    if fixed_values is not None:
        for k, v in fixed_values.items():
            if k in df.columns:
                df = df[df[k].astype(str) == str(v)]

    df[x_key] = pd.to_numeric(df[x_key], errors="coerce")
    df[y_key] = pd.to_numeric(df[y_key], errors="coerce")
    df = df.dropna(subset=[perturb_key, cellline_key, x_key, y_key])

    if logx:
        df = df[df[x_key] > 0]
    if logy:
        df = df[df[y_key] > 0]

    if len(df) == 0:
        raise ValueError("No prediction data left after filtering.")

    # -----------------------------------------------------------------
    # Prepare observed data
    # -----------------------------------------------------------------
    true_plot = None
    if true_df is not None and true_y_key in true_df.columns:
        true_plot = true_df.copy()

        if fixed_values is not None:
            for k, v in fixed_values.items():
                if k in true_plot.columns:
                    true_plot = true_plot[true_plot[k].astype(str) == str(v)]

        true_plot[x_key] = pd.to_numeric(true_plot[x_key], errors="coerce")
        true_plot[true_y_key] = pd.to_numeric(true_plot[true_y_key], errors="coerce")
        true_plot = true_plot.dropna(
            subset=[perturb_key, cellline_key, x_key, true_y_key]
        )

        if logx:
            true_plot = true_plot[true_plot[x_key] > 0]
        if logy:
            true_plot = true_plot[true_plot[true_y_key] > 0]

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    summary = summarize_particle_df(
        particle_df=df,
        x_key=x_key,
        y_key=y_key,
        perturb_key=perturb_key,
        cellline_key=cellline_key,
        fixed_values=None,
    )

    # -----------------------------------------------------------------
    # Panel order
    # -----------------------------------------------------------------
    panel_keys = list(df.groupby([perturb_key, cellline_key]).groups.keys())

    if order is not None:
        ordered_keys = [(str(p), str(c)) for p, c in order]
        panel_keys = [
            k for target in ordered_keys
            for k in panel_keys
            if (str(k[0]), str(k[1])) == target
        ]
    elif cellline_order is not None:
        cellline_order = list(cellline_order)
        panel_keys = sorted(
            panel_keys,
            key=lambda k: (
                cellline_order.index(k[1]) if k[1] in cellline_order else 999,
                str(k[0]),
                str(k[1]),
            ),
        )
    else:
        panel_keys = sorted(panel_keys, key=lambda k: (str(k[0]), str(k[1])))

    if len(panel_keys) == 0:
        raise ValueError("No panels to plot.")

    n_panels = len(panel_keys)
    ncols = min(int(ncols), n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    # -----------------------------------------------------------------
    # Shared x positions
    # -----------------------------------------------------------------
    all_doses = np.sort(df[x_key].dropna().unique())
    all_xpos = _to_plot_x(all_doses, logx=logx)

    if len(all_xpos) > 1:
        min_spacing = np.min(np.diff(np.sort(all_xpos)))
        half_width = violin_width * min_spacing
    else:
        half_width = violin_width

    if violin_area is None:
        target_half_area = half_width * 0.28
    else:
        target_half_area = float(violin_area)

    # -----------------------------------------------------------------
    # X ticks
    # -----------------------------------------------------------------
    if x_tick_mode == "log_integer":
        if not logx:
            raise ValueError("x_tick_mode='log_integer' requires logx=True.")
        log_min = int(np.ceil(np.min(all_xpos)))
        log_max = int(np.floor(np.max(all_xpos)))
        major_xpos = np.arange(log_min, log_max + 1, dtype=float)
        major_labels = [str(int(v)) for v in major_xpos]

    elif x_tick_mode == "dose":
        if major_doses is None:
            candidate_ticks = np.array([10, 100, 1000, 10000, 100000], dtype=float)
            major_doses = [
                d for d in candidate_ticks
                if d >= all_doses.min() and d <= all_doses.max()
            ]
            if len(major_doses) < 3:
                keep_idx = np.linspace(
                    0, len(all_doses) - 1, min(5, len(all_doses))
                ).round().astype(int)
                major_doses = all_doses[keep_idx]

        major_doses = np.asarray(major_doses, dtype=float)
        major_xpos = _to_plot_x(major_doses, logx=logx)
        major_labels = [_format_dose_tick_nm(d) for d in major_doses]

    elif x_tick_mode == "custom":
        if major_log_ticks is None:
            raise ValueError("Provide major_log_ticks when x_tick_mode='custom'.")
        major_xpos = np.asarray(major_log_ticks, dtype=float)
        major_labels = [_format_log_integer_tick(x) for x in major_xpos]

    else:
        raise ValueError("x_tick_mode must be one of {'log_integer', 'dose', 'custom'}.")

    # -----------------------------------------------------------------
    # Shared y limits
    # -----------------------------------------------------------------
    if y_limits is None:
        y_sources = [df[y_key].to_numpy(dtype=float)]
        if true_plot is not None and len(true_plot) > 0:
            y_sources.append(true_plot[true_y_key].to_numpy(dtype=float))

        y_all = np.concatenate(y_sources)
        y_all = y_all[np.isfinite(y_all)]
        if logy:
            y_all = y_all[y_all > 0]
        if len(y_all) == 0:
            raise ValueError("No finite y-values to plot.")

        y0, y1 = np.quantile(y_all, [0.005, 0.995])
        pad = 0.08 * (y1 - y0)
        ymin = max(0, y0 - pad) if y0 >= 0 else y0 - pad
        ymax = y1 + pad

        if baseline is not None:
            ymin = min(ymin, baseline - 0.04)
            ymax = max(ymax, baseline + 0.04)

        y_limits = (ymin, ymax)

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharey=sharey,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    # -----------------------------------------------------------------
    # Plot panels
    # -----------------------------------------------------------------
    for ax_i, ax in enumerate(axes_flat):
        if ax_i >= n_panels:
            ax.axis("off")
            continue

        pert, cellline = panel_keys[ax_i]

        cur_df = df[
            (df[perturb_key].astype(str) == str(pert)) &
            (df[cellline_key].astype(str) == str(cellline))
        ].copy()

        cur_summary = summary[
            (summary[perturb_key].astype(str) == str(pert)) &
            (summary[cellline_key].astype(str) == str(cellline))
        ].copy().sort_values(x_key)

        dose_values = np.sort(cur_summary[x_key].dropna().unique())

        if logy:
            ax.set_yscale("log")

        # Baseline
        if baseline is not None:
            ax.axhline(
                baseline,
                linestyle=(0, (2.2, 2.2)),
                linewidth=0.70,
                color=baseline_color,
                alpha=0.85,
                zorder=0,
            )

        # Grid
        if show_grid:
            ax.grid(
                True,
                axis="y",
                linestyle="-",
                linewidth=0.35,
                color=grid_color,
                alpha=grid_alpha,
                zorder=0,
            )
            ax.grid(False, axis="x")

        # -------------------------------------------------------------
        # Background violins
        # -------------------------------------------------------------
        if show_violin:
            max_n_per_panel = cur_df.groupby(x_key)[y_key].size().max()
            if not np.isfinite(max_n_per_panel) or max_n_per_panel <= 0:
                max_n_per_panel = 1

            for dose in dose_values:
                vals = cur_df.loc[cur_df[x_key] == dose, y_key].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]

                if logy:
                    vals = vals[vals > 0]
                if len(vals) < min_points_per_violin:
                    continue

                q_low, q_high = np.quantile(vals, trim_quantile)
                if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
                    c = np.median(vals)
                    spread = max(abs(c) * 0.05, 1e-3)
                    q_low, q_high = c - spread, c + spread

                pad = 0.10 * (q_high - q_low)
                y_grid = np.linspace(q_low - pad, q_high + pad, int(y_grid_size))

                if logy:
                    y_grid = y_grid[y_grid > 0]
                if len(y_grid) < 5:
                    continue

                raw_density = _gaussian_kde_1d(vals, y_grid, bandwidth=bandwidth)
                if (
                    len(raw_density) == 0
                    or np.max(raw_density) <= 0
                    or not np.isfinite(np.max(raw_density))
                ):
                    continue

                if violin_scale == "width":
                    density = raw_density / np.max(raw_density) * half_width

                elif violin_scale == "area":
                    area = np.trapz(raw_density, y_grid)
                    if area <= 0 or not np.isfinite(area):
                        continue
                    density = raw_density / area * target_half_area
                    max_width = half_width * max_violin_width_factor
                    if np.max(density) > max_width:
                        density = density / np.max(density) * max_width

                elif violin_scale == "count":
                    area = np.trapz(raw_density, y_grid)
                    if area <= 0 or not np.isfinite(area):
                        continue
                    n_scale = np.sqrt(len(vals) / max_n_per_panel)
                    density = raw_density / area * target_half_area * n_scale
                    max_width = half_width * max_violin_width_factor
                    if np.max(density) > max_width:
                        density = density / np.max(density) * max_width

                else:
                    raise ValueError("violin_scale must be one of {'width', 'area', 'count'}.")

                x0 = _to_plot_x(np.asarray([dose]), logx=logx)[0]
                x_left = x0 - density
                x_right = x0 + density

                poly_x = np.concatenate([x_left, x_right[::-1]])
                poly_y = np.concatenate([y_grid, y_grid[::-1]])

                ax.fill(
                    poly_x,
                    poly_y,
                    facecolor=density_face,
                    edgecolor="none",
                    alpha=density_alpha,
                    zorder=1,
                )
                ax.plot(
                    x_left,
                    y_grid,
                    color=density_edge,
                    linewidth=density_outline_width,
                    alpha=density_outline_alpha,
                    zorder=1.2,
                )
                ax.plot(
                    x_right,
                    y_grid,
                    color=density_edge,
                    linewidth=density_outline_width,
                    alpha=density_outline_alpha,
                    zorder=1.2,
                )

        # -------------------------------------------------------------
        # Quantile ribbons
        # -------------------------------------------------------------
        x_line = _to_plot_x(cur_summary[x_key].to_numpy(dtype=float), logx=logx)

        if center == "median":
            y_line = cur_summary["pred_median"].to_numpy(dtype=float)
        elif center == "mean":
            y_line = cur_summary["pred_mean"].to_numpy(dtype=float)
        else:
            raise ValueError("center must be 'median' or 'mean'.")

        q10 = cur_summary["pred_q10"].to_numpy(dtype=float)
        q25 = cur_summary["pred_q25"].to_numpy(dtype=float)
        q75 = cur_summary["pred_q75"].to_numpy(dtype=float)
        q90 = cur_summary["pred_q90"].to_numpy(dtype=float)

        valid = np.isfinite(x_line) & np.isfinite(y_line)
        valid &= np.isfinite(q10) & np.isfinite(q25) & np.isfinite(q75) & np.isfinite(q90)

        if logy:
            valid &= (y_line > 0) & (q10 > 0) & (q25 > 0) & (q75 > 0) & (q90 > 0)

        x_valid = x_line[valid]
        y_valid = y_line[valid]
        q10_valid = q10[valid]
        q25_valid = q25[valid]
        q75_valid = q75[valid]
        q90_valid = q90[valid]

        if show_outer_ribbon and len(x_valid) > 1:
            ax.fill_between(
                x_valid,
                q10_valid,
                q90_valid,
                color=pred_fill_light,
                alpha=outer_ribbon_alpha,
                linewidth=0,
                zorder=2,
            )

        if show_inner_ribbon and len(x_valid) > 1:
            ax.fill_between(
                x_valid,
                q25_valid,
                q75_valid,
                color=pred_fill_color,
                alpha=inner_ribbon_alpha,
                linewidth=0,
                zorder=3,
            )

        # Median line
        if len(x_valid) > 0:
            median_line, = ax.plot(
                x_valid,
                y_valid,
                color=pred_line_color,
                linewidth=prediction_linewidth,
                alpha=0.98,
                zorder=4,
                solid_capstyle="round",
            )
            median_line.set_path_effects([
                pe.Stroke(linewidth=prediction_linewidth + 0.65, foreground="white"),
                pe.Normal(),
            ])

            # Refined predicted markers: smaller, flatter, less glossy
            ax.plot(
                x_valid,
                y_valid,
                linestyle="none",
                marker="o",
                markersize=prediction_marker_size,
                markerfacecolor=pred_marker_face,
                markeredgecolor=pred_marker_edge,
                markeredgewidth=prediction_marker_edge_width,
                alpha=0.98,
                zorder=4.8,
            )

        # -------------------------------------------------------------
        # Observed
        # -------------------------------------------------------------
        if true_plot is not None and len(true_plot) > 0:
            true_cur = true_plot[
                (true_plot[perturb_key].astype(str) == str(pert)) &
                (true_plot[cellline_key].astype(str) == str(cellline))
            ].copy()

            if len(true_cur) > 0:
                true_agg = (
                    true_cur.groupby(x_key, as_index=False)[true_y_key]
                    .mean()
                    .sort_values(x_key)
                )

                tx = true_agg[x_key].to_numpy(dtype=float)
                ty = true_agg[true_y_key].to_numpy(dtype=float)

                valid_true = np.isfinite(tx) & np.isfinite(ty)
                if logx:
                    valid_true &= tx > 0
                if logy:
                    valid_true &= ty > 0

                tx_plot = _to_plot_x(tx[valid_true], logx=logx)
                ty_plot = ty[valid_true]

                if observed_marker == "o":
                    ax.scatter(
                        tx_plot,
                        ty_plot,
                        s=observed_size,
                        marker="o",
                        facecolors=obs_face,
                        edgecolors=obs_color,
                        linewidths=observed_linewidth,
                        alpha=0.98,
                        zorder=6,
                    )
                else:
                    ax.scatter(
                        tx_plot,
                        ty_plot,
                        s=observed_size,
                        marker=observed_marker,
                        color=obs_color,
                        linewidths=observed_linewidth,
                        alpha=0.98,
                        zorder=6,
                    )

        # -------------------------------------------------------------
        # Axes
        # -------------------------------------------------------------
        ax.set_xlim(
            all_xpos.min() - half_width * 1.55,
            all_xpos.max() + half_width * 1.55,
        )
        ax.set_ylim(*y_limits)

        ax.set_xticks(major_xpos)
        ax.set_xticklabels(
            major_labels,
            fontsize=base_fontsize,
            rotation=rotate_xticks,
            ha="center" if rotate_xticks == 0 else "right",
        )

        if minor_x_ticks:
            ax.set_xticks(all_xpos, minor=True)
        else:
            ax.set_xticks([], minor=True)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=base_fontsize,
            width=0.6,
            length=2.7,
            pad=1.8,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            width=0.45,
            length=1.5,
        )

        row_i = ax_i // ncols
        col_i = ax_i % ncols

        ax.set_xlabel(
            xlabel if row_i == nrows - 1 else "",
            fontsize=axis_label_fontsize,
            labelpad=2.6,
        )
        ax.set_ylabel(
            ylabel if col_i == 0 else "",
            fontsize=axis_label_fontsize,
            labelpad=2.8,
        )

        # Panel title
        if panel_title_mode == "cellline":
            panel_title = str(cellline)
        elif panel_title_mode == "full":
            panel_title = f"{pert} | {cellline}"
        elif panel_title_mode == "none":
            panel_title = ""
        else:
            raise ValueError("panel_title_mode must be 'cellline', 'full', or 'none'.")

        ax.set_title(
            panel_title,
            fontsize=title_fontsize,
            pad=3.6,
            weight="bold",
        )

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.spines["left"].set_color(spine_color)
        ax.spines["bottom"].set_color(spine_color)

    # -----------------------------------------------------------------
    # Panel labels
    # -----------------------------------------------------------------
    if show_panel_labels:
        if panel_labels is None:
            panel_labels_used = list("abcdefghijklmnopqrstuvwxyz")[:n_panels]
        else:
            panel_labels_used = list(panel_labels)[:n_panels]

        for ax_i, ax in enumerate(axes_flat[:n_panels]):
            if ax_i >= len(panel_labels_used):
                break

            ax.text(
                panel_label_offset[0],
                panel_label_offset[1],
                panel_labels_used[ax_i],
                transform=ax.transAxes,
                fontsize=panel_label_fontsize,
                fontweight=panel_label_weight,
                family=font_family,
                va="top",
                ha="left",
            )

    # -----------------------------------------------------------------
    # Optional figure-wide title
    # -----------------------------------------------------------------
    if title_prefix is not None:
        pert_names = sorted(df[perturb_key].astype(str).unique())
        if len(pert_names) == 1:
            suptitle = f"{title_prefix}: {pert_names[0]}"
        else:
            suptitle = title_prefix

        fig.suptitle(
            suptitle,
            fontsize=suptitle_fontsize,
            weight="bold",
            y=0.985,
        )

    # -----------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------
    if show_legend:
        handles = []

        if show_violin:
            density_handle = Patch(
                facecolor=density_face,
                edgecolor=density_edge,
                linewidth=density_outline_width,
                alpha=min(density_alpha + 0.10, 1.0),
                label=violin_label,
            )
            handles.append(density_handle)

        if show_outer_ribbon:
            outer_handle = Patch(
                facecolor=pred_fill_light,
                edgecolor="none",
                alpha=outer_ribbon_alpha,
                label=outer_ribbon_label,
            )
            handles.append(outer_handle)

        if show_inner_ribbon:
            inner_handle = Patch(
                facecolor=pred_fill_color,
                edgecolor="none",
                alpha=inner_ribbon_alpha,
                label=inner_ribbon_label,
            )
            handles.append(inner_handle)

        # Legend marker also refined
        median_handle = Line2D(
            [0], [0],
            color=pred_line_color,
            marker="o",
            markersize=4.0,
            linewidth=prediction_linewidth,
            markerfacecolor=pred_marker_face,
            markeredgecolor=pred_marker_edge,
            markeredgewidth=prediction_marker_edge_width,
            label=prediction_label,
        )
        handles.append(median_handle)

        if observed_marker == "o":
            obs_handle = Line2D(
                [0], [0],
                marker="o",
                markersize=4.6,
                markerfacecolor=obs_face,
                markeredgecolor=obs_color,
                markeredgewidth=observed_linewidth,
                linewidth=0,
                label=observed_label,
            )
        else:
            obs_handle = Line2D(
                [0], [0],
                marker=observed_marker,
                markersize=4.6,
                color=obs_color,
                markeredgewidth=observed_linewidth,
                linewidth=0,
                label=observed_label,
            )
        handles.append(obs_handle)

        if legend_loc == "lower center":
            fig.legend(
                handles=handles,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.016),
                ncol=len(handles),
                fontsize=legend_fontsize,
                handlelength=1.5,
                columnspacing=1.15,
                handletextpad=0.50,
                borderaxespad=0.0,
            )
            bottom = 0.245
        else:
            fig.legend(
                handles=handles,
                frameon=False,
                loc=legend_loc,
                fontsize=legend_fontsize,
                handlelength=1.5,
                columnspacing=1.15,
                handletextpad=0.50,
            )
            bottom = 0.16
    else:
        bottom = 0.145

    top = 0.83 if title_prefix is not None else 0.86

    fig.subplots_adjust(
        left=0.078,
        right=0.992,
        top=top,
        bottom=bottom,
        wspace=0.27,
        hspace=0.34,
    )

    if show:
        plt.show()

    return fig, axes


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
