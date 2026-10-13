"""Plot functions for the two experiment notebooks."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

MODE_COLORS = {"A_nofire": "#1b9e77", "C_fire_accumulate": "#d95f02"}


def slide_style():
    """Large fonts and no top/right frame, for figures that go into a presentation."""
    plt.rcParams.update({
        "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
        "figure.titlesize": 17, "lines.linewidth": 2.2,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# ---- short kick experiment ----------------------------------------------------------------

def log2_axis(ax, horizons):
    """x axis with base 2 and ticks 1, 2, 4, ..."""
    ax.set_xscale("log", base=2)
    ticks = [2 ** k for k in range(int(np.log2(max(horizons))) + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(min(horizons) / 1.15, max(horizons) * 1.15)


def log2_yaxis(ax):
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def draw_range(ax, x, samples, color, line=None):
    """Shaded min-max range of the bootstrap draws (draws x horizons) and the line of the full data."""
    ax.fill_between(x, samples.min(axis=0), samples.max(axis=0), color=color, alpha=0.25, lw=0)
    if line is not None:
        ax.plot(x, line, color=color, marker="o", ms=3)


def plot_heatmaps(results, horizons, kicks, min_share=0.95):
    """Arrow length of every kick against the horizon (log colour scale); grey = selected in < min_share of the draws."""
    shown = {m: np.where(r["share"] >= min_share, r["norm"], np.nan) for m, r in results.items()}
    vmax = max(np.nanmax(s) for s in shown.values() if np.any(np.isfinite(s)))
    positive = np.concatenate([s[np.isfinite(s) & (s > 0)] for s in shown.values()])
    norm = LogNorm(vmin=max(10 ** np.floor(np.log10(positive.min())), vmax * 1e-4), vmax=vmax)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#eeeeee")
    p = np.log2(np.array(horizons, dtype=float))
    mid = (p[1:] + p[:-1]) / 2
    edges = 2 ** np.concatenate([[p[0] - (mid[0] - p[0])], mid, [p[-1] + (p[-1] - mid[-1])]])

    fig = plt.figure(figsize=(8.5, 3.6 * len(results) + 0.6))
    grid = fig.add_gridspec(len(results), 2, width_ratios=[40, 1], hspace=0.25, wspace=0.03)
    first = None
    for row, (mode, s) in enumerate(shown.items()):
        ax = fig.add_subplot(grid[row, 0], sharex=first)
        first = first or ax
        image = ax.pcolormesh(edges, np.arange(len(kicks) + 1) - 0.5, s, cmap=cmap, norm=norm)
        ax.set_ylim(len(kicks) - 0.5, -0.5)
        ax.set_yticks(range(len(kicks)))
        ax.set_yticklabels(kicks, fontsize=13)
        ax.set_ylabel(mode, fontsize=16, fontweight="bold")
        log2_axis(ax, horizons)
        if row < len(results) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel("years after the kick (log2 scale)")
    cbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, 1]))
    cbar.set_label(f"arrow length per full kick (z-units, log scale)\ngrey = selected in < {min_share * 100:.0f} % of the draws",
                   fontsize=13)
    return fig


def plot_modes(results, horizons, key, ylabel, relative=False, log=False):
    """One curve per mode with bootstrap range. relative=True: divided by the value at the first horizon."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, res in results.items():
        ref = res[key][0] if relative else 1.0
        draw_range(ax, horizons, res[key + "_b"] / ref, MODE_COLORS[mode], res[key] / ref)
        ax.plot([], [], color=MODE_COLORS[mode], label=mode)
    if log:
        log2_yaxis(ax)
    log2_axis(ax, horizons)
    ax.set_xlabel("years after the kick (log2 scale)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_first_years(results, horizons, max_year=32):
    """Top: response size relative to year 1; bottom: test R2. One column per mode, only the first max_year years."""
    keep = np.array(horizons) <= max_year
    h = np.array(horizons)[keep]
    fig, axes = plt.subplots(2, len(results), figsize=(5.4 * len(results), 7.2), sharex=True, squeeze=False)
    for col, (mode, res) in enumerate(results.items()):
        ref = res["size"][keep][0]
        line, band = res["size"][keep] / ref, res["size_b"][:, keep] / ref
        draw_range(axes[0, col], h, band, "#1f77b4", line)
        axes[0, col].set_title(mode, fontsize=16)
        axes[0, col].axhline(1, color="k", lw=0.6, ls=":")
        lo, hi = min(band.min(), line.min()), max(band.max(), line.max())
        axes[0, col].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        draw_range(axes[1, col], h, res["r2_b"][:, keep], "#555555", res["r2"][keep])   # grey: not a mode colour
        low = min(res["r2"][keep].min(), res["r2_b"][:, keep].min())
        axes[1, col].set_ylim(max(np.floor((low - 0.01) * 20) / 20, -0.3), 1.005)
        log2_axis(axes[1, col], h)
        axes[1, col].set_xlabel("years after the kick (log2 scale)")
    axes[0, 0].set_ylabel("response size\n(first year = 1)")
    axes[1, 0].set_ylabel("R$^2$ (test)")
    fig.suptitle(f"First {max_year} years (shaded: range over the bootstrap draws)", fontsize=16)
    fig.tight_layout()
    return fig


# ---- long rain experiment ----------------------------------------------------------------

def symlog2(axes, values):
    """y axis linear in [-1, 1] and logarithmic (base 2) outside, so spikes and negative values are both visible."""
    ymin, ymax = float(np.min(values)), float(np.max(values))
    top = [2 ** k for k in range(0, int(np.ceil(np.log2(max(ymax, 1.0)))) + 1)]
    bottom = [-2 ** k for k in range(0, int(np.ceil(np.log2(max(-ymin, 1.0)))) + 1)]
    for ax in axes:
        ax.set_yscale("symlog", base=2, linthresh=1.0, linscale=0.7)
        ax.set_yticks(sorted(set(bottom + [0] + top)))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylim(min(-1.0, 1.15 * ymin if ymin < 0 else -1.0), max(1.0, 1.15 * ymax))
        ax.axhline(0, color="grey", lw=0.5, zorder=0)


def plot_sections(runs, sections, symlog_modes=("C_fire_accumulate",)):
    """
    Control (black), members (blue) and member mean (orange), one row per mode, two cut-out sections.
    runs[mode] = (values, t) with values[run, year] and run 0 = control.
    """
    modes = list(runs)
    fig = plt.figure(figsize=(14, 2.9 * len(modes) + 1.2))
    grid = fig.add_gridspec(len(modes), len(sections), width_ratios=[b - a for a, b in sections], wspace=0.07, hspace=0.3)
    for row, mode in enumerate(modes):
        values, t = runs[mode]
        mean = values[1:].mean(axis=0)
        axes_row = []
        for col, (a, b) in enumerate(sections):
            ax = fig.add_subplot(grid[row, col])
            axes_row.append(ax)
            mask = (t >= a) & (t < b)
            for v in values[1:]:
                ax.plot(t[mask], v[mask], color="#7aa6c9", lw=0.6, alpha=0.5)
            ax.plot(t[mask], mean[mask], color="#e08a00", lw=2.0)
            ax.plot(t[mask], values[0][mask], color="black", lw=2.0)
            ax.set_xlim(a, b)
            ax.set_xticks(np.arange(a, b + (1 if col == len(sections) - 1 else 0), 100 if b - a <= 500 else 200))
            ax.tick_params(labelsize=13)
            if col > 0:
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False, labelleft=False)
            if col < len(sections) - 1:                      # "//" marks where the time axis is cut
                ax.spines["right"].set_visible(False)
            for side, show in ((1, col < len(sections) - 1), (0, col > 0)):
                if show:
                    dx = 0.012 * 500 / (b - a)
                    for y0 in (0, 1):
                        ax.plot((side - dx, side + dx), (y0 - 0.03, y0 + 0.03), transform=ax.transAxes,
                                color="k", clip_on=False, lw=1)
            if row < len(modes) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("years after perturbation", fontsize=14)
        if mode in symlog_modes:
            symlog2(axes_row, values)
        axes_row[0].set_ylabel(mode, fontsize=16, fontweight="bold")
    fig.suptitle("Net ecosystem exchange (+ = carbon lost)", fontsize=17)
    fig.subplots_adjust(top=0.9, left=0.06, right=0.995, bottom=0.14 if len(modes) == 1 else 0.11)
    return fig


def plot_groups(group_results):
    """group_results[mode] = (years, labels). Number of groups and share of members in the control's group."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for mode, (years, labels) in group_results.items():
        axes[0].plot(years, labels.max(axis=1) + 1, color=MODE_COLORS[mode], label=mode)
        axes[1].plot(years, np.mean(labels[:, 1:] == 0, axis=1), color=MODE_COLORS[mode], label=mode)
    axes[0].set_ylabel("number of groups")
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    axes[1].set_ylabel("share of members\nin the control group")
    axes[1].set_xlabel("years after perturbation")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_running_correlation(correlations):
    """correlations[mode] = table from running_correlation() in notebook 02."""
    fig, axes = plt.subplots(len(correlations), 1, figsize=(10, 2.6 * len(correlations) + 0.8), sharex=True, squeeze=False)
    for ax, (mode, c) in zip(axes[:, 0], correlations.items()):
        ax.fill_between(c.centre, c.p10, c.p90, color="#1f5f99", alpha=0.2, lw=0)
        ax.plot(c.centre, c.r_members, color="#1f5f99", label="r_m: members vs control (mean, 10-90 % band)")
        ax.plot(c.centre, c.r_ensemble_mean, color="#e08a00", ls="--", label="r_e: ensemble mean vs control")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("correlation")
        ax.text(0.01, 0.06, mode, transform=ax.transAxes, fontweight="bold", fontsize=16)
    axes[0, 0].legend(fontsize=13, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    axes[-1, 0].set_xlabel("years after perturbation (window centre)")
    fig.tight_layout()
    return fig
