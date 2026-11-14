"""Regenerate the Stokes and width-selection plots from saved results.

Run from the repository root:

    MPLCONFIGDIR=/tmp/mpl python3 -m experiments.plot_revision_figures
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
import numpy as np  # noqa: E402

from .publication_style import configure, align_panels, save_panels

ROOT = Path("results/paper_revision")
FIGURES = Path("figures")
STYLE = {3: ("#0072B2", "o"), 5: ("#D55E00", "s"), 7: ("#009E73", "^")}
FIELDS = (
    ("laplace", "Laplace", *STYLE[3]),
    ("pressure", "Pressure", *STYLE[5]),
    ("velocity", "Velocity", *STYLE[7]),
)
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 9,
    "lines.linewidth": 1.3, "lines.markersize": 4.5,
})


def _load(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _series_key(label, color, marker):
    return Line2D([], [], color=color, marker=marker, linestyle="none", label=label)


def _style_key(label, linestyle):
    color = "0.25"
    return Line2D([], [], color=color, linestyle=linestyle, label=label)


def _order_keys():
    return [_series_key(fr"$p={order}$", color, marker)
            for order, (color, marker) in STYLE.items()]


def _field_keys():
    return [_series_key(label, color, marker) for _, label, color, marker in FIELDS]


def _legend(fig, keys):
    fig._publication_handles = getattr(fig, "_publication_handles", []) + keys


def _log_axes(ax, ticks, labels=None):
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if labels is not None else [f"{tick:g}" for tick in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(False, which="both")
    ax.set_axisbelow(True)


def _reference(ax, x, y, slope, scale=2.5):
    segment = np.geomspace(min(x), np.sqrt(min(x) * max(x)), 20)
    reference = min(y) * scale * (segment / min(x))**slope
    ax.loglog(segment, reference, "--", color="0.4", lw=.9)


def _plot_stokes_pair(filename, output, x_key, x_label, ticks, second_error_key,
                      first_label, second_label, y_label):
    """Place the Stokeslet and stresslet on the same error scale."""
    rows = _load(filename)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6.5, 2.65))
    for ax, operator, title in zip(axes, ("single", "double"),
                                   ("Stokeslet", "Stresslet")):
        for order, (color, marker) in STYLE.items():
            selected = sorted(
                (float(row[x_key]), float(row["error_inf"]), float(row[second_error_key]))
                for row in rows
                if row["operator"] == operator and int(row["p"]) == order
                and float(row.get("lambda_signed", -0.5)) == -0.5
            )
            x, error, comparison = np.array(selected).T
            ax.loglog(x, error, color=color, marker=marker)
            ax.loglog(x, comparison, color=color, marker=marker,
                      mfc=color, linestyle="--")
        _log_axes(ax, ticks)
        ax.set_title(title)
        if filename == "width_sweep.csv":
            ax._publication_legend_ncol = 1
            ax._publication_legend_loc = "lower left"
    axes[0].set_ylabel(y_label)
    for ax in axes:
        ax.set_xlabel(x_label)
    _legend(fig, _order_keys() + [_style_key(first_label, "-"),
                                  _style_key(second_label, "--")])
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.23, top=0.80, wspace=0.10)
    align_panels(fig, list(axes))
    save_panels(fig, FIGURES / output)
    plt.close(fig)


def plot_stokes_corrections():
    _plot_stokes_pair("stokes_ablation.csv", "fig_stokes_correction_ablation_rev.pdf",
                      "delta", r"Smoothing width $\delta$", (0.00125, 0.005, 0.01, 0.04),
                      "raw_error_inf", "With correction", "Without correction", "Maximum vector error")


def plot_width_selection():
    _plot_stokes_pair("width_sweep.csv", "fig_width_selection_rev.pdf",
                      "delta_over_h_phys", r"Smoothing width $\delta/h_{\rm phys}$",
                      (0.35, 1, 1.5, 2, 4, 8), "source_doubling_inf",
                      "Error", "Doubling", "Maximum vector difference")


def plot_near_curve_limits():
    """Errors against exact solutions on both sides of the curve."""
    rows = _load("manufactured_tiny_lambda.csv")
    fig, ax = plt.subplots(figsize=(4.9, 2.85))
    for field, _, color, marker in FIELDS:
        for side in (-1, 1):
            selected = sorted((abs(float(row["lambda_signed"])), float(row["error_inf"]))
                              for row in rows if row["field"] == field
                              and np.sign(float(row["lambda_signed"])) == side)
            x, error = np.array(selected).T
            ax.loglog(x, error, marker=marker, color=color,
                      mfc=color, ls="-" if side < 0 else ":")
        on_curve = next(float(row["error_inf"]) for row in rows
                        if row["field"] == field and float(row["lambda_signed"]) == 0.0)
        ax.axhline(on_curve, color=color, lw=0.8, ls="--", alpha=0.8, zorder=1)
    _log_axes(ax, (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.5),
              [fr"$10^{{{power}}}$" for power in (-10, -8, -6, -4, -2)] + ["0.5"])
    ax.set(xlabel=r"Normal distance $|b|/\delta$", ylabel="Maximum error", ylim=(1e-9, 6e-7))
    _legend(fig, _field_keys())
    _legend(fig, [_style_key("Interior", "-"),
                  _style_key("Exterior", ":"),
                  _style_key("On the curve", "--")])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.79)
    align_panels(fig, [ax], columns=1)
    save_panels(fig, FIGURES / "fig_near_curve_limits.pdf")
    plt.close(fig)


def plot_oncurve_stresslet():
    """Stresslet integral at targets on the curve."""
    rows = _load("reduced_stresslet.csv")
    fig, ax = plt.subplots(figsize=(4.9, 2.85))
    for order, (color, marker) in STYLE.items():
        selected = sorted((float(row["delta"]), float(row["reduced_error_inf"]),
                           float(row["full_error_inf"]))
                          for row in rows if int(row["p"]) == order)
        width, reduced, full = np.array(selected).T
        ax.loglog(width, reduced, marker=marker, color=color)
        ax.loglog(width, full, marker=marker, color=color, mfc=color, ls=":")
        _reference(ax, width, reduced, order, scale=0.25)
    _log_axes(ax, (0.00125, 0.005, 0.01, 0.04))
    ax.set(xlabel=r"Smoothing width $\delta$", ylabel="Maximum vector error")
    _legend(fig, _order_keys())
    _legend(fig, [_style_key("Simplified", "-"),
                  _style_key("General", ":"),
                  _style_key("Slopes 3, 5, 7", "--")])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.79)
    align_panels(fig, [ax], columns=1)
    save_panels(fig, FIGURES / "fig_oncurve_stresslet.pdf")
    plt.close(fig)


def plot_allgrid_widths():
    """Compare two smoothing widths on the same Cartesian grids."""
    with (ROOT / "allgrid_width_comparison.json").open() as stream:
        data = json.load(stream)
    columns = {name: i for i, name in enumerate(data["columns"])}
    rows = np.array(data["rows"], dtype=float)
    grid_sizes = rows[:, columns["M"]].astype(int)
    spacing = 3.0 / (grid_sizes + 1)
    fig, ax = plt.subplots(figsize=(4.9, 2.85))
    for field, _, color, marker in FIELDS:
        ax.loglog(spacing, rows[:, columns[f"separated_{field}_inf"]], marker=marker,
                  color=color)
        ax.loglog(spacing, rows[:, columns[f"practical_{field}_inf"]], marker=marker,
                  color=color, mfc=color, ls="--")
    _log_axes(ax, spacing, [str(size) if size != 768 else "" for size in grid_sizes])
    ax.set(xlabel=r"Grid size $M$", ylabel="Maximum error")
    _legend(fig, _field_keys())
    _legend(fig, [_style_key(r"$\delta=h^{5/7}$", "-"),
                  _style_key(r"$\delta=2h_c$", "--")])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.79)
    align_panels(fig, [ax], columns=1)
    save_panels(fig, FIGURES / "fig_allgrid_widths.pdf")
    plt.close(fig)


def main():
    configure()
    plot_stokes_corrections()
    plot_oncurve_stresslet()
    plot_near_curve_limits()
    plot_width_selection()
    plot_allgrid_widths()


if __name__ == "__main__":
    main()
