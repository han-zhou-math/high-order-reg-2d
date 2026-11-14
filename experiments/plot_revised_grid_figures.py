"""Regenerate the Cartesian-grid figures from existing results.

Run from the repository root with python3 -m experiments.plot_revised_grid_figures.
The convergence panels use the saved JSON reports.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.ticker import NullFormatter, NullLocator

from .allgrid import (GRADIENT_OFFSETS, LAPLACE_OFFSETS, classify_crossing,
                      grid_coordinates)
from .plot_swapped_production import FIGURES, _grid_reports
from .swapped_low_order import STOKES_STARFISH


from .publication_style import configure, align_panels, save_panels

GRID_FIELDS = (
    ("laplace", "Laplace", "#0072B2", "o"),
    ("pressure", "Stokes pressure", "#D55E00", "s"),
    ("velocity", "Stokes velocity", "#009E73", "^"),
)


def _grid_size_axis(ax, rows):
    """Label refinement levels at their logarithmic spacing coordinates."""
    ax.set_xticks([row["h_c"] for row in rows])
    ax.set_xticklabels([str(row["m_grid"]) if row["m_grid"] != 768 else "" for row in rows], fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Grid size $M$")
    ax.grid(False, which="both")
    ax.set_axisbelow(True)


def _slope(ax, left, right, value, order):
    ax.loglog([left, right], [value, value * (right / left)**order],
              color="0.35", linestyle="--", linewidth=.9, zorder=1)
    ax.annotate(f"slope {order}", (right, value * (right / left)**order),
                xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color="0.30")


def plot_allgrid_convergence():
    """Plot measured convergence of the complete Cartesian-grid method."""
    laplace = _grid_reports("laplace")
    stokes = _grid_reports("stokes")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True)
    for ax, norm, title in zip(axes, ("inf", "l2"),
                               ("(a) Maximum error", r"(b) Discrete $L^2$ error")):
        for field, label, color, marker in GRID_FIELDS:
            rows = laplace if field == "laplace" else stokes
            ax.loglog([row["h_c"] for row in rows],
                      [row[field][norm] for row in rows], color=color,
                      label=label, marker=marker, markersize=4)
        ax.set_title(title, loc="left", fontsize=9)
    smallest = laplace[-1]
    _slope(axes[1], smallest["h_c"], laplace[2]["h_c"],
           0.5 * smallest["laplace"]["l2"], 4)
    for ax in axes:
        _grid_size_axis(ax, laplace)
    fig.subplots_adjust(left=.091, right=.987, bottom=.185, top=.79, wspace=.28)
    align_panels(fig, list(axes))
    save_panels(fig, FIGURES / "fig_allgrid_convergence_rev.pdf")
    plt.close(fig)


def plot_exact_data_reconstruction():
    """Plot the grid-only convergence controls with exact interface data."""
    root = Path("results/paper_revision")
    rows = json.loads((root / "exact_data_reconstruction.json").read_text())["rows"]
    fits = json.loads((root / "exact_data_reconstruction_fits.json").read_text())
    curves = (
        ("Laplace", "Laplace", "laplace", "laplace", "#0072B2", "o"),
        ("Pressure", "Pressure", "stokes", "pressure", "#D55E00", "s"),
        ("Velocity, numerical pressure gradient", "Computed gradient", "stokes",
         "velocity", "#009E73", "^"),
        ("Velocity, exact pressure gradient", "Exact gradient", "stokes_exact_gradient",
         "velocity", "#7A5195", "D"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharex=True)
    for index, (fit, label, group, field, color, marker) in enumerate(curves):
        ax = axes[index // 2]
        series = sorted((row for row in rows if row["diagnostic_label"] == group),
                        key=lambda row: row["m_grid"])
        spacing = np.array([row["h_c"] for row in series])
        errors = np.array([row[field]["inf"] for row in series])
        ax.loglog(spacing, errors, color=color, marker=marker, markersize=4,
                  label=f"{label} ({fits[fit]['rate']:.2f})")
        if index in (0, 2):
            order = 6 if index == 0 else 4
            scale = .35 if order == 6 else .1
            _slope(ax, spacing[-1], spacing[2], scale * errors[2] *
                   (spacing[-1] / spacing[2])**order, order)
        _grid_size_axis(ax, series)
    for ax, title in zip(axes, ("(a) Harmonic fields", "(b) Stokes velocity")):
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("Maximum error")
    fig.subplots_adjust(left=.105, right=.987, bottom=.18, top=.89, wspace=.27)
    align_panels(fig, list(axes))
    save_panels(fig, FIGURES / "fig_exact_data_reconstruction_rev.pdf")
    plt.close(fig)


def plot_stencils():
    """Show both irregular-point sets and the interface on the M=128 grid."""
    problem = STOKES_STARFISH
    xx, yy, h = grid_coordinates(128, problem.box)
    x, y = xx[1:-1, 1:-1], yy[1:-1, 1:-1]
    c_delta = classify_crossing(xx, yy, h, LAPLACE_OFFSETS, problem.inside)
    c_grad = classify_crossing(xx, yy, h, GRADIENT_OFFSETS, problem.inside)
    extra = c_grad & ~c_delta
    boundary = problem.geometry(np.linspace(0, 2 * np.pi, 1200))[0]

    fig, ax = plt.subplots(figsize=(4.4, 4.35))
    interface, = ax.plot(*boundary.T, color="0.12", linewidth=.8,
                         zorder=3, label=r"$\Gamma_S$")
    laplace = ax.scatter(x[c_delta], y[c_delta], s=5, color="#0072B2",
                         linewidths=0, label=r"$\mathcal{C}_\Delta$")
    gradient = ax.scatter(x[extra], y[extra], s=7, marker="^", color="#D55E00",
                          linewidths=.68,
                          label=r"$\mathcal{C}_\nabla\backslash\mathcal{C}_\Delta$")
    ticks = np.arange(-.8, .81, .4)
    ax.set(xlim=(-.95, .98), ylim=(-.965, .965), xticks=ticks, yticks=ticks,
           xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
    ax.legend(handles=[interface, laplace, gradient], loc="upper right", ncol=1,
               frameon=True, markerscale=1.8, fontsize=8, facecolor="white")
    fig.subplots_adjust(left=.14, right=.98, bottom=.12, top=.936)
    fig.savefig(FIGURES / "fig_allgrid_stencils_rev.pdf")
    plt.close(fig)


def main():
    configure()
    plot_allgrid_convergence()
    plot_exact_data_reconstruction()
    plot_stencils()


if __name__ == "__main__":
    main()
