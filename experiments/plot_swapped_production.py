"""Figures for the swapped low-order production experiments."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .swapped_low_order import (laplace_kite_geometry,
                                stokes_starfish_geometry)


ROOT = Path("results/swapped_low_order")
FIGURES = Path("figures")
STYLE = {3: ("#0072B2", "o"), 5: ("#D55E00", "s"),
         7: ("#009E73", "^")}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "lines.linewidth": 1.25})


def _rows():
    with (ROOT / "local" / "errors.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    extension = ROOT / "local" / "p7_extension.csv"
    if extension.exists():
        with extension.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    for row in rows:
        for key in ("n_gamma", "h_gamma", "order", "b_over_delta", "error_inf"):
            row[key] = float(row[key])
    return rows


def _panel(ax, label):
    ax.text(0.02, 0.98, f"({label})", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold")


def _reference(ax, x, y, slope):
    segment = np.geomspace(min(x), np.sqrt(min(x) * max(x)), 20)
    reference = min(y) * 2.5 * (segment / min(x))**slope
    ax.loglog(segment, reference, "--", color="0.35", lw=.9,
              label=fr"slope ${slope}$")


def _series(rows, problem, field, order, lambdas=(-.5, 0., .5)):
    grouped = {}
    for row in rows:
        if (row["problem"] == problem and row["field"] == field
                and int(row["order"]) == order
                and row["b_over_delta"] in lambdas):
            n = int(row["n_gamma"])
            grouped[n] = max(grouped.get(n, 0.0), row["error_inf"])
    return np.array(sorted((2 * np.pi / n, value)
                           for n, value in grouped.items()))


def _format_convergence(ax, ylabel="maximum error"):
    ax.set(xlabel=r"$h$", ylabel=ylabel)
    ax.grid(which="both", alpha=.25)
    ax.legend()


def plot_local_laplace(rows):
    from .fixed_width_paper import panel
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for order, slope in ((3, 2), (5, 4), (7, 5)):
        data = _series(rows, "laplace_kite", "laplace", order)
        color, marker = STYLE[order]
        axes[0].loglog(data[:, 0], data[:, 1], color=color, marker=marker,
                       markersize=3.5, label=fr"$p={order}$")
        _reference(axes[0], data[:, 0], data[:, 1], slope)
    panel(axes[1], "kite_laplace", "laplace")
    for lam, marker in zip((0., .5, 1., 2.), ("o", "s", "^", "D")):
        data = _series(rows, "laplace_kite", "laplace", 7,
                       (lam,) if lam == 0 else (-lam, lam))
        axes[2].loglog(data[:, 0], data[:, 1], marker=marker, markersize=3.5,
                       label=fr"$|\lambda|={lam:g}$")
    for ax in (axes[0], axes[2]):
        _format_convergence(ax)
    for ax, label, title in zip(axes, "abc", (r"$\delta=h^q$",
            r"$p=7,\ \delta=\kappa h_{\rm phys}$",
            r"$p=7,\ \delta=h^{5/7}$")):
        _panel(ax, label)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=.7, w_pad=.7)
    fig.savefig(FIGURES / "fig_laplace_kite.pdf")
    plt.close(fig)


def plot_local_stokes(rows):
    from .fixed_width_paper import panel
    fig, grid = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for ax, field in zip(grid[0], ("velocity", "pressure")):
        for order, slope in ((3, 2), (5, 4), (7, 5)):
            data = _series(rows, "stokes_starfish", field, order)
            color, marker = STYLE[order]
            ax.loglog(data[:, 0], data[:, 1], color=color, marker=marker,
                      markersize=4, label=fr"$p={order}$")
            _reference(ax, data[:, 0], data[:, 1], slope)
        _format_convergence(ax)
        ax.set_title(field + r": $\delta=h^q$", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    for ax, field in zip(grid[1], ("velocity", "pressure")):
        panel(ax, "starfish_stokes", field)
        ax.set_title(field + r": $p=7,\ \delta=\kappa h_{\rm phys}$", fontsize=9)
    for ax, label in zip(grid.flat, "abcd"):
        _panel(ax, label)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_stokes_starfish.pdf")
    plt.close(fig)


def _grid_reports(problem):
    return [json.loads(path.read_text()) for path in sorted(
        (ROOT / "allgrid" / problem).glob("M*.json"),
        key=lambda path: int(path.stem[1:]))]


def plot_allgrid_convergence(ax=None, norm="inf"):
    laplace = _grid_reports("laplace")
    stokes = _grid_reports("stokes")
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = ((laplace, "laplace", "Laplace", "o", "#0072B2"),
              (stokes, "pressure", "Stokes pressure", "s", "#D55E00"),
              (stokes, "velocity", "Stokes velocity", "^", "#009E73"))
    for rows, field, label, marker, color in series:
        ax.loglog([row["h_c"] for row in rows],
                  [row[field][norm] for row in rows], marker=marker,
                  color=color, markersize=4, label=label)
    x = np.array([row["h_c"] for row in laplace])
    y = np.array([row["laplace"][norm] for row in laplace])
    _reference(ax, x, y, 4)
    ax.set(xlabel=r"$h_c$", ylabel="maximum error" if norm == "inf"
           else r"discrete $L^2$ error")
    ax.grid(which="both", alpha=.25)
    ax.legend()
    if standalone:
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_allgrid_convergence.pdf")


def plot_allgrid_errors(axes=None):
    laplace = np.load(ROOT / "allgrid" / "laplace" / "fields_M1024.npz")
    stokes = np.load(ROOT / "allgrid" / "stokes" / "fields_M1024.npz")
    fields = ((laplace, np.abs(laplace["error"]), laplace_kite_geometry,
               "Laplace error"),
              (stokes, np.abs(stokes["pressure_error"]),
               stokes_starfish_geometry, "pressure error"),
              (stokes, np.linalg.norm(stokes["velocity_error"], axis=-1),
               stokes_starfish_geometry, "velocity error"))
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.55))
    else:
        fig = axes[0].figure
    for ax, (data, error, geometry, title), label in zip(
            axes, fields, "abc" if standalone else "cde"):
        positive = error[error > 0]
        low = max(-16, np.floor(np.log10(np.quantile(positive, .01))))
        high = np.ceil(np.log10(error.max()))
        image = ax.pcolormesh(data["xx"], data["yy"],
            np.log10(np.maximum(error, 1e-16)), shading="auto",
            rasterized=True, cmap="magma", vmin=low, vmax=high)
        boundary = geometry(np.linspace(0, 2 * np.pi, 1200))[0]
        ax.plot(boundary[:, 0], boundary[:, 1], color="cyan", lw=.75)
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal", title=title)
        fig.colorbar(image, ax=ax, label=r"$\log_{10}|e|$" if standalone else "")
        _panel(ax, label)
    if standalone:
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_allgrid_errors.pdf")


def plot_allgrid_combined():
    """Two convergence panels above three finest-grid error maps."""
    fig = plt.figure(figsize=(10, 7))
    grid = fig.add_gridspec(2, 6, height_ratios=(1.15, 1),
                          left=.075, right=.97, bottom=.08, top=.96,
                          wspace=1.15, hspace=.32)
    top = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:])]
    for ax, norm, label in zip(top, ("inf", "l2"), "ab"):
        plot_allgrid_convergence(ax, norm)
        _panel(ax, label)
        ax.legend(loc="lower right")
    top[1].sharex(top[0])
    plot_allgrid_errors([fig.add_subplot(grid[1, 2*i:2*i+2]) for i in range(3)])
    fig.savefig(FIGURES / "fig_allgrid_combined.pdf")
    plt.close(fig)


def plot_stencils(m=128):
    """Irregular points for the starfish, recomputed on a grid coarse enough to show them."""
    from .allgrid import (GRADIENT_OFFSETS, LAPLACE_OFFSETS, classify_crossing,
                          grid_coordinates)
    from .swapped_low_order import STOKES_STARFISH
    xx, yy, h = grid_coordinates(m, STOKES_STARFISH.box)
    centers = np.stack((xx[1:-1, 1:-1], yy[1:-1, 1:-1]), axis=-1)
    c_delta = classify_crossing(xx, yy, h, LAPLACE_OFFSETS, STOKES_STARFISH.inside)
    c_gradient = classify_crossing(xx, yy, h, GRADIENT_OFFSETS, STOKES_STARFISH.inside)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    boundary = stokes_starfish_geometry(np.linspace(0, 2 * np.pi, 1200))[0]
    ax.plot(boundary[:, 0], boundary[:, 1], "k", label=r"$\Gamma_S$")
    ax.scatter(centers[..., 0][c_delta], centers[..., 1][c_delta], s=7,
               label=r"$\mathcal{C}_\Delta$")
    extra = c_gradient & ~c_delta
    ax.scatter(centers[..., 0][extra], centers[..., 1][extra], s=11, marker="x",
               label=r"$\mathcal{C}_\nabla\setminus\mathcal{C}_\Delta$")
    ax.set_aspect("equal")
    ax.legend()
    ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_allgrid_stencils.pdf")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    plot_local_laplace(rows)
    plot_local_stokes(rows)
    plot_allgrid_convergence()
    plot_allgrid_errors()
    plot_allgrid_combined()
    plot_stencils()


if __name__ == "__main__":
    main()
