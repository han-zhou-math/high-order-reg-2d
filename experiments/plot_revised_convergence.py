"""Render the revised paper's circle, kite, and starfish convergence figures.

Run from the repository root with python3 -m experiments.plot_revised_convergence.
The figures use saved errors and the fit ranges documented in the captions.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator
import numpy as np

from .fixed_width_paper import STYLE as WIDTH_STYLE, resolved, series
from .plot_swapped_production import _rows, _series
from .config import FIGURE_ROOT, RESULT_ROOT
from .plotting import ORDER_STYLE, _aggregate, _numeric
from .result_io import read_rows

from .publication_style import configure, align_panels, save_panels

ORDER_SLOPES = {3: 2, 5: 4, 7: 5}
CIRCLE_FITS = {3: (2048, 16384), 5: (2048, 16384), 7: (512, 4096)}
SOLUTION_FITS = {3: (4096, 16384), 5: (4096, 16384), 7: (8192, 32768)}
GUIDE_STYLE = dict(color="0.42", linestyle="--", linewidth=.9, zorder=1)


def _format(ax, label, title, xlabel):
    ax.set_title(f"({label})  {title}", loc="left", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.grid(False, which="both")
    ax.yaxis.set_major_locator(LogLocator(numticks=7))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", length=3, labelsize=8)
    ax.tick_params(axis="x", which="minor", length=1.8)


def _error_series(ax, x, y, color, marker, linestyle="-"):
    """Draw all saved values with filled markers."""
    ax.loglog(x, y, color=color, linestyle=linestyle, marker=marker,
              markerfacecolor=color, markersize=4.1, zorder=2)


def _slope_guide(ax, x, y, slope, factor):
    """Anchor a short reference segment within the resolved mesh levels."""
    reference = y[0] * factor * (x / x[0])**slope
    ax.loglog(x, reference, **GUIDE_STYLE)
    ax.annotate(str(slope), (x[-1], reference[-1]), xytext=(3, 0),
                textcoords="offset points", color="0.33", fontsize=8,
                va="center")


def _order_panel(ax, values, fits):
    guide_factors = (4.0, 7.5, .32)
    for (order, slope), factor in zip(ORDER_SLOPES.items(), guide_factors):
        x, y = values(order).T
        n = np.rint(2 * np.pi / x).astype(int)
        first, last = fits[order]
        included = (first <= n) & (n <= last)
        color, marker = ORDER_STYLE[order]
        _error_series(ax, x, y, color, marker)
        _slope_guide(ax, x[included][:3], y[included][:3], slope, factor)


def _width_panel(ax, geometry, field):
    spacing = "h_gamma" if geometry == "circle" else "h_phys"
    for width, (color, marker, linestyle) in WIDTH_STYLE.items():
        rows = series(geometry, field, width)
        x = np.array([row[spacing] for row in rows])
        y = np.array([row["error"] for row in rows])
        included = np.array([resolved(row) for row in rows])
        _error_series(ax, x, y, color, marker, linestyle)
        if width == 3:
            x_guide, y_guide = x[included][1:4][::-1], y[included][1:4][::-1]
            _slope_guide(ax, x_guide, y_guide, 7, .23)


def _order_handles():
    return [Line2D([], [], color=color, marker=marker, label=fr"$p={order}$")
            for order, (color, marker) in ORDER_STYLE.items()] + [
        Line2D([], [], **GUIDE_STYLE, label="reference slopes")]


def _width_handles():
    return [Line2D([], [], color=color, marker=marker, linestyle=style,
                   label=fr"$c={width}$")
            for width, (color, marker, style) in WIDTH_STYLE.items()] + [
        Line2D([], [], **GUIDE_STYLE, label="slope 7")]


def _two_row_figure(panels, width_geometry, fits, spacing, output):
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.85), sharey="row", sharex="row")
    for column, (title, field, values) in enumerate(panels):
        top, bottom = axes[:, column]
        _order_panel(top, values, fits)
        top._publication_handles = _order_handles()
        bottom._publication_handles = _width_handles()
        _width_panel(bottom, width_geometry, field)
        _format(top, "ab"[column], fr"{title}, $\delta=h^q$", "$h$")
        _format(bottom, "cd"[column], fr"{title}, $p=7$, $\delta=c{spacing}$",
                f"${spacing}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Maximum error")
    fig.subplots_adjust(left=.117, right=.98, top=.858, bottom=.11,
                        hspace=.75, wspace=.14)
    align_panels(fig, list(axes.flat), rows=2)
    _save(fig, output)


def _save(fig, output):
    save_panels(fig, FIGURE_ROOT / output)
    plt.close(fig)


def plot_circle():
    rows = _numeric(read_rows(RESULT_ROOT / "circle_layers/errors.csv"))

    def order_values(field):
        groups = _aggregate(rows, f"circle_{field}")

        def values(order):
            return np.array(sorted((2 * np.pi / n, error)
                                   for (p, n), error in groups.items() if p == order))

        return values

    panels = [(title, field, order_values(field))
              for title, field in (("Single layer", "single"), ("Double layer", "double"))]
    _two_row_figure(panels, "circle", CIRCLE_FITS, "h",
                    "fig_circle_layers_convergence_rev.pdf")


def plot_kite(rows):
    fig = plt.figure()
    axes = [fig.add_subplot(2, 2, index + 1) for index in range(3)]
    _order_panel(axes[0], lambda p: _series(rows, "laplace_kite", "laplace", p),
                 SOLUTION_FITS)
    _width_panel(axes[1], "kite_laplace", "laplace")
    colors = [ORDER_STYLE[p][0] for p in ORDER_STYLE] + ["#882255"]
    offsets = (0., .5, 1., 2.)
    handles = []
    for distance, color, marker in zip(offsets, colors, ("o", "s", "^", "D")):
        lambdas = (-distance, distance) if distance else (0.,)
        x, y = _series(rows, "laplace_kite", "laplace", 7, lambdas).T
        _error_series(axes[2], x, y, color, marker)
        handles.append(Line2D([], [], color=color, marker=marker,
                              label=fr"$|\lambda|={distance:g}$"))
    for ax, label, title, xlabel, keys in zip(
            axes, "abc", (r"$\delta=h^q$", r"$p=7$, $\delta=ch_{\mathrm{phys}}$",
                          r"$p=7$, $\delta=h^{5/7}$"),
            ("$h$", r"$h_{\mathrm{phys}}$", "$h$"),
            (_order_handles(), _width_handles(), handles)):
        _format(ax, label, title, xlabel)
        ax._publication_handles = keys
    axes[0].set_ylabel("Maximum error")
    fig.subplots_adjust(left=.114, right=.985, top=.89, bottom=.31, wspace=.44)
    align_panels(fig, axes, rows=2, center_last=True)
    _save(fig, "fig_laplace_kite_rev.pdf")


def plot_starfish(rows):
    def order_values(field):
        return lambda order: _series(rows, "stokes_starfish", field, order)

    panels = [(field.title(), field, order_values(field))
              for field in ("velocity", "pressure")]
    _two_row_figure(panels, "starfish_stokes", SOLUTION_FITS, r"h_{\mathrm{phys}}",
                    "fig_stokes_starfish_rev.pdf")


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 9.5,
                         "lines.linewidth": 1.25})
    configure()
    rows = _rows()
    plot_circle()
    plot_kite(rows)
    plot_starfish(rows)


if __name__ == "__main__":
    main()
