"""Plot saved experiment data without recomputing any layer potential."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .config import FIGURE_ROOT, KITE_BOX, RESULT_ROOT, STARFISH_BOX
from .geometries import laplace_starfish_geometry, stokes_kite_geometry
from .manufactured_laplace import exact_piecewise_laplace
from .manufactured_stokes import exact_piecewise_stokes
from .result_io import read_rows

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "lines.linewidth": 1.25})
ORDER_STYLE = {3: ("#0072B2", "o"), 5: ("#D55E00", "s"),
               7: ("#009E73", "^")}


def _numeric(rows):
    for row in rows:
        for key, value in tuple(row.items()):
            try: row[key] = float(value)
            except (TypeError, ValueError): pass
    return rows


def _aggregate(rows, solution, lambdas=None):
    selected = [r for r in rows if r["solution"] == solution
                and (lambdas is None or r["lambda"] in lambdas)]
    groups = {}
    for row in selected:
        key = (int(row["order"]), int(row["n_gamma"]))
        groups[key] = max(groups.get(key, 0.0), row["error_inf"])
    return groups


def _reference(ax, x, y, slope, label=True):
    lo, hi = np.min(x), np.max(x)
    segment = np.geomspace(lo, np.sqrt(lo * hi), 20)
    anchor = np.min(y) * 2.5
    ref = anchor * (segment / lo)**slope
    ax.loglog(segment, ref, "--", color="0.35", lw=.9, alpha=.8,
              label=fr"slope ${slope}$" if label else None)


def _panel(ax, label):
    ax.text(0.02, 0.98, f"({label})", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold")


def plot_circle_layers(data=RESULT_ROOT / "circle_layers/errors.csv",
                       output=FIGURE_ROOT / "fig_circle_layers_convergence.pdf"):
    rows = _numeric(read_rows(data)); fig, grid = plt.subplots(2, 2, figsize=(7.2, 5.6))
    axes = grid[0]
    for ax, solution, title in zip(axes, ("circle_single", "circle_double"),
                                   ("single layer", "double layer")):
        groups = _aggregate(rows, solution)
        for order, slope in ((3, 2), (5, 4), (7, 5)):
            values = sorted((2 * np.pi / n, e) for (p, n), e in groups.items() if p == order)
            x, y = map(np.asarray, zip(*values)); color, marker = ORDER_STYLE[order]
            ax.loglog(x, y, marker=marker, color=color,
                      label=fr"$p={order}$",
                      markersize=4)
            _reference(ax, x, y, slope)
        ax.set(xlabel=r"$h$", ylabel="maximum error")
        ax.set_title(title + r": $\delta=h^q$", fontsize=9)
        ax.grid(which="both", alpha=.25); ax.legend(fontsize=7, loc="lower right")
    from .fixed_width_paper import panel
    for ax, field, title in zip(grid[1], ("single", "double"), ("single layer", "double layer")):
        panel(ax, "circle", field)
        ax.set_title(title + r": $p=7,\ \delta=\kappa h$", fontsize=9)
    for ax, label in zip(grid.flat, "abcd"):
        _panel(ax, label)
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output)


def _field_grid(box, count=350):
    x = np.linspace(box[0], box[1], count); y = np.linspace(box[2], box[3], count)
    return np.meshgrid(x, y)


def plot_laplace_starfish(data=RESULT_ROOT / "laplace_starfish/errors.csv",
                          output=FIGURE_ROOT / "fig_laplace_starfish.pdf"):
    rows = _numeric(read_rows(data)); fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    xx, yy = _field_grid(STARFISH_BOX); points = np.stack((xx, yy), axis=-1)
    shown = axes[0].pcolormesh(xx, yy, exact_piecewise_laplace(points),
                               shading="auto", rasterized=True, cmap="viridis")
    t = np.linspace(0, 2 * np.pi, 1000); boundary = laplace_starfish_geometry(t)[0]
    axes[0].plot(boundary[:, 0], boundary[:, 1], "k", lw=1); fig.colorbar(shown, ax=axes[0])
    groups = _aggregate(rows, "combined_laplace", lambdas={0.0, 0.5})
    for order, slope in ((3, 2), (5, 4), (7, 5)):
        values = sorted((2 * np.pi / n, e) for (p, n), e in groups.items() if p == order)
        x, y = map(np.asarray, zip(*values)); color, marker = ORDER_STYLE[order]
        axes[1].loglog(x, y, marker=marker, color=color, markersize=4,
                       label=f"$r={order}$")
        _reference(axes[1], x, y, slope)
    for lam in (0.0, 0.5, 1.0, 2.0):
        selected = _aggregate(rows, "combined_laplace", lambdas={lam})
        values = sorted((2 * np.pi / n, e) for (p, n), e in selected.items() if p == 7)
        x, y = map(np.asarray, zip(*values));
        axes[2].loglog(x, y, marker=("o", "s", "^", "D")[(0.0, .5, 1.0, 2.0).index(lam)],
                       markersize=4, label=fr"$\lambda={lam:g}$")
    for ax in axes[1:]: ax.grid(which="both", alpha=.25); ax.legend(fontsize=8); ax.set_xlabel(r"$h_\Gamma$")
    axes[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
    axes[1].set_ylabel("maximum error"); axes[2].set_ylabel("maximum error")
    for ax, label in zip(axes, "abc"): _panel(ax, label)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output)


def plot_stokes_kite(data=RESULT_ROOT / "stokes_kite/errors.csv",
                     output=FIGURE_ROOT / "fig_stokes_kite.pdf"):
    rows = _numeric(read_rows(data)); fig, axes = plt.subplots(2, 2, figsize=(8.4, 7))
    xx, yy = _field_grid(KITE_BOX); points = np.stack((xx, yy), axis=-1)
    u, p = exact_piecewise_stokes(points); t = np.linspace(0, 2 * np.pi, 1000)
    boundary = stokes_kite_geometry(t)[0]
    for ax, field, title in ((axes[0, 0], np.linalg.norm(u, axis=-1), r"exact $|u|$"),
                             (axes[0, 1], p, "exact pressure")):
        shown = ax.pcolormesh(xx, yy, field, shading="auto", rasterized=True,
                              cmap="viridis")
        ax.plot(boundary[:, 0], boundary[:, 1], "k", lw=1)
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
        fig.colorbar(shown, ax=ax)
    for ax, solution, title in ((axes[1, 0], "combined_stokes_velocity", "velocity error"),
                                (axes[1, 1], "combined_stokes_pressure", "pressure error")):
        groups = _aggregate(rows, solution)
        for order, slope in ((3, 2), (5, 4), (7, 5)):
            values = sorted((2 * np.pi / n, e) for (pp, n), e in groups.items() if pp == order)
            x, y = map(np.asarray, zip(*values)); color, marker = ORDER_STYLE[order]
            ax.loglog(x, y, marker=marker, color=color, markersize=4,
                      label=f"$r={order}$")
            _reference(ax, x, y, slope)
        ax.set(xlabel=r"$h_\Gamma$", ylabel="maximum error")
        ax.grid(which="both", alpha=.25); ax.legend(fontsize=8)
    for ax, label in zip(axes.flat, "abcd"): _panel(ax, label)
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output)


def plot_allgrid_convergence(output=FIGURE_ROOT / "fig_allgrid_convergence.pdf"):
    lap = _numeric(read_rows(RESULT_ROOT / "allgrid_laplace/errors.csv"))
    sto = _numeric(read_rows(RESULT_ROOT / "allgrid_stokes/errors.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = ((lap, "error_inf", "Laplace", "o", "#0072B2"),
              (sto, "pressure_error_inf", "Stokes pressure", "s", "#D55E00"),
              (sto, "velocity_error_inf", "Stokes velocity", "^", "#009E73"))
    for rows, key, label, marker, color in series:
        rows = sorted(rows, key=lambda row: row["h_c"])
        ax.loglog([r["h_c"] for r in rows], [r[key] for r in rows],
                  marker=marker, color=color, markersize=4, label=label)
    x = np.asarray([r["h_c"] for r in lap]); _reference(ax, x, np.asarray([r["error_inf"] for r in lap]), 4)
    ax.set(xlabel=r"$h_c$", ylabel="maximum error"); ax.grid(which="both", alpha=.25)
    ax.legend(); output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output)


def plot_allgrid_errors(output=FIGURE_ROOT / "fig_allgrid_errors.pdf"):
    lap = np.load(RESULT_ROOT / "allgrid_laplace/finest_fields.npz")
    sto = np.load(RESULT_ROOT / "allgrid_stokes/finest_fields.npz")
    metadata = json.loads((RESULT_ROOT / "allgrid_stokes/metadata.json").read_text())
    floor = float(metadata["plot_error_floor"])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    fields = (np.abs(lap["error"]), np.abs(sto["pressure_error"]),
              np.linalg.norm(sto["velocity_error"], axis=-1))
    for ax, field, title in zip(axes, fields, ("Laplace", "pressure", "velocity")):
        shown = ax.pcolormesh(lap["xx"] if title == "Laplace" else sto["xx"],
                              lap["yy"] if title == "Laplace" else sto["yy"],
                              np.log10(np.maximum(field, floor)), shading="auto",
                              rasterized=True, cmap="magma", vmin=-12, vmax=-3)
        geometry = laplace_starfish_geometry if title == "Laplace" else stokes_kite_geometry
        boundary = geometry(np.linspace(0, 2 * np.pi, 1000))[0]
        ax.plot(boundary[:, 0], boundary[:, 1], color="cyan", lw=.8)
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$", aspect="equal")
        fig.colorbar(shown, ax=ax, label=r"$\log_{10}$ error")
    for ax, label in zip(axes, "abc"): _panel(ax, label)
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output)


def plot_allgrid_stencils(output=FIGURE_ROOT / "fig_allgrid_stencils.pdf"):
    data = np.load(RESULT_ROOT / "allgrid_stokes/finest_fields.npz")
    xx, yy = data["xx"], data["yy"]
    centers = np.stack((xx[1:-1, 1:-1], yy[1:-1, 1:-1]), axis=-1)
    c_delta, c_gradient = data["c_delta"], data["c_gradient"]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    boundary = stokes_kite_geometry(np.linspace(0, 2 * np.pi, 1200))[0]
    ax.plot(boundary[:, 0], boundary[:, 1], "k", label=r"$\Gamma_S$")
    ax.scatter(centers[..., 0][c_delta], centers[..., 1][c_delta], s=7,
               label=r"$\mathcal{C}_\Delta$")
    extra = c_gradient & ~c_delta
    ax.scatter(centers[..., 0][extra], centers[..., 1][extra], s=11, marker="x",
               label=r"$\mathcal{C}_\nabla\setminus\mathcal{C}_\Delta$")
    ax.set_aspect("equal"); ax.legend(); ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")
    output.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output)
