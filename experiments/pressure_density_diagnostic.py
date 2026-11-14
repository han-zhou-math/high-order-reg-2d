"""Compare analytic and Fourier pressure densities in the current starfish test."""
import csv
from pathlib import Path
import time

import numpy as np

from regbie.regularized_dlp import double_layer
from regbie.regularized_slp import single_layer
from .bie_evaluators import PeriodicSamples, curve_adapter
from .boundary_data import pressure_densities
from .swapped_low_order import (stokes_starfish_geometry, stokes_boundary_data,
                                stokes_pressure, stokes_grad_pressure)


def main():
    root = Path("results/paper_revision")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    geometry = stokes_starfish_geometry
    curve = curve_adapter(geometry)
    for n in (4096, 8192, 16384):
        t = 2 * np.pi * np.arange(n) / n
        x, _, _, normal, _, _ = geometry(t)
        q, f = stokes_boundary_data(t)
        numerical = pressure_densities(geometry, q, f)
        analytic_phi = -np.sum(stokes_grad_pressure(x) * normal, axis=-1)
        analytic_psi = -stokes_pressure(x)
        density_error_phi = np.max(np.abs(numerical.phi - analytic_phi))
        density_error_psi = np.max(np.abs(numerical.psi - analytic_psi))
        physical_h = np.max(np.linalg.norm(np.roll(x, -1, axis=0) - x, axis=-1))
        for c_delta in (2., 4.):
            delta = c_delta * physical_h
            for phase in (0., .5):
                theta = 2 * np.pi * np.arange(33) / 33 + phase * 2 * np.pi / n
                x0, _, _, n0, _, _ = geometry(theta)
                for lam in (-.5, 0., .5):
                    b = np.full(len(theta), lam * delta)
                    targets = x0 + b[:, None] * n0
                    multiplier = 1. if lam < 0 else (.5 if lam == 0 else 0.)
                    exact = multiplier * stokes_pressure(targets)
                    values = {}
                    for name, phi, psi in (
                            ("fourier", numerical.phi, numerical.psi),
                            ("analytic", analytic_phi, analytic_psi)):
                        start = time.perf_counter()
                        value = single_layer(curve, PeriodicSamples(phi), targets,
                                             delta, n, 7, b)
                        value -= double_layer(curve, PeriodicSamples(psi), targets,
                                              delta, n, 7, closest_t=theta, b=b)
                        values[name] = value
                        rows.append(dict(n_gamma=n, c_delta=c_delta,
                            h_physical=physical_h, delta=delta, phase=phase,
                            signed_lambda=lam, density_method=name,
                            error_inf=float(np.max(np.abs(value - exact))),
                            evaluation_seconds=time.perf_counter() - start,
                            density_phi_difference_inf=float(density_error_phi),
                            density_psi_difference_inf=float(density_error_psi),
                            field_difference_inf=0.))
                    diff = float(np.max(np.abs(values["fourier"] - values["analytic"])))
                    rows[-1]["field_difference_inf"] = diff
                    rows[-2]["field_difference_inf"] = diff
            print(n, c_delta, "finished", flush=True)
        with (root / "pressure_density_diagnostic.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
