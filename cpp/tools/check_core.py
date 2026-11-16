"""Compare the C++ core library with the Python code.

Run from the repository root after building:
    cpp/build/core_values > values.json
    python3 cpp/tools/check_core.py values.json
Prints the largest relative difference in each group.
"""
import json
import sys

import numpy as np
from scipy.special import erfcx, exp1

from experiments.allgrid import classify_crossing, closest_points, grid_coordinates
from experiments.bie_evaluators import eval_pressure_regularized, eval_stokes_regularized
from experiments.boundary_data import spectral_derivative
from experiments.bie_evaluators import PeriodicSamples
from experiments.swapped_low_order import KITE_SCALE, STARFISH_SCALE, STOKES_STARFISH
from regbie.all_grid_reconstruction import (LAPLACE_OFFSETS, dst_solve_nine_point, grad4,
                                            nine_point_laplacian)
from regbie.regularized_dlp import double_layer, smoothing_over_r2
from regbie.regularized_slp import coefficients, regularized_kernel, single_layer, starfish
from regbie.surface_operators import radial_factors
from regbie.verify_stokes_derivation import scaled_moment_n0

ORDERS = (3, 5, 7)


def rel(a, b):
    """Largest difference relative to the largest reference value of the group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a - b)))


def main(path):
    c = json.load(open(path))
    report = {}

    coef, mom = [], []
    for p in ORDERS:
        for lam in (0.0, 0.3, 1.0, 1.5, 3.0, 10.0):
            cs = list(coefficients(p, lam)) + [0.0] * 3
            coef.append([float(v) for v in cs[:3]])
            mom.append(scaled_moment_n0(p, lam))
    report["coefficients"] = rel(c["coefficients"], coef)
    report["scaled_moment_n0"] = rel(c["scaled_moment_n0"], mom)

    delta = 0.1
    kern, smooth, radial = [], [], []
    for p in ORDERS:
        for lam in (0.0, 0.5, 2.0):
            b = lam * delta
            for r in (0.0, 1e-3, 0.05, 0.07, 0.2, 1.0, 3.0):
                kern.append(float(regularized_kernel(r, delta, p, b)))
                smooth.append(float(smoothing_over_r2(r * r, delta, p, b)))
            for u in (0.0, 0.1, 0.49, 0.51, 2.0, 30.0):
                radial.append([float(v) for v in radial_factors(np.array(u), p, lam)])
    report["regularized_kernel"] = rel(c["regularized_kernel"], kern)
    report["smoothing_over_r2"] = rel(c["smoothing_over_r2"], smooth)
    report["radial_factors"] = rel(c["radial_factors"], radial)
    report["exp1"] = rel(c["special"]["exp1"], exp1([0.01, 0.5, 0.99, 1.0, 2.5, 10.0, 50.0]))
    report["erfcx"] = rel(c["special"]["erfcx"], erfcx([0.0, 0.3, 1.0, 3.0, 6.0, 30.0]))

    targets, b, s = [], [], []
    for i in range(12):
        si = 2 * np.pi * (i + 0.3) / 12
        bi = 0.02 * (i % 5 - 2)
        x, dx = starfish(np.array([si]))
        nrm = np.array([dx[0, 1], -dx[0, 0]]) / np.linalg.norm(dx[0])
        targets.append(x[0] + bi * nrm)
        b.append(bi)
        s.append(si)
    targets, b, s = np.array(targets), np.array(b), np.array(s)
    dens = lambda t: 1 + 0.3 * np.sin(2 * t)
    worst_s = worst_d = 0.0
    for p in ORDERS:
        sl = single_layer(starfish, dens, targets, 0.05, 256, p, b)
        dl = double_layer(starfish, dens, targets, 0.05, 256, p, closest_t=s, b=b)
        worst_s = max(worst_s, rel(c["layers"][f"single_{p}"], sl))
        worst_d = max(worst_d, rel(c["layers"][f"double_{p}"], dl))
    report["single_layer"] = worst_s
    report["double_layer"] = worst_d

    sp = STOKES_STARFISH
    st = []
    for i in range(10):
        si = 2 * np.pi * (i + 0.17) / 10
        g = sp.geometry(np.array([si]))
        st.append(g[0][0] + (0.015 * (i % 3) - 0.01) * g[3][0])
    st = np.array(st)
    ct, cb = closest_points(st, sp.geometry, sp.inside)
    q, f = sp.boundary_data(2 * np.pi * np.arange(256) / 256, sp.mu)
    pres = eval_pressure_regularized(sp.geometry, q, f, st, ct, cb, 0.04, 7, sp.mu)
    vel = eval_stokes_regularized(sp.geometry, q, f, st, ct, cb, 0.04, 7, sp.mu)
    report["closest_t"] = rel(c["stokes"]["closest_t"], ct)
    report["closest_b"] = rel(c["stokes"]["b"], cb)
    report["pressure"] = rel(c["stokes"]["pressure"], pres)
    report["velocity"] = max(rel(c["stokes"]["velocity_x"], vel[:, 0]), rel(c["stokes"]["velocity_y"], vel[:, 1]))
    report["scales"] = rel([c["stokes"]["starfish_scale"], c["stokes"]["kite_scale"]], [STARFISH_SCALE, KITE_SCALE])

    xx, yy, h = grid_coordinates(30, sp.box)
    full = np.sin(1.3 * xx + 0.4) * np.cos(0.7 * yy) + xx * yy**2
    boundary = np.zeros_like(full)
    boundary[[0, -1], :] = full[[0, -1], :]
    boundary[:, [0, -1]] = full[:, [0, -1]]
    rhs = np.cos(xx - 2 * yy)[1:-1, 1:-1]
    solved = dst_solve_nine_point(rhs, boundary, h)
    g = grad4(full, h)
    report["dst_solve"] = rel(c["grid"]["solved"], solved.ravel())
    report["grad4"] = max(rel(c["grid"]["gx"], g[..., 0].ravel()), rel(c["grid"]["gy"], g[..., 1].ravel()))
    report["nine_point"] = rel(c["grid"]["laplacian"], nine_point_laplacian(full, h).ravel())
    cross = classify_crossing(xx, yy, h, LAPLACE_OFFSETS, sp.inside)
    report["crossing_mismatches"] = int(np.sum(np.array(c["grid"]["crossing"]) != cross.ravel()))

    samples = np.exp(np.sin(2 * np.pi * np.arange(64) / 64))
    report["spectral_derivative"] = max(rel(c["periodic"]["derivative"], spectral_derivative(samples, 1)),
                                        rel(c["periodic"]["second_derivative"], spectral_derivative(samples, 2)))
    report["interpolant"] = rel(c["periodic"]["interpolant"], PeriodicSamples(samples)(np.array([0.1, 1.7, 4.0])))
    expected = [repr(v) for v in (0.1, 1e-5, 1.5e-16, 100.0, 1e16, 123456789012345680.0, 2.0 / 3, -0.0, 0.00012, 1e22)]
    report["repr_mismatches"] = [(a, b) for a, b in zip(c["repr"], expected) if a != b]

    for key, value in report.items():
        print(f"{key:22s} {value}")


if __name__ == "__main__":
    main(sys.argv[1])
