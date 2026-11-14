"""Small version of the paper's convergence tests for the kite and starfish.

Laplace: S f + D g for the nonconvex kite, with densities taken from an
exact harmonic solution. Stokes: the Stokeslet and stresslet integrals
(with the local corrections) for the starfish, and the pressure, compared
with an exact Stokes solution. Targets are 97 points at signed distance
b = lambda*delta from the curve, lambda in {-0.5, 0, 0.5}; delta = h^q.

This is the same computation as experiments.run_swapped_production
(run_local), without the 2N quadrature check and stopping at N = 8192
instead of 16384; it takes about 15 seconds. Its errors agree with the
saved values in results/swapped_low_order/local/errors.csv.
Output: examples/output/kite_and_starfish.txt. Run from the repository root:

    python3 -m examples.kite_and_starfish
"""
import numpy as np

from experiments.bie_evaluators import (PeriodicSamples, eval_laplace_regularized,
                                        eval_pressure_regularized,
                                        eval_stokes_regularized)
from experiments.config import ORDER_TO_Q
from experiments.local_drivers import _targets
from experiments.swapped_low_order import LAPLACE_KITE, STOKES_STARFISH

LEVELS = (256, 512, 1024, 2048, 4096, 8192)
LAMBDAS = (-0.5, 0.0, 0.5)


def side(lam):
    return "average" if lam == 0 else ("inside" if lam < 0 else "outside")


def laplace_error(n, order):
    delta = (2 * np.pi / n)**ORDER_TO_Q[order]
    f, g = LAPLACE_KITE.boundary_densities(2 * np.pi * np.arange(n) / n)
    error = 0.0
    for theta, targets, b, lam in _targets(LAPLACE_KITE.geometry, delta, LAMBDAS):
        value = eval_laplace_regularized(
            LAPLACE_KITE.geometry, PeriodicSamples(f), PeriodicSamples(g),
            targets, theta, b, delta, n, order)
        error = max(error, np.max(np.abs(value - LAPLACE_KITE.exact(targets, side(lam)))))
    return error


def stokes_errors(n, order):
    delta = (2 * np.pi / n)**ORDER_TO_Q[order]
    q, f = STOKES_STARFISH.boundary_data(2 * np.pi * np.arange(n) / n)
    velocity = pressure = 0.0
    for theta, targets, b, lam in _targets(STOKES_STARFISH.geometry, delta, LAMBDAS):
        u = eval_stokes_regularized(STOKES_STARFISH.geometry, q, f, targets,
                                    theta, b, delta, order)
        p = eval_pressure_regularized(STOKES_STARFISH.geometry, q, f, targets,
                                      theta, b, delta, order)
        u_exact, p_exact = STOKES_STARFISH.exact(targets, side(lam))
        velocity = max(velocity, np.max(np.linalg.norm(u - u_exact, axis=-1)))
        pressure = max(pressure, np.max(np.abs(p - p_exact)))
    return velocity, pressure


def report(title, columns, compute):
    print(title)
    print(f"{'p':>2} {'N':>6} " + " ".join(f"{c:>10} {'rate':>5}" for c in columns))
    for order in (3, 5, 7):
        previous = None
        for n in LEVELS:
            errors = np.atleast_1d(compute(n, order))
            cells = []
            for i, e in enumerate(errors):
                rate = "" if previous is None else f"{np.log2(previous[i] / e):5.2f}"
                cells.append(f"{e:10.2e} {rate:>5}")
            print(f"{order:2d} {n:6d} " + " ".join(cells))
            previous = errors
    print()


def main():
    report("Laplace, kite: max error of S f + D g", ("error",), laplace_error)
    report("Stokes, starfish: max errors", ("velocity", "pressure"), stokes_errors)


if __name__ == "__main__":
    main()
