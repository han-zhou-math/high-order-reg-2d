"""Regularized single-layer potential for the unit circle.

Uses only the core package `regbie`. The density is f(t) = 1 + 0.7 cos 3t,
and targets lie at signed distance b = lambda*delta from the circle
(b < 0 inside, b > 0 outside, b = 0 on the circle). With G(r) = log(r)/(2 pi),
the exact potential at rho*(cos theta, sin theta) is

    S f = log(max(1, rho)) - (0.7/6) min(rho, 1/rho)^3 cos 3 theta.

The smoothing width is delta = h^q with h = 2 pi / N, so the error should
decrease like h^(p q): about h^2, h^4, h^5 for p = 3, 5, 7.

Run from the repository root:  python3 -m examples.circle_single_layer
"""
import numpy as np

from regbie.regularized_slp import circle, single_layer

ORDER_TO_Q = {3: 2 / 3, 5: 4 / 5, 7: 5 / 7}
LEVELS = (128, 256, 512, 1024, 2048)
LAMBDAS = (-0.5, 0.0, 0.5)


def density(t):
    return 1 + 0.7 * np.cos(3 * t)


def exact(targets):
    rho = np.linalg.norm(targets, axis=-1)
    theta = np.arctan2(targets[:, 1], targets[:, 0])
    return (np.log(np.maximum(1.0, rho))
            - 0.7 / 6 * np.minimum(rho, 1 / rho)**3 * np.cos(3 * theta))


def main():
    theta = 2 * np.pi * (np.arange(97) + 0.37) / 97
    normal = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    print("max error over 97 targets and lambda = b/delta in {-0.5, 0, 0.5}")
    print(f"{'p':>2} {'N':>6} {'delta':>10} {'error':>10} {'rate in h':>10}")
    for order, q in ORDER_TO_Q.items():
        previous = None
        for n in LEVELS:
            h = 2 * np.pi / n
            delta = h**q
            error = 0.0
            for lam in LAMBDAS:
                b = np.full(len(theta), lam * delta)
                targets = normal * (1 + b)[:, None]
                value = single_layer(circle, density, targets, delta, n,
                                     order=order, b=b)
                error = max(error, np.max(np.abs(value - exact(targets))))
            rate = "" if previous is None else f"{np.log2(previous / error):10.2f}"
            print(f"{order:2d} {n:6d} {delta:10.3e} {error:10.2e} {rate}")
            previous = error


if __name__ == "__main__":
    main()
