"""Using regbie.api: Laplace and Stokes layer potentials on an ellipse given by
boundary points, at targets on, near and far from it.

Laplace: u = Re(z^3) is harmonic, so S(-du/dn) + D(u) = u inside, 0 outside.
Stokes: the rigid rotation u = (-y, x), p = 0 has zero traction on any curve,
so D q with q = u equals u inside and 0 outside.

Run from the repository root:  python3 -m examples.layer_potentials
The C++ version is cpp/examples/layer_potentials.cpp.
"""
import numpy as np

from regbie.api import Curve, LayerPotentials


def main():
    n = 512
    t = 2 * np.pi * np.arange(n) / n
    curve = Curve.from_points(np.stack((1.4 * np.cos(t), 0.8 * np.sin(t)), axis=-1))
    layers = LayerPotentials(curve, n)  # order 7, delta = 3 * node spacing

    x, _, _, normal = layers.nodes[:4]
    grad = np.stack((3 * x[:, 0]**2 - 3 * x[:, 1]**2, -6 * x[:, 0] * x[:, 1]), axis=-1)
    f = -np.sum(grad * normal, axis=-1)
    g = x[:, 0]**3 - 3 * x[:, 0] * x[:, 1]**2

    p, _, _, n1 = curve(np.array([1.0]))[:4]
    offsets = np.array([-0.5, -0.02, -1e-3, 0.0, 1e-3, 0.02, 0.5])
    targets = p + offsets[:, None] * n1
    u = targets[:, 0]**3 - 3 * targets[:, 0] * targets[:, 1]**2
    exact = np.where(offsets < 0, u, np.where(offsets > 0, 0.0, 0.5 * u))
    value = layers.single_layer(f, targets) + layers.double_layer(g, targets)
    print(f"delta = {layers.delta:.3e}")
    print(f"{'b':>9} {'S f + D g':>16} {'exact':>16}")
    for b, v, e in zip(offsets, value, exact):
        print(f"{b:9.3g} {v:16.12f} {e:16.12f}")

    q = np.stack((-x[:, 1], x[:, 0]), axis=-1)
    velocity = layers.stresslet(q, targets)
    pressure = layers.stokes_pressure(None, q, targets)
    print(f"\n{'b':>9} {'D q (exact (-y, x) inside)':>30} {'pressure':>12}")
    for b, v, pr in zip(offsets, velocity, pressure):
        print(f"{b:9.3g} {v[0]:14.10f} {v[1]:14.10f} {pr:12.2e}")


if __name__ == "__main__":
    main()
