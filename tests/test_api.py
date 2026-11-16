"""Accuracy of regbie.api on the kite (Laplace) and starfish (Stokes) tests.

Curves given by functions and by points; targets on, near and far from the
curve. With n = 4096 and delta = 3 node spacings the errors are about 1e-8,
as in the fixed-width runs of experiments/fixed_width.py.
"""
import unittest

import numpy as np

from experiments.swapped_low_order import LAPLACE_KITE, STOKES_STARFISH
from regbie.api import Curve, LayerPotentials

N = 4096
TOLERANCE = 5e-8


def targets_for(curve, box, delta):
    s = np.linspace(0, 1, 41)
    xx, yy = np.meshgrid(box[0] + (box[1] - box[0]) * s, box[2] + (box[3] - box[2]) * s)
    grid = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    x, _, _, normal = curve(2 * np.pi * (np.arange(64) + 0.3) / 64)[:4]
    offsets = delta * np.array([0.0, 0.25, -0.25, 1.0, -1.0, 3.0, -3.0])
    near = (x[:, None] + offsets[None, :, None] * normal[:, None]).reshape(-1, 2)
    return np.concatenate((grid, near))


def sides(layers, targets):
    """Exact-solution branch per target: piecewise far away, by signed distance near the curve."""
    labels = np.full(len(targets), None, dtype=object)
    index, _, b = layers.near_targets(targets)
    labels[index] = np.where(b < 0, "inside", np.where(b > 0, "outside", "average"))
    return labels


def curves(problem):
    points = problem.geometry(2 * np.pi * np.arange(N) / N)[0]
    return {"function": Curve.from_function(problem.geometry), "points": Curve.from_points(points)}


class LaplaceApi(unittest.TestCase):
    def test_kite(self):
        problem = LAPLACE_KITE
        for name, curve in curves(problem).items():
            layers = LayerPotentials(curve, N)
            f, g = problem.boundary_densities(layers.parameters)
            y = targets_for(curve, problem.box, layers.delta)
            value = layers.single_layer(f, y) + layers.double_layer(g, y)
            exact = np.array([problem.exact(p[None], side)[0] for p, side in zip(y, sides(layers, y))])
            with self.subTest(curve=name):
                self.assertLess(np.max(np.abs(value - exact)), TOLERANCE)


class StokesApi(unittest.TestCase):
    def test_starfish(self):
        problem = STOKES_STARFISH
        for name, curve in curves(problem).items():
            layers = LayerPotentials(curve, N, mu=problem.mu)
            q, f = problem.boundary_data(layers.parameters, problem.mu)
            y = targets_for(curve, problem.box, layers.delta)
            u = layers.stokeslet(f, y) + layers.stresslet(q, y)
            p = layers.stokes_pressure(f, q, y)
            exact = [problem.exact(point[None], side, problem.mu) for point, side in zip(y, sides(layers, y))]
            u_exact = np.array([e[0][0] for e in exact])
            p_exact = np.array([e[1][0] for e in exact])
            with self.subTest(curve=name):
                self.assertLess(np.max(np.linalg.norm(u - u_exact, axis=-1)), TOLERANCE)
                self.assertLess(np.max(np.abs(p - p_exact)), TOLERANCE)


class Conventions(unittest.TestCase):
    def test_clockwise_points_rejected(self):
        t = 2 * np.pi * np.arange(64) / 64
        with self.assertRaises(ValueError):
            Curve.from_points(np.stack((np.cos(t), -np.sin(t)), axis=-1))

    def test_density_shape_checked(self):
        t = 2 * np.pi * np.arange(64) / 64
        layers = LayerPotentials(Curve.from_points(np.stack((np.cos(t), np.sin(t)), axis=-1)), 64)
        with self.assertRaises(ValueError):
            layers.single_layer(np.ones(63), np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
