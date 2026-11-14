"""Check the physical viscosity scaling against an exact circle layer."""
import unittest

import numpy as np

from experiments.bie_evaluators import eval_stokes_regularized
from experiments.geometries import circle_geometry


class StokesViscosityTests(unittest.TestCase):
    def test_constant_density_circle_single_layer(self):
        # On the unit circle, the constant-density Stokes single layer is
        # f/(4*mu): the mean logarithmic term vanishes and the tensor mean
        # is I/2. This reference is independent of the regularized kernel.
        n = 1024
        theta = np.array([0., .43, 1.2, 2.51])
        targets = circle_geometry(theta)[0]
        f = np.broadcast_to([1.1, -.7], (n, 2)).copy()
        q = np.zeros_like(f)
        for mu in (.4, 1., 2.5):
            actual = eval_stokes_regularized(circle_geometry, q, f, targets,
                                             theta, np.zeros(len(theta)),
                                             .04, 7, mu=mu)
            np.testing.assert_allclose(actual, f[:len(theta)] / (4 * mu),
                                       rtol=0., atol=3e-12)

    def test_single_layer_inverse_viscosity_near_curve(self):
        n = 512
        t = 2 * np.pi * np.arange(n) / n
        theta = np.array([.17, 1.3, 2.7])
        x, _, _, normal, _, _ = circle_geometry(theta)
        b = np.array([-.04, 0., .04])
        targets = x + b[:, None] * normal
        f = np.column_stack((.3 + np.cos(2 * t), -.2 + np.sin(3 * t)))
        q = np.zeros_like(f)
        one = eval_stokes_regularized(circle_geometry, q, f, targets, theta,
                                      b, .08, 7, mu=1.)
        scaled = eval_stokes_regularized(circle_geometry, q, f, targets, theta,
                                         b, .08, 7, mu=2.5)
        np.testing.assert_allclose(scaled, one / 2.5, rtol=3e-14, atol=3e-15)


if __name__ == "__main__":
    unittest.main()
