import unittest
import numpy as np

from experiments import manufactured_laplace as laplace
from experiments import manufactured_stokes as stokes
from experiments.boundary_data import pressure_densities
from experiments.geometries import stokes_kite_geometry
from experiments.validation import (finite_difference_exactness,
                                    validate_boundary_finiteness,
                                    validate_kite_poles,
                                    validate_starfish_poles)


class RedesignedExperimentTests(unittest.TestCase):
    def test_circle_closed_forms_and_signs(self):
        n = 8192
        t = 2 * np.pi * np.arange(n) / n
        x = np.stack((np.cos(t), np.sin(t)), axis=-1)
        rho = laplace.circle_density(t)
        for radius, theta in ((0.6, 0.23), (1.4, -0.31)):
            target = radius * np.array((np.cos(theta), np.sin(theta)))
            difference = x - target
            r2 = np.sum(difference**2, axis=-1)
            single = np.mean(np.log(np.sqrt(r2)) * rho)
            double = np.mean(np.sum(difference * x, axis=-1) / r2 * rho)
            self.assertAlmostEqual(single,
                float(laplace.exact_circle_single(radius, theta)), places=12)
            self.assertAlmostEqual(double,
                float(laplace.exact_circle_double(radius, theta)), places=12)

    def test_poles_and_boundary_data(self):
        self.assertTrue(all(value > 0 for value in validate_starfish_poles().values()))
        self.assertTrue(all(value > 0 for value in validate_kite_poles().values()))
        validate_boundary_finiteness(256)

    def test_complex_derivatives(self):
        z = np.array([0.12 + 0.08j, -0.31 + 0.27j])
        eps = 1e-6
        self.assertLess(np.max(np.abs((laplace.Fminus(z + eps)
                                      - laplace.Fminus(z - eps)) / (2 * eps)
                                     - laplace.dFminus(z))), 2e-9)
        self.assertLess(np.max(np.abs((stokes.dF(z + eps) - stokes.dF(z - eps))
                                     / (2 * eps) - stokes.ddF(z))), 2e-9)
        self.assertLess(np.max(np.abs((stokes.dG(z + eps) - stokes.dG(z - eps))
                                     / (2 * eps) - stokes.ddG(z))), 2e-9)

    def test_goursat_kinematics_and_pressure_jumps(self):
        point = np.array([[0.13, -0.17]])
        eps = 2e-6
        ex, ey = np.array([eps, 0.0]), np.array([0.0, eps])
        grad_fd = np.stack(((stokes.velocity(point + ex)[0]
                             - stokes.velocity(point - ex)[0]) / (2 * eps),
                            (stokes.velocity(point + ey)[0]
                             - stokes.velocity(point - ey)[0]) / (2 * eps)), axis=-1)
        self.assertLess(np.max(np.abs(grad_fd - stokes.grad_velocity(point)[0])), 2e-8)
        t = 2 * np.pi * np.arange(512) / 512
        x, _, _, normal, _, _ = stokes_kite_geometry(t)
        q, f = stokes.stokes_boundary_data(t)
        pdata = pressure_densities(stokes_kite_geometry, q, f)
        self.assertLess(np.max(np.abs(pdata.psi + stokes.pressure(x))), 2e-8)
        exact_normal_jump = -np.sum(stokes.grad_pressure(x) * normal, axis=-1)
        self.assertLess(np.max(np.abs(pdata.phi - exact_normal_jump)), 2e-6)

    def test_lightweight_pde_residuals(self):
        residuals = finite_difference_exactness(samples=5)
        self.assertLess(residuals["laplace_residual"], 2e-5)
        self.assertLess(residuals["divergence_residual"], 2e-8)
        self.assertLess(residuals["momentum_residual"], 2e-4)


if __name__ == "__main__":
    unittest.main()
