import unittest

import numpy as np

from regbie import all_grid_reconstruction as all_grid


class TestPressureAndAllGrid(unittest.TestCase):
    def test_pressure_jump_densities(self):
        pressure = all_grid.pressure_densities(512)
        t = 2 * np.pi * np.arange(512) / 512
        x, _, _, normal, _, _ = all_grid.geometry(t)
        self.assertLess(np.max(np.abs(pressure.psi.samples + 2 * x[:, 0])), 1e-10)
        self.assertLess(np.max(np.abs(pressure.phi.samples + 2 * normal[:, 0])), 1e-9)
        self.assertLess(pressure.jump_residual, 1e-10)
        self.assertLess(pressure.normal_jump_residual, 1e-9)

    def test_cartesian_operators_and_dst(self):
        result = all_grid.validate_cartesian_operators()
        self.assertLess(result["dst_residual"], 1e-12)
        self.assertLess(result["harmonic_errors"][2], result["harmonic_errors"][0])
        self.assertGreater(min(result["harmonic_rates"][:2]), 5.0)
        self.assertGreater(min(np.log2(result["stokes_errors"][i] /
                                       result["stokes_errors"][i + 1])
                               for i in range(3)), 3.5)

    def test_stencil_union_has_gradient_only_nodes(self):
        xx, yy, h = all_grid.grid_coordinates(64)
        laplace = all_grid.classify_crossing(xx, yy, h, all_grid.LAPLACE_OFFSETS)
        gradient = all_grid.classify_crossing(xx, yy, h, all_grid.GRADIENT_OFFSETS)
        self.assertGreater(np.count_nonzero(gradient & ~laplace), 0)
        self.assertGreater(np.min(np.abs(all_grid.level_value(
            np.stack((xx, yy), axis=-1)))), 100 * np.finfo(float).eps)
        centers = np.stack((xx[1:-1, 1:-1], yy[1:-1, 1:-1]), axis=-1)
        for crossing, offsets in ((laplace, all_grid.LAPLACE_OFFSETS),
                                  (gradient, all_grid.GRADIENT_OFFSETS)):
            center_inside = all_grid.level_value(centers) < 0
            safe = ~crossing
            for di, dj in offsets:
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    sample = centers + fraction * np.array([di * h, dj * h])
                    sample_inside = all_grid.level_value(sample) < 0
                    self.assertFalse(np.any(safe & (sample_inside != center_inside)))


if __name__ == "__main__":
    unittest.main()
