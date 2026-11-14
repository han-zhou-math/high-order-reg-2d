import unittest

import numpy as np

import experiments.swapped_low_order as problems


class SwappedLowOrderTests(unittest.TestCase):
    def test_scaled_curves_and_pole_locations(self):
        t = 2 * np.pi * np.arange(16384) / 16384
        for geometry in (problems.laplace_kite_geometry,
                         problems.stokes_starfish_geometry):
            self.assertLessEqual(np.max(np.abs(geometry(t)[0])), 0.895000000001)
        outside_poles = problems._LM_POLES + problems._SF_POLES + problems._SG_POLES
        self.assertFalse(np.any(problems.laplace_inside(np.array(
            [[z.real, z.imag] for z in problems._LM_POLES]))))
        self.assertTrue(np.all(problems.laplace_inside(np.array(
            [[z.real, z.imag] for z in problems._LP_POLES]))))
        self.assertFalse(np.any(problems.stokes_inside(np.array(
            [[z.real, z.imag] for z in outside_poles[2:]]))))

    def test_stokes_field_is_incompressible(self):
        points = np.array([[0.0, 0.0], [0.2, 0.1], [-0.15, 0.25]])
        step = 2e-5
        dx = (problems.stokes_velocity(points + [step, 0])
              - problems.stokes_velocity(points - [step, 0])) / (2 * step)
        dy = (problems.stokes_velocity(points + [0, step])
              - problems.stokes_velocity(points - [0, step])) / (2 * step)
        self.assertLess(np.max(np.abs(dx[:, 0] + dy[:, 1])), 2e-10)


if __name__ == "__main__":
    unittest.main()
