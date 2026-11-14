"""Physical spacing and exact-direct reconstruction regression checks."""
import unittest
import numpy as np
from experiments.matched_allgrid import spacing, run
from experiments.problems import LAPLACE_STARFISH, STOKES_KITE


class MatchedAllgridTests(unittest.TestCase):
    def test_actual_chords_match_cartesian_spacing(self):
        for problem in (LAPLACE_STARFISH, STOKES_KITE):
            for h in (.05, .01, .003):
                for ratio in (1., .5):
                    result = spacing(problem.geometry, h, ratio=ratio)
                    n = result['n_gamma']
                    x = problem.geometry(2*np.pi*np.arange(n)/n)[0]
                    actual = np.linalg.norm(np.roll(x, -1, axis=0)-x, axis=1).max()
                    self.assertAlmostEqual(actual, result['chord_max'])
                    self.assertTrue(.8 <= actual/(h*ratio) <= 1.2)
                    self.assertEqual(n % 2, 0)

    def test_exact_direct_laplace_refinement(self):
        coarse, _ = run(LAPLACE_STARFISH, 32, mode='exact_direct')
        fine, _ = run(LAPLACE_STARFISH, 64, mode='exact_direct')
        self.assertEqual(fine['laplace_source']['inf'], 0.)
        self.assertEqual(fine['direct_laplace']['inf'], 0.)
        self.assertLess(fine['laplace']['inf'], coarse['laplace']['inf']/8)

    def test_boundary_refinement_holds_smoothing_width_fixed(self):
        base, _ = run(LAPLACE_STARFISH, 32, mode='exact_direct')
        delta = (2*np.pi/base['n_gamma'])**(5/7)
        refined, _ = run(LAPLACE_STARFISH, 32, mode='exact_direct',
                         delta_override=delta, n_override=2*base['n_gamma'])
        self.assertEqual(refined['n_gamma'], 2*base['n_gamma'])
        self.assertEqual(refined['delta'], delta)
        self.assertEqual(refined['h_c'], base['h_c'])
