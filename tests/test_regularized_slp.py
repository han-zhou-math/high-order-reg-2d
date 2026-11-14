"""Independent moment, singular-limit, and analytic-potential checks."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1
from regbie.regularized_slp import coefficients, regularized_kernel, single_layer, circle
from verification.convergence import benchmark


class RegularizationTests(unittest.TestCase):
    def test_surface_coefficients(self):
        for p, expected in [(3, [2]), (5, [4, 4/3]), (7, [6, 4, 8/15])]:
            np.testing.assert_allclose(coefficients(p, 0.), expected, rtol=2e-15)

    def test_cancelled_moments(self):
        # Integrate the correction itself, independently of the kernel evaluator.
        for p in [3, 5, 7]:
            for lam in [0., .5, 1., 3., 5.]:
                cs = coefficients(p, lam)
                def correction(t):
                    u = t*t+lam*lam
                    poly = -cs[0]
                    if p >= 5:
                        poly += cs[1]*(1+u)
                    if p >= 7:
                        poly -= cs[2]*(2+2*u+u*u)
                    return (exp1(u)+np.exp(-u)*poly)/(4*np.pi)
                for j in range((p-1)//2):
                    value = 2*quad(lambda t: t**(2*j)*correction(t), 0, np.inf,
                                   epsabs=2e-12, epsrel=2e-12)[0]
                    self.assertLess(abs(value), 3e-11, (p, lam, j, value))

    def test_coincident_limit_and_small_arguments(self):
        delta = .13
        for p in [1, 3, 5, 7]:
            cs = coefficients(p, 0.)
            import math
            poly = sum((-1)**k*math.factorial(k-1)*float(c) for k,c in enumerate(cs, 1))
            expected = (2*np.log(delta)-np.euler_gamma+poly)/(4*np.pi)
            self.assertAlmostEqual(float(regularized_kernel(0., delta, p)), expected, places=14)
            self.assertAlmostEqual(float(regularized_kernel(1e-12, delta, p)), expected, places=14)
            r = np.array([.05, .1, .2, 1.])
            u = (r/delta)**2
            direct = np.log(r)/(2*np.pi)+exp1(u)/(4*np.pi)
            poly_array = np.zeros_like(r)
            if p >= 3: poly_array -= cs[0]
            if p >= 5: poly_array += cs[1]*(1+u)
            if p >= 7: poly_array -= cs[2]*(2+2*u+u*u)
            direct += np.exp(-u)*poly_array/(4*np.pi)
            np.testing.assert_allclose(regularized_kernel(r, delta, p), direct, atol=3e-15)

    def test_analytic_convergence(self):
        theta = np.linspace(0, 2*np.pi, 37, endpoint=False)
        for case in ['circle_on', 'circle_inside', 'circle_outside', 'ellipse_on']:
            for p in [1, 3, 5, 7]:
                errors = []
                for delta in [.1, .05]:
                    curve, density, y, b, exact = benchmark(case, theta, delta)
                    approx = single_layer(curve, density, y, delta, 2048, p, b)
                    errors.append(np.max(np.abs(approx-exact)))
                rate = np.log2(errors[0]/errors[1])
                self.assertGreater(rate, p-.5, (case, p, rate))
                self.assertLess(rate, p+.6, (case, p, rate))

    def test_target_dependent_offsets_and_chunks(self):
        theta = np.array([0., .4, .9])
        offsets = np.array([0., -.01, .025])
        y = circle(theta)[0]*(1+offsets[:, None])
        density = lambda t: np.cos(3*t)
        full = single_layer(circle, density, y, .05, 512, 7, offsets)
        split = single_layer(circle, density, y, .05, 512, 7, offsets, chunk_size=1)
        np.testing.assert_allclose(full, split, atol=2e-15)


if __name__ == '__main__':
    unittest.main()
