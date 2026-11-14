import unittest
import numpy as np
from scipy.integrate import quad
from regbie.regularized_slp import starfish
from regbie.starfish_reference import starfish_reference


class StarfishTests(unittest.TestCase):
    def test_derivative(self):
        t = np.linspace(0, 2*np.pi, 31, endpoint=False)
        step = 1e-5
        derivative = (starfish(t+step)[0]-starfish(t-step)[0])/(2*step)
        np.testing.assert_allclose(derivative, starfish(t)[1], atol=4e-10)

    def test_reference_against_adaptive_singular_integral(self):
        # Independent original logarithmic integral, split at its singularity.
        for theta in [.17, 1.23, 3.4]:
            y = starfish(np.array([theta]))[0][0]
            def integrand(t):
                x = starfish(np.array([t]))[0][0]
                return np.log(np.linalg.norm(x-y))*(1+.7*np.cos(3*t))/(2*np.pi)
            value = sum(quad(integrand, lo, hi, epsabs=2e-12, epsrel=2e-12,
                             limit=200)[0] for lo,hi in [(0, theta), (theta, 2*np.pi)])
            self.assertAlmostEqual(value, starfish_reference(np.array([theta]))[0], places=11)

    def test_reference_resolution(self):
        theta = np.linspace(0, 2*np.pi, 97, endpoint=False)
        np.testing.assert_allclose(starfish_reference(theta, 4096),
                                   starfish_reference(theta, 8192), atol=2e-14, rtol=0)
