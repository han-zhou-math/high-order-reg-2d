"""Independent high-precision checks of special numerical branches."""
import unittest
import numpy as np
import mpmath as mp
from regbie.regularized_slp import coefficients, regularized_kernel
from regbie.surface_operators import radial_factors
from regbie.verify_stokes_derivation import scaled_moment_n0, moment_n0


class StabilityTests(unittest.TestCase):
    def test_scalar_and_radial_switches(self):
        with mp.workdps(90):
            points = [0., 1e-30, 1e-16, 1e-8, .02, .5-1e-12, .5, .5+1e-12, 2., 20.]
            for order in (3, 5, 7):
                for lam in (0., .5, 2.):
                    cs = [mp.mpf(float(c)) for c in coefficients(order, lam)]
                    for v in points:
                        u = mp.mpf(v)
                        delta = .13
                        poly = lambda z: 1+sum((-1)**k*c*z**k for k,c in enumerate(cs,1))
                        if v == 0:
                            base = 2*mp.log(delta)-mp.euler
                            aref = 1+cs[0]
                            apref = -mp.mpf('.5')-cs[0]-(cs[1] if len(cs)>1 else 0)
                            tpref = aref
                        else:
                            base = 2*mp.log(delta)+mp.log(u)+mp.e1(u)
                            t = lambda z: 1-mp.exp(-z)*poly(z)
                            aref = t(u)/u
                            apref = mp.diff(lambda z: t(z)/z, u)
                            tpref = mp.diff(t, u)
                        corr = -cs[0]
                        if len(cs)>1: corr += cs[1]*(1+u)
                        if len(cs)>2: corr -= cs[2]*(2+2*u+u*u)
                        ref = float((base+mp.exp(-u)*corr)/(4*mp.pi))
                        val = float(regularized_kernel(delta*np.sqrt(v),delta,order,lam*delta))
                        self.assertLess(abs(val-ref), 3e-15)
                        actual = radial_factors(v,order,lam)
                        np.testing.assert_allclose(actual, [float(aref),float(apref),float(tpref)],
                                                   rtol=3e-13,atol=5e-15)

    def test_large_distance_coefficients(self):
        with mp.workdps(70):
            for lam in (1.0001, 2., 4., 5., 20., 1000.):
                z = mp.mpf(lam)**2
                d = [mp.quad(lambda s: mp.exp(-s)*s**j*(1+s/z)**(-j-mp.mpf('1.5')),
                             [0,1,mp.inf])/z**(j+1) for j in range(3)]
                c3=d[2]/2
                c2=d[1]+(2*z+5)*c3
                c1=d[0]+(z+mp.mpf('1.5'))*c2-(z*z+3*z+mp.mpf('3.75'))*c3
                np.testing.assert_allclose(coefficients(7,lam),list(map(float,(c1,c2,c3))),rtol=8e-14)

    def test_scaled_near_surface_moment(self):
        for p in (3,5,7):
            for lam in (0.,1e-300,1e-200,1e-20):
                self.assertAlmostEqual(scaled_moment_n0(p,lam),np.pi,places=14)
            for lam in (.1,.5,2.):
                self.assertAlmostEqual(scaled_moment_n0(p,lam),lam*moment_n0(p,lam),places=14)


if __name__ == '__main__':
    unittest.main()
