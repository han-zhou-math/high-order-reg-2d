import unittest
import numpy as np
from scipy.integrate import quad
from regbie.regularized_slp import circle,starfish
from regbie.regularized_dlp import normals
from regbie.surface_operators import radial_factors,evaluate_surface
from verification.surface_operator_convergence import starfish_reference,benchmark
from regbie.manufactured_convergence import densities


class SurfaceOperatorTests(unittest.TestCase):
    def test_mixed_normal_derivative(self):
        r=np.array([.07,-.04])
        nx=np.array([.6,.8])
        ny=np.array([-.8,.6])
        delta=.1
        step=2e-6
        for p in [3,5,7]:
            def source_derivative(epsilon):
                rr=r-epsilon*ny
                a,_,_=radial_factors(np.dot(rr,rr)/delta**2,p)
                return np.dot(nx,rr)*a/(2*np.pi*delta**2)
            fd=(-source_derivative(2*step)+8*source_derivative(step)
                -8*source_derivative(-step)+source_derivative(-2*step))/(12*step)
            a,ap,_=radial_factors(np.dot(r,r)/delta**2,p)
            exact=-(np.dot(nx,ny)*a/delta**2+2*np.dot(nx,r)*np.dot(ny,r)*ap/delta**4)/(2*np.pi)
            self.assertAlmostEqual(float(fd),float(exact),delta=2e-8)
            a0,ap0,tp0=radial_factors(0.,p)
            self.assertTrue(np.all(np.isfinite([a0,ap0,tp0])))

    def test_constant_hypersingular_and_raw_divergence(self):
        theta=np.array([0.,.2,.7])
        ones=lambda t: np.ones_like(t)
        zeros=lambda t: np.zeros_like(t)
        delta=.03
        for p,c in [(3,2/np.sqrt(np.pi)),(5,8/(3*np.sqrt(np.pi))),(7,16/(5*np.sqrt(np.pi)))]:
            v=evaluate_surface(circle,ones,ones,zeros,theta,delta,2048,p)
            np.testing.assert_allclose(v['Wmaue'],0.,atol=0)
            np.testing.assert_allclose(v['Wcorrected'],0.,atol=2e-10)
            np.testing.assert_allclose(delta*v['Wraw'],-c,atol=4e-4)

    def test_starfish_reference_against_adaptive_quadrature(self):
        f,g=densities(starfish)
        for theta in [.17,1.2]:
            y,dy=starfish(np.array([theta]))
            ny=normals(dy)[0]
            def integrand(t,adjoint):
                x,dx=starfish(np.array([t]))
                r=x[0]-y[0]
                numerator=-np.dot(ny,r) if adjoint else np.dot(normals(dx)[0],r)
                density=f(np.array([t]))[0] if adjoint else g(np.array([t]))[0]
                return numerator/np.dot(r,r)*density*np.linalg.norm(dx[0])/(2*np.pi)
            ref=starfish_reference(np.array([theta]),4096)
            for adjoint,key in [(False,'K'),(True,'Kprime')]:
                value=sum(quad(lambda t:integrand(t,adjoint),lo,hi,epsabs=2e-12,
                               epsrel=2e-12,limit=150)[0]
                          for lo,hi in [(theta-np.pi,theta),(theta,theta+np.pi)])
                self.assertAlmostEqual(value,float(ref[key][0]),delta=3e-11)

    def test_maue_equivalence_and_starfish_convergence(self):
        theta=2*np.pi*np.arange(37)/37
        curve,f,g,g_dt,exact,_=benchmark('starfish',theta)
        for p in [3,5,7]:
            errors=[]
            for delta in [.03,.015]:
                v=evaluate_surface(curve,f,g,g_dt,theta,delta,2048,p)
                np.testing.assert_allclose(v['Wcorrected'],v['Wmaue'],atol=8e-10,rtol=0)
                errors.append(np.max(np.abs(v['Wmaue']-exact['W'])))
            rate=np.log2(errors[0]/errors[1])
            self.assertGreater(rate,p-.8,(p,rate))
            self.assertLess(rate,p+.8,(p,rate))


if __name__=='__main__':
    unittest.main()
