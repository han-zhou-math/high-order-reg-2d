import unittest
import numpy as np
from regbie.regularized_slp import regularized_kernel, circle
from regbie.regularized_dlp import smoothing_over_r2, double_layer
from verification.differentiated_kernel_convergence import leading_bias


class DifferentiatedKernelTests(unittest.TestCase):
    def test_source_normal_derivative(self):
        # Freeze target-dependent b while varying the source in its normal direction.
        x=np.array([[.8,.1],[.2,.7],[-.4,.3]])
        y=np.array([[.76,.12],[.1,.65],[-.2,.4]])
        normal=np.array([[1.,0.],[0.,1.],[.6,.8]])
        r=x-y
        b=np.array([.01,-.02,.03])
        delta=.08
        step=2e-6
        for p in [3,5,7]:
            def evaluate(eps):
                return regularized_kernel(np.linalg.norm(r+eps*normal,axis=1),delta,p,b)
            finite_difference=(-evaluate(2*step)+8*evaluate(step)-8*evaluate(-step)+evaluate(-2*step))/(12*step)
            analytic=np.sum(normal*r,axis=1)*smoothing_over_r2(np.sum(r*r,axis=1),delta,p,b)/(2*np.pi)
            np.testing.assert_allclose(finite_difference,analytic,atol=2e-10,rtol=2e-10)

    def test_raw_constant_density_near_curve_bias(self):
        t0=np.array([0.,.7])
        for p in [3,5,7]:
            for sign in [-1,1]:
                delta=.015
                b=sign*.5*delta
                y=(1+b)*circle(t0)[0]
                raw=double_layer(circle,lambda t: np.ones_like(t),y,delta,2048,p,
                                 closest_t=t0,b=b,subtract=False)
                chi=1. if sign<0 else 0.
                expected=sign*leading_bias(p,.5)
                np.testing.assert_allclose(raw-chi,expected,atol=.002,rtol=0)
                self.assertGreater(np.max(np.abs(raw-chi)),.01)

    def test_raw_surface_has_no_jump_correction(self):
        t=np.array([0.,.4])
        y=circle(t)[0]
        raw=double_layer(circle,lambda t: np.ones_like(t),y,.05,1024,7,
                         closest_t=t,b=0.,subtract=False)
        np.testing.assert_allclose(raw,.5,atol=1e-10,rtol=0)


if __name__=='__main__':
    unittest.main()
