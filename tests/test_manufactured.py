import unittest
import numpy as np
from regbie.regularized_slp import circle, starfish
from regbie.regularized_dlp import double_layer, smoothing_over_r2
from regbie.manufactured_convergence import field, densities, target_family, reconstruct
from verification.verify_double_layer import circle_dlp


class ManufacturedTests(unittest.TestCase):
    def test_harmonic_field_gradients(self):
        x=np.array([[.7,.2],[1.1,-.3],[-.4,.8]])
        for side in ['inside','outside']:
            _,grad=field(x,side)
            for j in [0,1]:
                step=np.zeros_like(x)
                step[:,j]=1e-6
                fd=(field(x+step,side)[0]-field(x-step,side)[0])/(2e-6)
                np.testing.assert_allclose(fd,grad[:,j],atol=4e-10,rtol=2e-9)

    def test_general_double_layer_against_circle_formula(self):
        theta=2*np.pi*np.arange(23)/23
        g=lambda t: 1+.7*np.cos(3*t)
        for p in [3,5,7]:
            for offset in [-.5,0.,.5]:
                delta=.07
                y=(1+offset*delta)*circle(theta)[0]
                value=double_layer(circle,g,y,delta,512,p,closest_t=theta,b=offset*delta)
                check,_=circle_dlp(theta,delta,512,p,offset)
                np.testing.assert_allclose(value,check,atol=3e-15,rtol=0)

    def test_constant_double_layer_and_diagonal(self):
        theta=np.array([0.,.2,1.7])
        b=np.array([-.01,0.,.02])
        x,dx=starfish(theta)
        from regbie.regularized_dlp import normals
        y=x+b[:,None]*normals(dx)
        for p in [3,5,7]:
            value=double_layer(starfish,lambda t: np.full_like(t,2.),y,.05,128,p,
                               closest_t=theta,b=b,chunk_size=1)
            np.testing.assert_allclose(value,[2,1,0],atol=0)
            self.assertTrue(np.isfinite(smoothing_over_r2(0.,.05,p)))
        with self.assertRaises(ValueError):
            smoothing_over_r2(.01,.05,7,b=.01,surface=True)

    def test_green_representation_convergence_starfish(self):
        theta=2*np.pi*np.arange(37)/37
        for p in [3,5,7]:
            for offset in [-.5,0.,.5]:
                errors=[]
                for delta in [.04,.02]:
                    b=offset*delta
                    y,exact=target_family(starfish,theta,b)
                    s,d=reconstruct(starfish,theta,y,b,delta,2048,p)
                    errors.append(np.max(np.abs(s+d-exact)))
                rate=np.log2(errors[0]/errors[1])
                self.assertGreater(rate,p-.8,(p,offset,rate,errors))
                self.assertLess(rate,p+.8,(p,offset,rate,errors))

    def test_exact_circle_layers_for_jump_densities(self):
        # Individual exact layers ensure the manufactured sum is not hiding
        # equal and opposite layer errors or a jump-sign mistake.
        theta=2*np.pi*np.arange(29)/29
        radius=.9
        y=radius*circle(theta)[0]
        s,d=reconstruct(circle,theta,y,radius-1,.025,1024,7)
        exact_s=(.35*radius*np.cos(theta)+.2*radius**2*np.cos(2*theta)+
                 .5*radius**3*np.cos(3*theta))
        exact_d=1-.05*radius*np.cos(theta)+.5*radius**3*np.cos(3*theta)
        np.testing.assert_allclose(s,exact_s,atol=2e-12,rtol=0)
        np.testing.assert_allclose(d,exact_d,atol=2e-12,rtol=0)


if __name__=='__main__':
    unittest.main()
