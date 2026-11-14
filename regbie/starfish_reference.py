"""Unregularized on-curve reference by analytic logarithmic subtraction.

For f(t)|x'(t)|=1+0.7*cos(3t), the circular log contribution is exact.
Only the smooth geometric remainder is integrated numerically.
"""
import numpy as np


def starfish_reference(theta, n=4096):
    theta = np.asarray(theta)
    t = 2*np.pi*np.arange(n)/n
    half_difference = (t[None, :]-theta[:, None])/2
    # (R(t)-R(theta))/(2 sin((t-theta)/2)), with its removable
    # singularity evaluated analytically using sin(3v)/sin(v).
    radial_quotient = (-0.2*np.sin(1.5*(t[None, :]+theta[:, None])) *
                       (3-4*np.sin(half_difference)**2))
    rt = 1+0.2*np.cos(3*t)
    ry = 1+0.2*np.cos(3*theta)
    ratio_squared = radial_quotient**2+ry[:, None]*rt[None, :]
    smooth = .5*np.log(ratio_squared)
    g = 1+0.7*np.cos(3*t)
    return -0.7*np.cos(3*theta)/6+smooth @ g/n
