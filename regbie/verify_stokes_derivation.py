"""Checks of the regularized Stokes kernels; not a general Stokes solver.

Run: OPENBLAS_NUM_THREADS=1 python3 -m regbie.verify_stokes_derivation
Uses the existing scalar coefficients and an independent Stokes representation.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc, gamma
from .regularized_slp import coefficients, regularized_kernel
from .surface_operators import radial_factors


def geometry(t):
    t = np.asarray(t)
    z = [sum(c*(1j*k)**j*np.exp(1j*k*t)
             for k, c in [(1, 1.), (4, .1), (-2, .1)]) for j in range(4)]
    x, d, dd, ddd = [np.stack((v.real, v.imag), axis=-1) for v in z]
    speed = np.linalg.norm(d, axis=-1)
    tau = d/speed[..., None]
    normal = np.stack((tau[..., 1], -tau[..., 0]), axis=-1)
    cross = lambda a, b: a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    kappa = cross(d, dd)/speed**3
    ks = cross(d, ddd)/speed**4-3*cross(d, dd)*np.sum(d*dd, axis=-1)/speed**6
    return x, speed, tau, normal, kappa, ks


def field(x):
    a, b = x[..., 0], x[..., 1]
    return np.stack((a*a, -2*a*b), axis=-1)


def traction(x, normal):
    a, b = x[..., 0], x[..., 1]
    return np.stack((2*a*normal[..., 0]-2*b*normal[..., 1],
                     -2*b*normal[..., 0]-6*a*normal[..., 1]), axis=-1)


def q_derivatives(x, tau, normal, kappa):
    a, b = x[..., 0], x[..., 1]
    def grad_apply(v):
        return np.stack((2*a*v[..., 0], -2*b*v[..., 0]-2*a*v[..., 1]), axis=-1)
    qs = grad_apply(tau)
    qss = np.stack((2*tau[..., 0]**2, -4*tau[..., 0]*tau[..., 1]), axis=-1)
    qss -= kappa[..., None]*grad_apply(normal)
    return qs, qss


def scaled_moment_n0(order, lam):
    """Return lam*N0 with its finite endpoint pi; never form 1/lam."""
    if not np.isfinite(lam) or lam < 0:
        raise ValueError('lam must be finite and nonnegative')
    cs = coefficients(order, lam)
    ans = np.pi*erfc(lam)
    from math import comb
    for k, c in enumerate(cs, 1):
        moment = sum(comb(k-1, j)*lam**(2*(k-1-j))*gamma(j+.5) for j in range(k))
        ans += lam*(-1)**k*c*np.exp(-lam*lam)*moment
    return ans


def moment_n0(order, lam):
    """Unscaled moment for moderate-lambda diagnostic integrals only."""
    if lam <= 0:
        raise ValueError('N0 itself is singular at zero; use scaled_moment_n0.')
    return scaled_moment_n0(order, lam)/lam


def evaluate(n, delta, order, b, theta, *, return_parts=False):
    x, v, _, normal, _, _ = geometry(2*np.pi*np.arange(n)/n)
    f, q = traction(x, normal), field(x)
    x0, _, tau0, n0, kap, ks = geometry(theta)
    q0, f0 = field(x0), traction(x0, n0)
    qs, qss = q_derivatives(x0, tau0, n0, kap)
    y = x0+b*n0
    r = x[None, :, :]-y[:, None, :]
    r2 = np.sum(r*r, axis=-1)
    u = r2/delta**2
    # The derivative series evaluates V/u² without subtracting A-T'.
    a, ap, _ = radial_factors(u, order, abs(b)/delta)
    vp = -ap
    gf = regularized_kernel(np.sqrt(r2), delta, order, b)
    rf = np.sum(r*f[None, :, :], axis=-1)
    sl_integrand = r*rf[..., None]*a[..., None]/(4*np.pi*delta**2)-gf[..., None]*f[None, :, :]/2
    rq = np.sum(r*(q[None, :, :]-q0[:, None, :]), axis=-1)
    rn = np.sum(r*normal[None, :, :], axis=-1)
    dl_integrand = r*(rq*rn*vp/(np.pi*delta**4))[..., None]
    weights = 2*np.pi*v/n
    sl = np.einsum('mni,n->mi', sl_integrand, weights)
    chi = 1. if b < 0 else (0. if b > 0 else .5)
    dl = np.einsum('mni,n->mi', dl_integrand, weights)+chi*q0
    raw = sl+dl
    if b != 0:
        aa = np.sqrt(1+b*kap)
        nnqs = n0*np.sum(n0*qs, axis=-1)[:, None]
        nnqss = n0*np.sum(n0*qss, axis=-1)[:, None]
        symqs = n0*np.sum(tau0*qs, axis=-1)[:, None]+tau0*np.sum(n0*qs, axis=-1)[:, None]
        # Factor out b² analytically: (b²/delta)*N0=abs(b)*(lam*N0).
        b0_scaled = n0*np.sum(n0*f0, axis=-1)[:, None]/aa[:, None]
        c2_scaled = symqs/aa[:, None]**3-b*nnqss/(2*aa[:, None]**3)
        c2_scaled += (b*b*ks/(2*aa**5))[:, None]*nnqs
        scale = abs(b)*scaled_moment_n0(order, abs(b)/delta)
        sl += scale*b0_scaled/(4*np.pi)
        dl += scale*c2_scaled/(2*np.pi)
    exact = chi*field(y) if b != 0 else .5*field(y)
    if return_parts:
        return sl, dl
    return sl+dl, raw, exact


def check_moments():
    largest = 0.
    for order in (3, 5, 7):
        for lam in (.1, .5, 1., 2.):
            cs = coefficients(order, lam)
            def q_and_r(t):
                u = t*t+lam*lam
                p = 1+sum((-1)**k*c*u**k for k, c in enumerate(cs, 1))
                pp = sum(k*(-1)**k*c*u**(k-1) for k, c in enumerate(cs, 1))
                return np.exp(-u)*p, np.exp(-u)*((1+u)*p-u*pp)
            n0 = 2*quad(lambda t: q_and_r(t)[0]/(t*t+lam*lam), 0, np.inf, epsabs=1e-11)[0]
            l1 = 2*quad(lambda t: t*t*q_and_r(t)[1]/(t*t+lam*lam)**2, 0, np.inf, epsabs=1e-11)[0]
            largest = max(largest, abs(n0-moment_n0(order, lam)), abs(l1-n0/2))
            for k in range(2, (order-1)//2+2):
                lk = 2*quad(lambda t: t**(2*k)*q_and_r(t)[1]/(t*t+lam*lam)**2,
                            0, np.inf, epsabs=1e-10)[0]
                largest = max(largest, abs(lk))
    assert largest < 1e-8, largest
    print('Maximum moment residual:', largest)


def main():
    check_moments()
    theta = 2*np.pi*(np.arange(37)+.31)/37
    rows = []
    for order in (3, 5, 7):
        for lam in (-.5, 0., .5):
            previous = None
            for n in (256, 512, 1024, 2048):
                delta = (2*np.pi/n)**.8
                val, raw, exact = evaluate(n, delta, order, lam*delta, theta)
                fine, _, _ = evaluate(4*n, delta, order, lam*delta, theta)
                err = np.max(np.linalg.norm(val-exact, axis=-1))
                rate = np.log2(previous/err) if previous else float('nan')
                row = dict(order=order, lam=lam, n=n, delta=delta, corrected_error=err,
                           raw_error=np.max(np.linalg.norm(raw-exact, axis=-1)),
                           quadrature_check=np.max(np.linalg.norm(val-fine, axis=-1)), rate_h=rate)
                rows.append(row)
                previous = err
            print(order, lam, rows[-1])
    dest = Path('results/stokes_derivation')
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'manufactured.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
