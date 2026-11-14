"""2D Laplace single layer, G(r)=+log(r)/(2*pi).

Kernels from regularized_2d.pdf, equations (19), (26)--(32).
Only smooth, closed, 2*pi-periodic parameterizations are assumed.
"""
import numpy as np
from scipy.special import exp1, erfcx
from scipy.integrate import quad

ORDERS = (1, 3, 5, 7)


def coefficients(order, lam):
    """Return c_1,...,c_m; lam=abs(target distance)/delta (broadcastable)."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    lam = np.asarray(lam, dtype=float)
    if np.any(~np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError("lam must be finite and nonnegative")
    if order == 1:
        return ()
    # Recurrence (19) near the curve; positive integral (17) avoids
    # cancellation for large lambda. This branch is rarely needed for b=O(delta).
    z = lam**2
    h0 = 2 * (1 - np.sqrt(np.pi) * lam * erfcx(lam))
    h1 = (1-z*h0)/1.5
    h2 = (1-z*h1)/2.5
    # The differences also need stable evaluation: they decay as z^-2
    # and z^-3, while the individual H_j decay only as z^-1.
    h0, d1, d2 = [np.array(v, copy=True) for v in
                  (h0, h0-h1, h0-2*h1+h2)]
    for idx in np.ndindex(lam.shape):
        if lam[idx] > 1:
            zz = z[idx]
            for j, dest in enumerate((h0, d1, d2)):
                # Positive integral, scaled to stay O(1) during quadrature.
                val = quad(lambda s: np.exp(-s)*s**j*(1+s/zz)**(-j-1.5),
                           0, np.inf, epsabs=2e-14, epsrel=2e-14)[0]
                dest[idx] = val/zz**(j+1)
    if order == 3:
        return (h0,)
    if order == 5:
        c2 = d1
        return (h0+(z+1.5)*c2, c2)
    c3 = d2/2
    c2 = d1+(2*z+5)*c3
    c1 = h0+(z+1.5)*c2-(z*z+3*z+3.75)*c3
    return c1, c2, c3


def regularized_kernel(r, delta, order=7, b=0.0):
    """Finite kernel including r=0. r and b follow NumPy broadcasting."""
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    r = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("r must be finite and nonnegative")
    cs = coefficients(order, np.abs(np.asarray(b))/delta)
    u = (r/delta)**2
    # log(u)+E1(u)=-gamma+sum_{k>=1}(-1)^(k+1)u^k/(k*k!).
    # Evaluate the combined expression, never inf-inf at coincident nodes.
    small = u < 0.5
    base = np.empty_like(u)
    us = u[small]
    series = np.zeros_like(us)
    term = np.ones_like(us)
    for k in range(1, 19):
        term *= -us/k
        series -= term/k
    base[small] = 2*np.log(delta)-np.euler_gamma+series
    base[~small] = 2*np.log(r[~small])+exp1(u[~small])
    correction = np.zeros(np.broadcast_shapes(u.shape, np.shape(b)))
    if order >= 3:
        correction = correction-cs[0]
    if order >= 5:
        correction = correction+cs[1]*(1+u)
    if order >= 7:
        correction = correction-cs[2]*(2+2*u+u*u)
    return (base+np.exp(-u)*correction)/(4*np.pi)


def single_layer(curve, density, targets, delta, n, order=7, b=0.0,
                 chunk_size=128):
    """Composite periodic trapezoid: dt sum G_delta f(x(t_j)) |x'(t_j)|.

    curve(t) -> (points[N,2], derivatives[N,2]); density(t) -> values[N].
    targets: (M,2); b: scalar or length M, true signed closest distances.
    For general curves the caller must supply these distances; b=0 is only
    appropriate for on-curve targets. Density is per unit arclength.
    """
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer >= 4")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    t = 2*np.pi*np.arange(n)/n
    x, dx = curve(t)
    x, dx = np.asarray(x), np.asarray(dx)
    if x.shape != (n, 2) or dx.shape != (n, 2):
        raise ValueError("curve must return two (n,2) arrays")
    weights = np.broadcast_to(density(t), (n,))*np.linalg.norm(dx, axis=1)*2*np.pi/n
    y = np.asarray(targets, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("targets must have shape (M,2)")
    distances = np.broadcast_to(b, (len(y),))
    result = np.empty(len(y))
    for start in range(0, len(y), chunk_size):
        stop = start+chunk_size
        r = np.linalg.norm(y[start:stop, None, :]-x[None, :, :], axis=2)
        result[start:stop] = regularized_kernel(
            r, delta, order, distances[start:stop, None]) @ weights
    return result


def ellipse(t, a=1.4, b=0.8):
    return (np.column_stack((a*np.cos(t), b*np.sin(t))),
            np.column_stack((-a*np.sin(t), b*np.cos(t))))


def circle(t):
    return ellipse(t, 1., 1.)


def starfish(t):
    """Smooth nonconvex three-lobed curve, radius 1+0.2*cos(3t)."""
    radius = 1+0.2*np.cos(3*t)
    dr = -0.6*np.sin(3*t)
    return (np.column_stack((radius*np.cos(t), radius*np.sin(t))),
            np.column_stack((dr*np.cos(t)-radius*np.sin(t),
                             dr*np.sin(t)+radius*np.cos(t))))
