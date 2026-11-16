"""Density-subtracted 2D Laplace double layer; outward normals on CCW curves."""
import numpy as np
from .regularized_slp import coefficients


def normals(derivatives):
    dx = np.asarray(derivatives, dtype=float)
    speed = np.linalg.norm(dx, axis=-1)
    if np.any(speed <= 0):
        raise ValueError('curve must be regular')
    return np.stack((dx[..., 1], -dx[..., 0]), axis=-1)/speed[..., None]


def smoothing_over_r2(r2, delta, order, b=0.):
    """Stable T(r²/delta²)/r², including r=0; b broadcasts with r2."""
    if order not in (3, 5, 7):
        raise ValueError('order must be 3, 5, or 7')
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('delta must be positive and finite')
    original_b = np.asarray(b, float)
    r2, b = np.broadcast_arrays(np.asarray(r2, float), original_b)
    if np.any(r2 < 0) or not np.all(np.isfinite(r2)) or not np.all(np.isfinite(b)):
        raise ValueError('r2 and b must be finite, with r2 nonnegative')
    u = r2/delta**2
    # Compute coefficients with target-sized b before broadcasting over sources.
    cs = coefficients(order, np.abs(original_b)/delta)
    pm1_over_u = np.zeros_like(u)
    for k, c in enumerate(cs, 1):
        pm1_over_u += (-1)**k*c*u**(k-1)
    base = np.divide(-np.expm1(-u), u, out=np.ones_like(u), where=u != 0)
    return (base-np.exp(-u)*pm1_over_u)/delta**2


def double_layer(curve, density, targets, delta, n, order=7, *, closest_t,
                 b, chunk_size=128, subtract=True):
    """Return D off curve, or the average boundary operator K on curve.

    curve(t) returns positions and derivatives on a CCW 2*pi-periodic curve.
    density(t) is per-unit-arclength density g(x(t)). closest_t and b are
    supplied true closest parameters and signed distances (positive outside).
    Boundary traces are this value +/- g/2 (interior/exterior).
    subtract=False integrates the differentiated kernel against g directly;
    it does not preserve the jump for targets at distance O(delta).

    On the curve (b=0) the unregularized kernel n(x).(x-y)/(2*pi*|x-y|^2) is
    smooth, with value kappa/(4*pi) at x=y, so the plain trapezoidal rule is
    spectrally accurate there. The regularized kernel is still valid at b=0
    and is used here so that one formula covers targets on and near the curve.
    """
    if not isinstance(n, (int, np.integer)) or n < 4 or chunk_size < 1:
        raise ValueError('require integer n>=4 and positive chunk_size')
    y = np.asarray(targets, float)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError('targets must have shape (M,2)')
    bt = np.broadcast_to(b, (len(y),))
    t0 = np.broadcast_to(closest_t, (len(y),))
    g0 = np.broadcast_to(density(t0), (len(y),)) if subtract else np.zeros(len(y))
    t = 2*np.pi*np.arange(n)/n
    x, dx = curve(t)
    normal = normals(dx)
    weights = np.linalg.norm(dx, axis=1)/n  # includes dt/(2*pi)
    g = np.broadcast_to(density(t), (n,))
    chi = np.where(bt < 0, 1., np.where(bt > 0, 0., .5))
    answer = chi*g0
    for start in range(0, len(y), chunk_size):
        stop = start+chunk_size
        r = x[None, :, :]-y[start:stop, None, :]
        r2 = np.sum(r*r, axis=-1)
        factor = smoothing_over_r2(r2, delta, order, bt[start:stop, None])
        numerator = np.sum(r*normal[None, :, :], axis=-1)
        answer[start:stop] += np.sum(numerator*factor*
                                    (g[None, :]-g0[start:stop, None])*weights, axis=1)
    return answer
