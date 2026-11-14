"""Vectorized circle, five-lobed starfish, and nonconvex-kite geometries."""
import numpy as np
from scipy.optimize import minimize_scalar

from .config import (KITE_SECOND_HARMONIC, KITE_VERTICAL_SCALE,
                     STARFISH_AMPLITUDE, STARFISH_BASE_RADIUS,
                     STARFISH_LOBES)


def _frame(x, dx, ddx=None, dddx=None):
    speed = np.linalg.norm(dx, axis=-1)
    tau = dx / speed[..., None]
    normal = np.stack((tau[..., 1], -tau[..., 0]), axis=-1)
    if ddx is None:
        return x, speed, tau, normal
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    kappa = cross(dx, ddx) / speed**3
    if dddx is None:
        return x, speed, tau, normal, kappa
    kappa_s = (cross(dx, dddx) / speed**4
               - 3 * cross(dx, ddx) * np.sum(dx * ddx, axis=-1) / speed**6)
    return x, speed, tau, normal, kappa, kappa_s


def circle_geometry(t):
    t = np.asarray(t, dtype=float)
    x = np.stack((np.cos(t), np.sin(t)), axis=-1)
    dx = np.stack((-np.sin(t), np.cos(t)), axis=-1)
    ddx = -x
    dddx = -dx
    return _frame(x, dx, ddx, dddx)


def laplace_starfish_geometry(t):
    t = np.asarray(t, dtype=float)
    amplitude, lobes = STARFISH_AMPLITUDE, STARFISH_LOBES
    r = STARFISH_BASE_RADIUS + amplitude * np.cos(lobes * t)
    rp = -amplitude * lobes * np.sin(lobes * t)
    rpp = -amplitude * lobes**2 * np.cos(lobes * t)
    rppp = amplitude * lobes**3 * np.sin(lobes * t)
    e = np.stack((np.cos(t), np.sin(t)), axis=-1)
    ep = np.stack((-np.sin(t), np.cos(t)), axis=-1)
    x = r[..., None] * e
    dx = rp[..., None] * e + r[..., None] * ep
    ddx = (rpp - r)[..., None] * e + (2 * rp)[..., None] * ep
    dddx = (rppp - 3 * rp)[..., None] * e + (3 * rpp - r)[..., None] * ep
    return _frame(x, dx, ddx, dddx)


def starfish_level(points):
    points = np.asarray(points, dtype=float)
    theta = np.arctan2(points[..., 1], points[..., 0])
    return (np.linalg.norm(points, axis=-1)
            - (STARFISH_BASE_RADIUS
               + STARFISH_AMPLITUDE * np.cos(STARFISH_LOBES * theta)))


def stokes_kite_geometry(t):
    t = np.asarray(t, dtype=float)
    a, sy = KITE_SECOND_HARMONIC, KITE_VERTICAL_SCALE
    x = np.stack((np.cos(t) + a * np.cos(2 * t) - a,
                  sy * np.sin(t)), axis=-1)
    dx = np.stack((-np.sin(t) - 2 * a * np.sin(2 * t),
                   sy * np.cos(t)), axis=-1)
    ddx = np.stack((-np.cos(t) - 4 * a * np.cos(2 * t),
                    -sy * np.sin(t)), axis=-1)
    dddx = np.stack((np.sin(t) + 8 * a * np.sin(2 * t),
                     -sy * np.cos(t)), axis=-1)
    return _frame(x, dx, ddx, dddx)


def points_inside_curve(points, geometry, samples=4096):
    """Ray-crossing classifier for a smooth sampled closed curve."""
    points = np.asarray(points, dtype=float)
    t = 2 * np.pi * np.arange(samples) / samples
    polygon = geometry(t)[0]
    try:
        from matplotlib.path import Path
        return Path(polygon, closed=True).contains_points(
            points.reshape(-1, 2)).reshape(points.shape[:-1])
    except ImportError:
        pass
    a, b = polygon, np.roll(polygon, -1, axis=0)
    flat = points.reshape(-1, 2)
    inside = np.zeros(len(flat), dtype=bool)
    for start in range(0, len(flat), 2048):
        p = flat[start:start + 2048]
        py = p[:, 1, None]
        crossing_y = (a[None, :, 1] > py) != (b[None, :, 1] > py)
        xhit = ((b[None, :, 0] - a[None, :, 0])
                * (py - a[None, :, 1])
                / (b[None, :, 1] - a[None, :, 1] + 1e-300)
                + a[None, :, 0])
        inside[start:start + len(p)] = np.count_nonzero(
            crossing_y & (p[:, 0, None] < xhit), axis=1) % 2 == 1
    return inside.reshape(points.shape[:-1])


def minimum_pole_distance(pole, geometry, samples=4096):
    """Sample globally, then refine the closest parameter locally."""
    ts = 2 * np.pi * np.arange(samples) / samples
    points = geometry(ts)[0]
    seed = ts[np.argmin(np.abs(points[:, 0] + 1j * points[:, 1] - pole))]
    width = 2 * np.pi / samples
    objective = lambda t: abs(complex(*geometry(np.mod(t, 2 * np.pi))[0]) - pole)**2
    result = minimize_scalar(objective, bounds=(seed - width, seed + width),
                             method="bounded")
    return float(np.sqrt(result.fun))
