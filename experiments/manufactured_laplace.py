"""Exact circle and five-lobed-starfish Laplace benchmarks."""
import numpy as np

from .config import (CIRCLE_COS3, CIRCLE_SIN5,
                     STARFISH_FMINUS_COEFFICIENTS, STARFISH_FMINUS_POLES,
                     STARFISH_FPLUS_POLES, STARFISH_FPLUS_RESIDUES)
from .geometries import laplace_starfish_geometry, starfish_level


def circle_density(t):
    return 1 + CIRCLE_COS3 * np.cos(3 * t) + CIRCLE_SIN5 * np.sin(5 * t)


def exact_circle_single(r, theta):
    r = np.asarray(r, dtype=float)
    safe = np.where(r > 0, r, 1.0)
    radial3 = np.where(r <= 1, r**3, safe**-3)
    radial5 = np.where(r <= 1, r**5, safe**-5)
    return (np.log(np.maximum(r, 1)) - CIRCLE_COS3 / 6 * radial3 * np.cos(3 * theta)
            - CIRCLE_SIN5 / 10 * radial5 * np.sin(5 * theta))


def exact_circle_double(r, theta, side=None):
    r = np.asarray(r, dtype=float)
    safe = np.where(r > 0, r, 1.0)
    interior = (1 + CIRCLE_COS3 / 2 * r**3 * np.cos(3 * theta)
                + CIRCLE_SIN5 / 2 * r**5 * np.sin(5 * theta))
    exterior = (-CIRCLE_COS3 / 2 * safe**-3 * np.cos(3 * theta)
                - CIRCLE_SIN5 / 2 * safe**-5 * np.sin(5 * theta))
    if side == "average":
        return np.full(np.broadcast(r, theta).shape, 0.5)
    return np.where(r < 1, interior, np.where(r > 1, exterior, 0.5))


def Fminus(z):
    a1, a2 = STARFISH_FMINUS_POLES
    c = STARFISH_FMINUS_COEFFICIENTS; r1, r2 = c["residues"]
    return (c["exponential"] * np.exp(c["exponent"] * z)
            + c["z4"] * z**4 + c["z3"] * z**3
            + r1 / (z - a1) + r2 / (z - a2))


def dFminus(z):
    a1, a2 = STARFISH_FMINUS_POLES
    c = STARFISH_FMINUS_COEFFICIENTS; r1, r2 = c["residues"]
    return (c["exponential"] * c["exponent"] * np.exp(c["exponent"] * z)
            + 4 * c["z4"] * z**3 + 3 * c["z3"] * z**2
            - r1 / (z - a1)**2 - r2 / (z - a2)**2)


def Fplus(z):
    z1, z2, z3 = STARFISH_FPLUS_POLES
    r1, r2, r3 = STARFISH_FPLUS_RESIDUES
    return r1 / (z - z1) + r2 / (z - z2)**2 + r3 / (z - z3)**3


def dFplus(z):
    z1, z2, z3 = STARFISH_FPLUS_POLES
    r1, r2, r3 = STARFISH_FPLUS_RESIDUES
    return -r1 / (z - z1)**2 - 2 * r2 / (z - z2)**3 - 3 * r3 / (z - z3)**4


def _complex(points):
    points = np.asarray(points, dtype=float)
    return points[..., 0] + 1j * points[..., 1]


def uminus(points):
    return Fminus(_complex(points)).real


def uplus(points):
    return Fplus(_complex(points)).real


def grad_uminus(points):
    value = dFminus(_complex(points))
    return np.stack((value.real, -value.imag), axis=-1)


def grad_uplus(points):
    value = dFplus(_complex(points))
    return np.stack((value.real, -value.imag), axis=-1)


def laplace_boundary_densities(t):
    x, _, _, normal, _, _ = laplace_starfish_geometry(t)
    f = np.sum((grad_uplus(x) - grad_uminus(x)) * normal, axis=-1)
    g = uminus(x) - uplus(x)
    return f, g


def exact_piecewise_laplace(points, side=None):
    points = np.asarray(points, dtype=float)
    if side == "inside":
        return uminus(points)
    if side == "outside":
        return uplus(points)
    if side == "average":
        return 0.5 * (uminus(points) + uplus(points))
    inside = starfish_level(points) < 0
    result = np.empty(points.shape[:-1])
    result[inside] = uminus(points[inside])
    result[~inside] = uplus(points[~inside])
    return result
