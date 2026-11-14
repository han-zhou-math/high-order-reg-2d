"""Exact Goursat manufactured Stokes field on the nonconvex kite."""
import numpy as np

from .config import (KITE_F_COEFFICIENTS, KITE_F_POLES,
                     KITE_G_COEFFICIENTS, KITE_G_POLES)
from .geometries import points_inside_curve, stokes_kite_geometry


def F(z):
    a1, a2 = KITE_F_POLES
    c = KITE_F_COEFFICIENTS; r1, r2 = c["residues"]
    return (c["z4"] * z**4 + c["z2"] * z**2
            + c["exponential"] * np.exp(c["exponent"] * z)
            + r1 / (z - a1) + r2 / (z - a2))


def dF(z):
    a1, a2 = KITE_F_POLES
    c = KITE_F_COEFFICIENTS; r1, r2 = c["residues"]
    return (4 * c["z4"] * z**3 + 2 * c["z2"] * z
            + c["exponential"] * c["exponent"] * np.exp(c["exponent"] * z)
            - r1 / (z - a1)**2 - r2 / (z - a2)**2)


def ddF(z):
    a1, a2 = KITE_F_POLES
    c = KITE_F_COEFFICIENTS; r1, r2 = c["residues"]
    return (12 * c["z4"] * z**2 + 2 * c["z2"]
            + c["exponential"] * c["exponent"]**2 * np.exp(c["exponent"] * z)
            + 2 * r1 / (z - a1)**3 + 2 * r2 / (z - a2)**3)


def G(z):
    a3, a4 = KITE_G_POLES
    c = KITE_G_COEFFICIENTS; r1, r2 = c["residues"]
    return (c["z5"] * z**5 + c["z3"] * z**3
            + c["exponential"] * np.exp(c["exponent"] * z)
            + r1 / (z - a3) + r2 / (z - a4))


def dG(z):
    a3, a4 = KITE_G_POLES
    c = KITE_G_COEFFICIENTS; r1, r2 = c["residues"]
    return (5 * c["z5"] * z**4 + 3 * c["z3"] * z**2
            + c["exponential"] * c["exponent"] * np.exp(c["exponent"] * z)
            - r1 / (z - a3)**2 - r2 / (z - a4)**2)


def ddG(z):
    a3, a4 = KITE_G_POLES
    c = KITE_G_COEFFICIENTS; r1, r2 = c["residues"]
    return (20 * c["z5"] * z**3 + 6 * c["z3"] * z
            + c["exponential"] * c["exponent"]**2 * np.exp(c["exponent"] * z)
            + 2 * r1 / (z - a3)**3 + 2 * r2 / (z - a4)**3)


def _complex(points):
    points = np.asarray(points, dtype=float)
    return points[..., 0] + 1j * points[..., 1]


def psi(points):
    z = _complex(points)
    return np.real(np.conj(z) * F(z) + G(z))


def velocity(points):
    z = _complex(points); a, b, p = F(z), dF(z), dG(z)
    psi_x = np.real(a + np.conj(z) * b + p)
    psi_y = np.real(-1j * a + 1j * np.conj(z) * b + 1j * p)
    return np.stack((psi_y, -psi_x), axis=-1)


def grad_velocity(points):
    z = _complex(points); b, c, q = dF(z), ddF(z), ddG(z)
    psi_xx = np.real(2 * b + np.conj(z) * c + q)
    psi_xy = np.real(1j * (np.conj(z) * c + q))
    psi_yy = np.real(2 * b - np.conj(z) * c - q)
    return np.stack((np.stack((psi_xy, psi_yy), axis=-1),
                     np.stack((-psi_xx, -psi_xy), axis=-1)), axis=-2)


def pressure(points, mu=1.0):
    return -4 * mu * np.imag(dF(_complex(points)))


def grad_pressure(points, mu=1.0):
    c = ddF(_complex(points))
    return np.stack((-4 * mu * c.imag, -4 * mu * c.real), axis=-1)


def stress(points, mu=1.0):
    grad = grad_velocity(points)
    identity = np.eye(2)
    return (-pressure(points, mu)[..., None, None] * identity
            + mu * (grad + np.swapaxes(grad, -1, -2)))


def stokes_boundary_data(t, mu=1.0):
    x, _, _, normal, _, _ = stokes_kite_geometry(t)
    q = velocity(x)
    f = np.einsum("...ij,...j->...i", stress(x, mu), normal)
    return q, f


def exact_piecewise_stokes(points, side=None, mu=1.0):
    points = np.asarray(points, dtype=float)
    if side == "inside":
        return velocity(points), pressure(points, mu)
    if side == "outside":
        return np.zeros(points.shape), np.zeros(points.shape[:-1])
    if side == "average":
        return 0.5 * velocity(points), 0.5 * pressure(points, mu)
    inside = points_inside_curve(points, stokes_kite_geometry)
    u = np.zeros(points.shape); p = np.zeros(points.shape[:-1])
    u[inside] = velocity(points[inside]); p[inside] = pressure(points[inside], mu)
    return u, p
