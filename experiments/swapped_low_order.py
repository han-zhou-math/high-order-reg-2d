"""Swapped, scaled kite/starfish problems with low-order manufactured fields."""
import numpy as np

from .allgrid import LaplaceProblem, StokesProblem
from .geometries import (laplace_starfish_geometry, points_inside_curve,
                         stokes_kite_geometry)


BOX = (-1.5, 1.5, -1.5, 1.5)


def _coordinate_scale(geometry, radius=0.895):
    t = 2 * np.pi * np.arange(65536) / 65536
    return radius / np.max(np.abs(geometry(t)[0]))


KITE_SCALE = _coordinate_scale(stokes_kite_geometry)
STARFISH_SCALE = _coordinate_scale(laplace_starfish_geometry)


def _scaled_geometry(geometry, scale):
    def result(t):
        x, speed, tau, normal, kappa, kappa_s = geometry(t)
        return (scale * x, scale * speed, tau, normal,
                kappa / scale, kappa_s / scale**2)
    return result


laplace_kite_geometry = _scaled_geometry(stokes_kite_geometry, KITE_SCALE)
stokes_starfish_geometry = _scaled_geometry(laplace_starfish_geometry,
                                             STARFISH_SCALE)


def _complex(points):
    points = np.asarray(points, dtype=float)
    return points[..., 0] + 1j * points[..., 1]


# Laplace: F- is analytic inside the kite and F+ is analytic outside it.
_LM_POLES = (2.4 + 1.1j, -2.5 + 0.8j)
_LP_POLES = (0.08 + 0.10j, -0.22 - 0.08j)


def laplace_fminus(z):
    a, b = _LM_POLES
    return (0.12 + 0.20 * z + (0.07 - 0.04j) * z**2
            + 0.10 / (z - a) - 0.08j / (z - b))


def laplace_dfminus(z):
    a, b = _LM_POLES
    return (0.20 + 2 * (0.07 - 0.04j) * z
            - 0.10 / (z - a)**2 + 0.08j / (z - b)**2)


def laplace_fplus(z):
    a, b = _LP_POLES
    return (0.16 + 0.03j) / (z - a) - (0.07 - 0.04j) / (z - b)


def laplace_dfplus(z):
    a, b = _LP_POLES
    return -(0.16 + 0.03j) / (z - a)**2 + (0.07 - 0.04j) / (z - b)**2


def _harmonic_value(function, points):
    return function(_complex(points)).real


def _harmonic_gradient(derivative, points):
    value = derivative(_complex(points))
    return np.stack((value.real, -value.imag), axis=-1)


def laplace_inside(points):
    return points_inside_curve(points, laplace_kite_geometry)


def laplace_exact(points, side=None):
    minus = _harmonic_value(laplace_fminus, points)
    plus = _harmonic_value(laplace_fplus, points)
    if side == "inside":
        return minus
    if side == "outside":
        return plus
    if side == "average":
        return 0.5 * (minus + plus)
    return np.where(laplace_inside(points), minus, plus)


def laplace_boundary_densities(t):
    x, _, _, normal, _, _ = laplace_kite_geometry(t)
    grad_minus = _harmonic_gradient(laplace_dfminus, x)
    grad_plus = _harmonic_gradient(laplace_dfplus, x)
    f = np.sum((grad_plus - grad_minus) * normal, axis=-1)
    g = _harmonic_value(laplace_fminus, x) - _harmonic_value(laplace_fplus, x)
    return f, g


# Stokes: degree at most two in both Goursat functions; poles are exterior.
_SF_POLES = (2.5 + 1.0j, -2.4 + 1.2j)
_SG_POLES = (2.3 - 1.4j, -2.6 - 0.9j)


def stokes_f(z):
    a, b = _SF_POLES
    return (0.04 + (0.055 + 0.025j) * z + (0.018 - 0.012j) * z**2
            + 0.035 / (z - a) - 0.025j / (z - b))


def stokes_df(z):
    a, b = _SF_POLES
    return ((0.055 + 0.025j) + 2 * (0.018 - 0.012j) * z
            - 0.035 / (z - a)**2 + 0.025j / (z - b)**2)


def stokes_ddf(z):
    a, b = _SF_POLES
    return (2 * (0.018 - 0.012j) + 0.070 / (z - a)**3
            - 0.050j / (z - b)**3)


def stokes_g(z):
    a, b = _SG_POLES
    return (-0.03j + (0.045 - 0.020j) * z + (0.014 + 0.010j) * z**2
            + 0.030j / (z - a) - 0.022 / (z - b))


def stokes_dg(z):
    a, b = _SG_POLES
    return ((0.045 - 0.020j) + 2 * (0.014 + 0.010j) * z
            - 0.030j / (z - a)**2 + 0.022 / (z - b)**2)


def stokes_ddg(z):
    a, b = _SG_POLES
    return (2 * (0.014 + 0.010j) + 0.060j / (z - a)**3
            - 0.044 / (z - b)**3)


def stokes_velocity(points):
    z = _complex(points)
    a, b, p = stokes_f(z), stokes_df(z), stokes_dg(z)
    psi_x = np.real(a + np.conj(z) * b + p)
    psi_y = np.real(-1j * a + 1j * np.conj(z) * b + 1j * p)
    return np.stack((psi_y, -psi_x), axis=-1)


def stokes_grad_velocity(points):
    z = _complex(points)
    b, c, q = stokes_df(z), stokes_ddf(z), stokes_ddg(z)
    psi_xx = np.real(2 * b + np.conj(z) * c + q)
    psi_xy = np.real(1j * (np.conj(z) * c + q))
    psi_yy = np.real(2 * b - np.conj(z) * c - q)
    return np.stack((np.stack((psi_xy, psi_yy), axis=-1),
                     np.stack((-psi_xx, -psi_xy), axis=-1)), axis=-2)


def stokes_pressure(points, mu=1.0):
    return -4 * mu * np.imag(stokes_df(_complex(points)))


def stokes_grad_pressure(points, mu=1.0):
    value = stokes_ddf(_complex(points))
    return np.stack((-4 * mu * value.imag, -4 * mu * value.real), axis=-1)


def stokes_stress(points, mu=1.0):
    grad = stokes_grad_velocity(points)
    return (-stokes_pressure(points, mu)[..., None, None] * np.eye(2)
            + mu * (grad + np.swapaxes(grad, -1, -2)))


def stokes_inside(points):
    return points_inside_curve(points, stokes_starfish_geometry)


def stokes_boundary_data(t, mu=1.0):
    x, _, _, normal, _, _ = stokes_starfish_geometry(t)
    q = stokes_velocity(x)
    f = np.einsum("...ij,...j->...i", stokes_stress(x, mu), normal)
    return q, f


def stokes_exact(points, side=None, mu=1.0):
    u = stokes_velocity(points)
    p = stokes_pressure(points, mu)
    if side == "inside":
        return u, p
    if side == "outside":
        return np.zeros_like(u), np.zeros_like(p)
    if side == "average":
        return 0.5 * u, 0.5 * p
    inside = stokes_inside(points)
    return np.where(inside[..., None], u, 0.0), np.where(inside, p, 0.0)


def stokes_piecewise_grad_pressure(points):
    inside = stokes_inside(points)
    return np.where(inside[..., None], stokes_grad_pressure(points), 0.0)


LAPLACE_KITE = LaplaceProblem(
    name="scaled_kite_low_order_r0895",
    geometry=laplace_kite_geometry,
    inside=laplace_inside,
    boundary_densities=laplace_boundary_densities,
    exact=laplace_exact,
    box=BOX,
)


STOKES_STARFISH = StokesProblem(
    name="scaled_starfish_low_order_r0895",
    geometry=stokes_starfish_geometry,
    inside=stokes_inside,
    boundary_data=stokes_boundary_data,
    exact=stokes_exact,
    exact_gradient=stokes_piecewise_grad_pressure,
    box=BOX,
)
