"""Lightweight geometry and exact-solution validation; no convergence runs."""
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1

from .config import (KITE_F_POLES, KITE_G_POLES, STARFISH_FMINUS_POLES,
                     STARFISH_FPLUS_POLES)
from .geometries import (laplace_starfish_geometry, minimum_pole_distance,
                         circle_geometry, points_inside_curve, starfish_level,
                         stokes_kite_geometry)
from .manufactured_laplace import (Fminus, Fplus, laplace_boundary_densities)
from .manufactured_stokes import (F, G, grad_pressure, grad_velocity, pressure,
                                  stokes_boundary_data, stress, velocity)
from regbie.regularized_slp import coefficients
from regbie.surface_operators import radial_factors


def validate_starfish_poles():
    inside_poles = [starfish_level(np.array([z.real, z.imag])) < 0
                    for z in STARFISH_FPLUS_POLES]
    outside_poles = [starfish_level(np.array([z.real, z.imag])) > 0
                     for z in STARFISH_FMINUS_POLES]
    if not all(inside_poles) or not all(outside_poles):
        raise ValueError("a Laplace pole is on the wrong side of the starfish")
    poles = STARFISH_FMINUS_POLES + STARFISH_FPLUS_POLES
    return {str(z): minimum_pole_distance(z, laplace_starfish_geometry)
            for z in poles}


def validate_kite_poles():
    poles = KITE_F_POLES + KITE_G_POLES
    points = np.array([[z.real, z.imag] for z in poles])
    if np.any(points_inside_curve(points, stokes_kite_geometry)):
        raise ValueError("a Goursat pole lies inside the kite")
    return {str(z): minimum_pole_distance(z, stokes_kite_geometry) for z in poles}


def validate_boundary_finiteness(samples=512):
    t = 2 * np.pi * np.arange(samples) / samples
    xl = laplace_starfish_geometry(t)[0]
    zl = xl[:, 0] + 1j * xl[:, 1]
    lf, lg = laplace_boundary_densities(t)
    sq, sf = stokes_boundary_data(t)
    arrays = (Fminus(zl), Fplus(zl), lf, lg, sq, sf)
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError("nonfinite manufactured boundary data")


def finite_difference_exactness(samples=12, step=2e-5):
    """Independent Cartesian checks of harmonic and Stokes identities."""
    rng = np.random.default_rng(20260911)
    points = rng.uniform((-0.5, -0.5), (0.5, 0.5), size=(samples, 2))
    ex = np.array([step, 0.0]); ey = np.array([0.0, step])

    def laplacian_scalar(fun):
        return ((fun(points + ex) - 2 * fun(points) + fun(points - ex))
                + (fun(points + ey) - 2 * fun(points) + fun(points - ey))) / step**2

    from .manufactured_laplace import uminus
    laplace_residual = float(np.max(np.abs(laplacian_scalar(uminus))))
    u = velocity(points)
    dux = (velocity(points + ex) - velocity(points - ex)) / (2 * step)
    duy = (velocity(points + ey) - velocity(points - ey)) / (2 * step)
    divergence = float(np.max(np.abs(dux[:, 0] + duy[:, 1])))
    lap_u = np.stack([laplacian_scalar(lambda x, k=k: velocity(x)[:, k])
                      for k in range(2)], axis=-1)
    momentum = float(np.max(np.linalg.norm(-lap_u + grad_pressure(points), axis=-1)))
    return {"laplace_residual": laplace_residual,
            "divergence_residual": divergence,
            "momentum_residual": momentum}


def audit_regularizers():
    """Independent moment and radial-identity residuals at representative values."""
    largest_moment = 0.0
    largest_radial = 0.0
    values = {}
    for order in (3, 5, 7):
        m = (order - 1) // 2
        for lam in (0.0, 0.5, 1.0, 2.0):
            cs = coefficients(order, lam)
            values[f"p{order}_lambda{lam:g}"] = [float(c) for c in cs]
            for j in range(m):
                def correction(t):
                    xi = t * t + lam * lam
                    poly = -cs[0]
                    if order >= 5:
                        poly += cs[1] * (1 + xi)
                    if order >= 7:
                        poly -= cs[2] * (2 + 2 * xi + xi * xi)
                    return (exp1(xi) + np.exp(-xi) * poly) / (4 * np.pi)
                residual = 2 * quad(lambda t: t**(2*j) * correction(t), 0,
                                    np.inf, epsabs=2e-12, epsrel=2e-12)[0]
                largest_moment = max(largest_moment, abs(residual))
            for xi in (1e-7, 0.1, 1.0, 4.0):
                polynomial = 1 + sum((-1)**k * c * xi**k
                                     for k, c in enumerate(cs, 1))
                q_m = np.exp(-xi) * polynomial
                s2 = 1 - q_m
                a, ap, tp = radial_factors(xi, order, lam)
                # A=s2/xi, s3/xi^2=-A', and s2'=T'.
                polynomial_prime = sum(k * (-1)**k * c * xi**(k-1)
                                       for k, c in enumerate(cs, 1))
                s2_prime = np.exp(-xi) * (polynomial - polynomial_prime)
                s3 = s2 - xi * s2_prime
                largest_radial = max(largest_radial, abs(xi * a - s2),
                                     abs(tp - s2_prime),
                                     abs(-xi * xi * ap - s3))
    return {"coefficients": values,
            "maximum_cancelled_moment_residual": float(largest_moment),
            "maximum_radial_identity_residual": float(largest_radial)}


def audit_orientation(samples=2048, offset=1e-5):
    reports = {}
    for name, geometry, inside in (
        ("circle", circle_geometry,
         lambda p: np.linalg.norm(p, axis=-1) < 1),
        ("five_lobed_starfish", laplace_starfish_geometry,
         lambda p: starfish_level(p) < 0),
        ("nonconvex_kite", stokes_kite_geometry,
         lambda p: points_inside_curve(p, stokes_kite_geometry)),
    ):
        t = 2 * np.pi * np.arange(samples) / samples
        x, _, _, normal, _, _ = geometry(t)
        signed_area = 0.5 * np.sum(x[:, 0] * np.roll(x[:, 1], -1)
                                   - x[:, 1] * np.roll(x[:, 0], -1))
        outward_ok = np.mean(~inside(x + offset * normal))
        inward_ok = np.mean(inside(x - offset * normal))
        reports[name] = {"signed_area": float(signed_area),
                         "counterclockwise": bool(signed_area > 0),
                         "outward_fraction": float(outward_ok),
                         "inward_fraction": float(inward_ok)}
    return reports


def audit_stokes_boundary(samples=1024, step=2e-6):
    from .boundary_data import pressure_densities
    # pressure_densities receives values on the canonical FFT grid.
    t = 2 * np.pi * np.arange(samples) / samples
    x, _, _, normal, _, _ = stokes_kite_geometry(t)
    q, f = stokes_boundary_data(t)
    ex, ey = np.array([step, 0.0]), np.array([0.0, step])
    grad_fd = np.stack(((velocity(x + ex) - velocity(x - ex)) / (2 * step),
                        (velocity(x + ey) - velocity(x - ey)) / (2 * step)), axis=-1)
    stress_fd = (-pressure(x)[..., None, None] * np.eye(2)
                 + grad_fd + np.swapaxes(grad_fd, -1, -2))
    traction_fd = np.einsum("...ij,...j->...i", stress_fd, normal)
    pdata = pressure_densities(stokes_kite_geometry, q, f)
    psi_residual = np.max(np.abs(pdata.psi + pressure(x)))
    phi_residual = np.max(np.abs(
        pdata.phi + np.sum(grad_pressure(x) * normal, axis=-1)))
    return {"gradient_fd_error_inf": float(np.max(np.abs(grad_fd - grad_velocity(x)))),
            "traction_fd_error_inf": float(np.max(np.linalg.norm(traction_fd - f, axis=-1))),
            "stress_symmetry_error_inf": float(np.max(np.abs(stress(x) - np.swapaxes(stress(x), -1, -2)))),
            "psi_pressure_jump_residual_inf": float(psi_residual),
            "phi_normal_pressure_jump_residual_inf": float(phi_residual)}


def validate_all():
    validate_boundary_finiteness()
    return {"starfish_pole_distances": validate_starfish_poles(),
            "kite_pole_distances": validate_kite_poles(),
            "exactness": finite_difference_exactness(),
            "regularizers": audit_regularizers(),
            "orientation": audit_orientation(),
            "stokes_boundary": audit_stokes_boundary()}
