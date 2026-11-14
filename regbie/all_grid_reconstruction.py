"""Pressure validation and stencil-aware all-grid reconstruction.

The boundary kernels are imported from the existing Laplace and Stokes
implementations.  This module adds pressure densities, fourth-order Cartesian
operators, stencil classification, a nine-point DST solver, and reproducible
manufactured-solution experiments.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .manufactured_convergence import densities as laplace_densities
from .manufactured_convergence import field as laplace_field
from .regularized_dlp import double_layer, normals
from .regularized_slp import regularized_kernel, single_layer
from .surface_operators import radial_factors
from .verify_stokes_derivation import (
    field as stokes_field,
    q_derivatives,
    scaled_moment_n0,
    traction,
)


BOX = (-1.6, 1.6, -1.6, 1.6)
SHIFT = np.array([0.037, -0.023])


def curve_derivatives(t):
    """Shifted three-lobed curve and its first three parameter derivatives."""
    t = np.asarray(t, dtype=float)
    modes = ((1, 1.0), (4, 0.1), (-2, 0.1))
    z = [sum(c * (1j * k) ** j * np.exp(1j * k * t) for k, c in modes)
         for j in range(4)]
    arrays = [np.stack((v.real, v.imag), axis=-1) for v in z]
    arrays[0] = arrays[0] + SHIFT
    return arrays


def curve(t):
    x, dx, _, _ = curve_derivatives(t)
    return x, dx


def geometry(t):
    x, dx, ddx, dddx = curve_derivatives(t)
    speed = np.linalg.norm(dx, axis=-1)
    tau = dx / speed[..., None]
    normal = np.stack((tau[..., 1], -tau[..., 0]), axis=-1)
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    kappa = cross(dx, ddx) / speed ** 3
    kappa_s = (cross(dx, dddx) / speed ** 4
               - 3 * cross(dx, ddx) * np.sum(dx * ddx, axis=-1) / speed ** 6)
    return x, speed, tau, normal, kappa, kappa_s


def level_value(points):
    """Radial defining function: negative inside the shifted starfish."""
    z = np.asarray(points) - SHIFT
    theta = np.arctan2(z[..., 1], z[..., 0])
    return np.linalg.norm(z, axis=-1) - (1 + 0.2 * np.cos(3 * theta))


def closest_points(targets, samples=4096):
    """Closest parameters and signed distances for near-interface targets."""
    targets = np.asarray(targets, dtype=float)
    ts = 2 * np.pi * np.arange(samples) / samples
    xs, _ = curve(ts)
    _, index = cKDTree(xs).query(targets)
    t = ts[index]
    for _ in range(10):
        x, dx, ddx, _ = curve_derivatives(t)
        r = x - targets
        first = np.sum(r * dx, axis=-1)
        second = np.sum(dx * dx, axis=-1) + np.sum(r * ddx, axis=-1)
        step = first / second
        t = np.mod(t - step, 2 * np.pi)
        if np.max(np.abs(step)) < 2e-14:
            break
    x0, dx0 = curve(t)
    n0 = normals(dx0)
    residual = np.abs(np.sum((x0 - targets) * dx0, axis=-1))
    if np.max(residual) > 2e-9:
        raise RuntimeError(f"closest-point residual too large: {np.max(residual):.3e}")
    b = np.sum((targets - x0) * n0, axis=-1)
    expected = np.where(level_value(targets) < 0, -1, 1)
    actual = np.where(b < 0, -1, 1)
    if np.any(expected != actual):
        raise RuntimeError("closest-point side disagrees with phase classifier")
    return t, b


class PeriodicFourier:
    """Periodic trigonometric interpolant, with exact reuse on its sample grid."""

    def __init__(self, samples):
        self.samples = np.asarray(samples)
        self.n = self.samples.shape[0]
        self.coefficients = np.fft.fft(self.samples, axis=0) / self.n
        self.modes = np.fft.fftfreq(self.n, d=1 / self.n)

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        reference = 2 * np.pi * np.arange(self.n) / self.n
        if t.shape == reference.shape and np.max(np.abs(t - reference)) < 1e-13:
            return self.samples.copy()
        flat = t.ravel()
        result_shape = (flat.size,) + self.samples.shape[1:]
        out = np.empty(result_shape, dtype=float)
        for start in range(0, flat.size, 128):
            stop = min(start + 128, flat.size)
            phase = np.exp(1j * flat[start:stop, None] * self.modes[None, :])
            out[start:stop] = np.einsum("mk,k...->m...", phase,
                                        self.coefficients).real
        return out.reshape(t.shape + self.samples.shape[1:])


def spectral_derivative(samples, order=1):
    samples = np.asarray(samples)
    n = samples.shape[0]
    modes = np.fft.fftfreq(n, d=1 / n)
    multiplier = (1j * modes) ** order
    shape = (n,) + (1,) * (samples.ndim - 1)
    return np.fft.ifft(np.fft.fft(samples, axis=0) * multiplier.reshape(shape),
                       axis=0).real


@dataclass
class PressureDensities:
    phi: PeriodicFourier
    psi: PeriodicFourier
    jump_residual: float
    normal_jump_residual: float


def pressure_densities(n_boundary, mu=1.0):
    """Construct phi_p and psi_p using periodic spectral differentiation."""
    t = 2 * np.pi * np.arange(n_boundary) / n_boundary
    x, speed, tau, normal, _, _ = geometry(t)
    f = traction(x, normal)
    q = stokes_field(x)
    q_s = spectral_derivative(q) / speed[:, None]
    f_tau_s = spectral_derivative(np.sum(f * tau, axis=-1)) / speed
    q_s_normal_s = spectral_derivative(np.sum(q_s * normal, axis=-1)) / speed
    phi = f_tau_s - 2 * mu * q_s_normal_s
    psi = np.sum(f * normal, axis=-1) + 2 * mu * np.sum(q_s * tau, axis=-1)
    jump_residual = np.max(np.abs(psi + 2 * x[:, 0]))
    normal_jump_residual = np.max(np.abs(phi + 2 * normal[:, 0]))
    return PressureDensities(PeriodicFourier(phi), PeriodicFourier(psi),
                             float(jump_residual), float(normal_jump_residual))


def eval_laplace_regularized(targets, closest_t, b, delta, n_boundary,
                             f, g, order=7, d_sign=1.0):
    sl = single_layer(curve, f, targets, delta, n_boundary, order, b)
    dl = double_layer(curve, g, targets, delta, n_boundary, order,
                      closest_t=closest_t, b=b)
    return sl + d_sign * dl


def eval_pressure_regularized(targets, closest_t, b, delta, n_boundary,
                              pressure, order=7):
    return eval_laplace_regularized(targets, closest_t, b, delta, n_boundary,
                                    pressure.phi, pressure.psi, order, -1.0)


def eval_laplace_unregularized(targets, n_boundary, f, g, d_sign=1.0,
                               chunk_size=128):
    """Ordinary trapezoidal S + d_sign*D for targets separated from Gamma."""
    targets = np.asarray(targets)
    t = 2 * np.pi * np.arange(n_boundary) / n_boundary
    x, dx = curve(t)
    speed = np.linalg.norm(dx, axis=-1)
    normal = normals(dx)
    fv, gv = f(t), g(t)
    out = np.empty(len(targets))
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        r = x[None, :, :] - targets[start:stop, None, :]
        r2 = np.sum(r * r, axis=-1)
        sl = np.sum(np.log(np.sqrt(r2)) * (fv * speed)[None, :], axis=1) / n_boundary
        dl = np.sum(np.sum(r * normal[None, :, :], axis=-1) / r2
                    * (gv * speed)[None, :], axis=1) / n_boundary
        out[start:stop] = sl + d_sign * dl
    return out


def eval_stokes_regularized(targets, closest_t, b, delta, n_boundary,
                            order=7, mu=1.0, chunk_size=64):
    """Existing corrected Stokes representation evaluated at arbitrary targets."""
    targets = np.asarray(targets)
    closest_t = np.asarray(closest_t)
    b = np.asarray(b)
    t = 2 * np.pi * np.arange(n_boundary) / n_boundary
    x, speed, _, normal, _, _ = geometry(t)
    f, q = traction(x, normal), stokes_field(x)
    x0, _, tau0, n0, kappa, kappa_s = geometry(closest_t)
    f0, q0 = traction(x0, n0), stokes_field(x0)
    q_s, q_ss = q_derivatives(x0, tau0, n0, kappa)
    weights = 2 * np.pi * speed / n_boundary
    answer = np.empty((len(targets), 2))
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        slc = slice(start, stop)
        r = x[None, :, :] - targets[slc, None, :]
        r2 = np.sum(r * r, axis=-1)
        u = r2 / delta ** 2
        a, ap, _ = radial_factors(u, order, np.abs(b[slc, None]) / delta)
        gf = regularized_kernel(np.sqrt(r2), delta, order, b[slc, None])
        rf = np.sum(r * f[None, :, :], axis=-1)
        sl_integrand = (r * rf[..., None] * a[..., None] / (4 * np.pi * delta ** 2)
                        - gf[..., None] * f[None, :, :] / 2)
        rq = np.sum(r * (q[None, :, :] - q0[slc, None, :]), axis=-1)
        rn = np.sum(r * normal[None, :, :], axis=-1)
        dl_integrand = r * (rq * rn * (-ap) / (np.pi * delta ** 4))[..., None]
        sl_value = np.einsum("mni,n->mi", sl_integrand, weights)
        chi = np.where(b[slc] < 0, 1.0, np.where(b[slc] > 0, 0.0, 0.5))
        dl_value = np.einsum("mni,n->mi", dl_integrand, weights) + chi[:, None] * q0[slc]
        alpha = np.sqrt(1 + b[slc] * kappa[slc])
        scale = np.abs(b[slc]) * np.array([
            scaled_moment_n0(order, abs(value) / delta) for value in b[slc]])
        b0 = n0[slc] * np.sum(n0[slc] * f0[slc], axis=-1)[:, None] / alpha[:, None]
        sym = (n0[slc] * np.sum(tau0[slc] * q_s[slc], axis=-1)[:, None]
               + tau0[slc] * np.sum(n0[slc] * q_s[slc], axis=-1)[:, None])
        c2 = sym / alpha[:, None] ** 3
        c2 -= (b[slc] / (2 * alpha ** 3))[:, None] * n0[slc] * np.sum(
            n0[slc] * q_ss[slc], axis=-1)[:, None]
        c2 += (b[slc] ** 2 * kappa_s[slc] / (2 * alpha ** 5))[:, None] * n0[slc] * np.sum(
            n0[slc] * q_s[slc], axis=-1)[:, None]
        answer[slc] = (sl_value + scale[:, None] * b0 / (4 * np.pi * mu)
                       + dl_value + scale[:, None] * c2 / (2 * np.pi))
    return answer


def eval_stokes_unregularized(targets, n_boundary, mu=1.0, chunk_size=64):
    targets = np.asarray(targets)
    t = 2 * np.pi * np.arange(n_boundary) / n_boundary
    x, speed, _, normal, _, _ = geometry(t)
    f, q = traction(x, normal), stokes_field(x)
    weights = 2 * np.pi * speed / n_boundary
    answer = np.empty((len(targets), 2))
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        r = x[None, :, :] - targets[start:stop, None, :]
        r2 = np.sum(r * r, axis=-1)
        rf = np.sum(r * f[None, :, :], axis=-1)
        sl = (-np.log(np.sqrt(r2))[..., None] * f[None, :, :]
              + r * (rf / r2)[..., None]) / (4 * np.pi * mu)
        rq = np.sum(r * q[None, :, :], axis=-1)
        rn = np.sum(r * normal[None, :, :], axis=-1)
        dl = r * (rq * rn / (np.pi * r2 ** 2))[..., None]
        answer[start:stop] = np.einsum("mni,n->mi", sl + dl, weights)
    return answer


LAPLACE_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1))
GRADIENT_OFFSETS = ((0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0),
                    (0, 1), (0, -1), (0, 2), (0, -2))


def grid_coordinates(m, box=BOX):
    xmin, xmax, ymin, ymax = box
    if abs((xmax - xmin) - (ymax - ymin)) > 1e-14:
        raise ValueError("the current nine-point solver requires equal spacing")
    h = (xmax - xmin) / (m + 1)
    x = np.linspace(xmin, xmax, m + 2)
    y = np.linspace(ymin, ymax, m + 2)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, h


def classify_crossing(xx, yy, h, offsets):
    """Classify interface-crossing stencils by sides plus segment sampling."""
    center = np.stack((xx[1:-1, 1:-1], yy[1:-1, 1:-1]), axis=-1)
    center_inside = level_value(center) < 0
    crossing = np.zeros(center_inside.shape, dtype=bool)
    for di, dj in offsets:
        if di == dj == 0:
            continue
        endpoint = center + np.array([di * h, dj * h])
        crossing |= (level_value(endpoint) < 0) != center_inside
        # This local geometric check catches a crossing between same-side nodes.
        for fraction in (0.25, 0.5, 0.75):
            sample = center + fraction * np.array([di * h, dj * h])
            crossing |= (level_value(sample) < 0) != center_inside
    crossing |= np.abs(level_value(center)) < 100 * np.finfo(float).eps
    return crossing


def stencil_halo(mask, offsets=LAPLACE_OFFSETS):
    """Return a full-grid mask containing all stencil nodes of masked centers."""
    m, n = mask.shape
    halo = np.zeros((m + 2, n + 2), dtype=bool)
    for di, dj in offsets:
        halo[1 + dj:1 + dj + m, 1 + di:1 + di + n] |= mask
    return halo


def nine_point_laplacian(full_values, h):
    v = np.asarray(full_values)
    axis = (v[1:-1, 2:] + v[1:-1, :-2]
            + v[2:, 1:-1] + v[:-2, 1:-1])
    diagonal = (v[2:, 2:] + v[2:, :-2]
                + v[:-2, 2:] + v[:-2, :-2])
    return (4 * axis + diagonal - 20 * v[1:-1, 1:-1]) / (6 * h * h)


def _first_derivative_weights(index, size):
    if 2 <= index <= size - 3:
        nodes = np.arange(index - 2, index + 3)
    elif index < 2:
        nodes = np.arange(5)
    else:
        nodes = np.arange(size - 5, size)
    offsets = nodes - index
    matrix = np.vstack([offsets ** degree for degree in range(5)])
    rhs = np.array([0., 1., 0., 0., 0.])
    return nodes, np.linalg.solve(matrix, rhs)


def grad4(full_values, h):
    """Fourth-order Cartesian gradient, one-sided near the outer boundary."""
    v = np.asarray(full_values)
    ny, nx = v.shape[:2]
    gx = np.empty((ny - 2, nx - 2) + v.shape[2:])
    gy = np.empty_like(gx)
    for i in range(1, nx - 1):
        nodes, weights = _first_derivative_weights(i, nx)
        gx[:, i - 1] = np.tensordot(v[1:-1, nodes], weights, axes=(1, 0)) / h
    for j in range(1, ny - 1):
        nodes, weights = _first_derivative_weights(j, ny)
        gy[j - 1] = np.tensordot(weights, v[nodes, 1:-1], axes=(0, 0)) / h
    return np.stack((gx, gy), axis=-1)


def dst_solve_nine_point(rhs, boundary, h):
    """Solve the nine-point Dirichlet problem on a square interior grid."""
    rhs = np.asarray(rhs)
    m, n = rhs.shape
    if m != n or boundary.shape != (m + 2, n + 2):
        raise ValueError("incompatible square grid and boundary array")
    boundary_only = np.array(boundary, dtype=float, copy=True)
    boundary_only[1:-1, 1:-1] = 0
    adjusted = rhs - nine_point_laplacian(boundary_only, h)
    theta = np.arange(1, m + 1) * np.pi / (m + 1)
    cx = np.cos(theta)[None, :]
    cy = np.cos(theta)[:, None]
    eigenvalues = (8 * (cx + cy) + 4 * cx * cy - 20) / (6 * h * h)
    transformed = dstn(adjusted, type=1, norm="ortho") / eigenvalues
    interior = idstn(transformed, type=1, norm="ortho")
    full = np.array(boundary, dtype=float, copy=True)
    full[1:-1, 1:-1] = interior
    return full


def _boundary_points(xx, yy):
    mask = np.zeros(xx.shape, dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask, np.stack((xx[mask], yy[mask]), axis=-1)


def _fill_direct(mask, values, shape, trailing=()):
    full = np.full(shape + trailing, np.nan)
    full[mask] = values
    return full


def _scalar_norms(error, h):
    return float(np.max(np.abs(error))), float(h * np.sqrt(np.sum(np.abs(error) ** 2)))


def _vector_norms(error, h):
    magnitude = np.linalg.norm(error, axis=-1)
    return float(np.max(magnitude)), float(h * np.sqrt(np.sum(magnitude ** 2)))


def _rates(rows, keys):
    for key in keys:
        previous = None
        for row in rows:
            row[key + "_rate"] = (np.nan if previous is None else
                np.log(previous[key] / row[key]) / np.log(previous["h"] / row["h"]))
            previous = row


def _fitted_slope(rows, key, levels=3):
    """Fit log(error) = alpha log(h) + constant on the last levels."""
    selected = rows[-levels:]
    return float(np.polyfit(np.log([row["h"] for row in selected]),
                            np.log([row[key] for row in selected]), 1)[0])


def validate_cartesian_operators():
    """Return consistency and small-grid DST validation residuals."""
    harmonic = []
    stokes = []
    for m in (24, 48, 96, 192):
        xx, yy, h = grid_coordinates(m)
        v = np.exp(2 * xx) * np.cos(2 * yy)
        lv = nine_point_laplacian(v, h)
        harmonic.append(np.max(np.abs(lv[2:-2, 2:-2])))
        # A nontrivial polynomial Stokes field from the biharmonic stream
        # function psi=(x^2+y^2) Re(z^6); p=-28 Im(z^6).
        radius2 = xx ** 2 + yy ** 2
        harmonic6 = (xx ** 6 - 15 * xx ** 4 * yy ** 2
                     + 15 * xx ** 2 * yy ** 4 - yy ** 6)
        hx = 6 * xx ** 5 - 60 * xx ** 3 * yy ** 2 + 30 * xx * yy ** 4
        hy = -30 * xx ** 4 * yy + 60 * xx ** 2 * yy ** 3 - 6 * yy ** 5
        u = np.stack((2 * yy * harmonic6 + radius2 * hy,
                      -2 * xx * harmonic6 - radius2 * hx), axis=-1)
        p = -28 * (6 * xx ** 5 * yy - 20 * xx ** 3 * yy ** 3
                   + 6 * xx * yy ** 5)
        lu = np.stack([nine_point_laplacian(u[..., k], h) for k in range(2)], axis=-1)
        gp = grad4(p, h)
        stokes.append(np.max(np.linalg.norm((lu - gp)[2:-2, 2:-2], axis=-1)))
    # Compare DST with the explicitly assembled nine-point matrix.
    m = 7
    xx, yy, h = grid_coordinates(m)
    boundary = np.zeros((m + 2, m + 2))
    boundary_values = np.exp(0.3 * xx) * np.cos(0.7 * yy)
    boundary[[0, -1], :] = boundary_values[[0, -1], :]
    boundary[:, [0, -1]] = boundary_values[:, [0, -1]]
    rhs = np.sin(xx[1:-1, 1:-1]) * np.cos(2 * yy[1:-1, 1:-1])
    dst_solution = dst_solve_nine_point(rhs, boundary, h)[1:-1, 1:-1]
    one = np.ones(m - 1)
    off = diags([one, one], [-1, 1], shape=(m, m))
    ident = eye(m)
    matrix = (4 * (kron(ident, off) + kron(off, ident))
              + kron(off, off) - 20 * eye(m * m)) / (6 * h * h)
    boundary_only = boundary.copy()
    boundary_only[1:-1, 1:-1] = 0
    adjusted = rhs - nine_point_laplacian(boundary_only, h)
    direct = spsolve(matrix.tocsr(), adjusted.ravel()).reshape(m, m)
    return {
        "harmonic_errors": harmonic,
        "harmonic_rates": [np.log2(harmonic[i] / harmonic[i + 1]) for i in range(3)],
        "stokes_errors": stokes,
        "dst_residual": float(np.max(np.abs(dst_solution - direct))),
    }


def validate_pressure(output, deltas=(0.16, 0.08, 0.04, 0.02),
                      n_boundary=8192):
    """Direct near-interface validation of p_delta=S_delta phi-D_delta psi."""
    theta = 2 * np.pi * (np.arange(97) + 0.31) / 97
    x0, _, _, n0, _, _ = geometry(theta)
    rows = []
    # The jump identities are a density-differentiation check, distinct from
    # the highly resolved quadrature below.  A moderate Fourier grid avoids
    # amplifying roundoff in the second tangential derivative.
    jump_check = pressure_densities(min(512, n_boundary))
    jump_residual = jump_check.jump_residual
    normal_jump_residual = jump_check.normal_jump_residual
    for order in (3, 5, 7):
        pressure = pressure_densities(n_boundary)
        for delta in deltas:
            for side in (-0.5, 0.0, 0.5):
                b = side * delta
                targets = x0 + b * n0
                values = eval_pressure_regularized(
                    targets, theta, np.full(len(theta), b), delta,
                    n_boundary, pressure, order)
                exact = (2 * targets[:, 0] if side < 0 else
                         (x0[:, 0] if side == 0 else np.zeros(len(theta))))
                rows.append({"order": order, "delta": delta, "b_over_delta": side,
                             "error_inf": float(np.max(np.abs(values - exact)))})
    for side in (-0.5, 0.0, 0.5):
        for order in (3, 5, 7):
            group = [r for r in rows if r["order"] == order and r["b_over_delta"] == side]
            previous = None
            for row in group:
                row["rate_delta"] = (np.nan if previous is None else
                    np.log(previous["error_inf"] / row["error_inf"])
                    / np.log(previous["delta"] / row["delta"]))
                previous = row
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pressure_direct.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows, jump_residual, normal_jump_residual


def all_grid_laplace(m, boundary_ratio=8, kappa=1.0):
    start_total = time.perf_counter()
    xx, yy, h = grid_coordinates(m)
    n_boundary = boundary_ratio * m
    delta = kappa * h ** (5 / 7)
    c_delta = classify_crossing(xx, yy, h, LAPLACE_OFFSETS)
    e_l = stencil_halo(c_delta)
    points = np.stack((xx[e_l], yy[e_l]), axis=-1)
    closest_t, b = closest_points(points)
    f, g = laplace_densities(curve)
    start_bie = time.perf_counter()
    values = eval_laplace_regularized(points, closest_t, b, delta,
                                      n_boundary, f, g)
    boundary_mask, boundary_points = _boundary_points(xx, yy)
    boundary_values = eval_laplace_unregularized(boundary_points, n_boundary, f, g)
    bie_time = time.perf_counter() - start_bie
    integral_full = _fill_direct(e_l, values, xx.shape)
    source = np.zeros((m, m))
    source[c_delta] = nine_point_laplacian(integral_full, h)[c_delta]
    boundary = np.zeros(xx.shape)
    boundary[boundary_mask] = boundary_values
    start_dst = time.perf_counter()
    solved = dst_solve_nine_point(source, boundary, h)
    dst_time = time.perf_counter() - start_dst
    all_values = solved.copy()
    all_values[e_l] = integral_full[e_l]
    inside = level_value(np.stack((xx, yy), axis=-1)) < 0
    exact = np.where(inside, laplace_field(np.stack((xx, yy), axis=-1), "inside")[0],
                     laplace_field(np.stack((xx, yy), axis=-1), "outside")[0])
    error = all_values - exact
    error_inf, error_l2 = _scalar_norms(error, h)
    halo_inf = float(np.max(np.abs(error[e_l])))
    away_inf = float(np.max(np.abs(error[~e_l])))
    min_level = float(np.min(np.abs(level_value(np.stack((xx, yy), axis=-1)))))
    return {
        "m": m, "n_boundary": n_boundary, "h": h, "delta": delta,
        "c_delta": int(np.count_nonzero(c_delta)), "e_l": int(np.count_nonzero(e_l)),
        "error_inf": error_inf, "error_l2": error_l2,
        "halo_error_inf": halo_inf, "away_error_inf": away_inf,
        "min_level": min_level, "bie_time": bie_time, "dst_time": dst_time,
        "total_time": time.perf_counter() - start_total,
        "error_field": error, "classification": c_delta, "halo": e_l,
        "xx": xx, "yy": yy,
    }


def all_grid_stokes(m, boundary_ratio=8, kappa=1.0, mu=1.0):
    start_total = time.perf_counter()
    xx, yy, h = grid_coordinates(m)
    points_full = np.stack((xx, yy), axis=-1)
    n_boundary = boundary_ratio * m
    delta = kappa * h ** (5 / 7)
    c_delta = classify_crossing(xx, yy, h, LAPLACE_OFFSETS)
    c_gradient = classify_crossing(xx, yy, h, GRADIENT_OFFSETS)
    c_union = c_delta | c_gradient
    e_p = stencil_halo(c_union)
    e_u = stencil_halo(c_union)
    pressure = pressure_densities(n_boundary, mu)

    pressure_points = points_full[e_p]
    pressure_t, pressure_b = closest_points(pressure_points)
    start_bie = time.perf_counter()
    pressure_values = eval_pressure_regularized(
        pressure_points, pressure_t, pressure_b, delta, n_boundary, pressure, 7)
    boundary_mask, boundary_points = _boundary_points(xx, yy)
    pressure_boundary = eval_laplace_unregularized(
        boundary_points, n_boundary, pressure.phi, pressure.psi, -1.0)
    pressure_integral = _fill_direct(e_p, pressure_values, xx.shape)
    pressure_source = np.zeros((m, m))
    pressure_source[c_delta] = nine_point_laplacian(pressure_integral, h)[c_delta]
    pressure_boundary_full = np.zeros(xx.shape)
    pressure_boundary_full[boundary_mask] = pressure_boundary
    start_dst = time.perf_counter()
    pressure_solved = dst_solve_nine_point(pressure_source, pressure_boundary_full, h)
    pressure_dst_time = time.perf_counter() - start_dst
    pressure_all = pressure_solved.copy()
    pressure_all[e_p] = pressure_integral[e_p]

    velocity_points = points_full[e_u]
    velocity_t, velocity_b = closest_points(velocity_points)
    velocity_values = eval_stokes_regularized(
        velocity_points, velocity_t, velocity_b, delta, n_boundary, 7, mu)
    velocity_boundary = eval_stokes_unregularized(boundary_points, n_boundary, mu)
    bie_time = time.perf_counter() - start_bie
    velocity_integral = _fill_direct(e_u, velocity_values, xx.shape, (2,))
    velocity_source = np.empty((m, m, 2))
    l_velocity = np.stack([nine_point_laplacian(velocity_integral[..., k], h)
                           for k in range(2)], axis=-1)
    # Differentiate the globally solved pressure so a stencil never mixes the
    # direct and solved representations across the edge of the retained halo.
    pressure_gradient = grad4(pressure_solved, h)
    use_pressure_gradient = ~c_union
    assert not np.any(use_pressure_gradient & (c_delta | c_gradient))
    velocity_source[use_pressure_gradient] = pressure_gradient[use_pressure_gradient] / mu
    velocity_source[c_union] = l_velocity[c_union]
    velocity_boundary_full = np.zeros(xx.shape + (2,))
    velocity_boundary_full[boundary_mask] = velocity_boundary
    start_dst = time.perf_counter()
    velocity_solved = np.stack([
        dst_solve_nine_point(velocity_source[..., k],
                             velocity_boundary_full[..., k], h)
        for k in range(2)], axis=-1)
    velocity_dst_time = time.perf_counter() - start_dst
    velocity_all = velocity_solved.copy()
    velocity_all[e_u] = velocity_integral[e_u]

    inside = level_value(points_full) < 0
    pressure_exact = np.where(inside, 2 * xx, 0.0)
    velocity_exact = np.where(inside[..., None], stokes_field(points_full), 0.0)
    p_error = pressure_all - pressure_exact
    u_error = velocity_all - velocity_exact
    p_inf, p_l2 = _scalar_norms(p_error, h)
    u_inf, u_l2 = _vector_norms(u_error, h)
    exact_gradient = np.zeros_like(pressure_gradient)
    exact_gradient[inside[1:-1, 1:-1], 0] = 2.0
    safe_gradient = ~c_gradient
    gradient_error = np.linalg.norm(pressure_gradient - exact_gradient, axis=-1)
    gradient_inf = float(np.max(gradient_error[safe_gradient]))
    gradient_l2 = float(h * np.sqrt(np.sum(gradient_error[safe_gradient] ** 2)))
    min_level = float(np.min(np.abs(level_value(points_full))))
    return {
        "m": m, "n_boundary": n_boundary, "h": h, "delta": delta,
        "c_delta": int(np.count_nonzero(c_delta)),
        "c_gradient": int(np.count_nonzero(c_gradient)),
        "gradient_only": int(np.count_nonzero(c_gradient & ~c_delta)),
        "e_p": int(np.count_nonzero(e_p)), "e_u": int(np.count_nonzero(e_u)),
        "pressure_error_inf": p_inf, "pressure_error_l2": p_l2,
        "velocity_error_inf": u_inf, "velocity_error_l2": u_l2,
        "gradient_error_inf": gradient_inf, "gradient_error_l2": gradient_l2,
        "min_level": min_level,
        "pressure_boundary_error": float(np.max(np.abs(pressure_boundary))),
        "velocity_boundary_error": float(np.max(np.linalg.norm(velocity_boundary, axis=-1))),
        "jump_residual": pressure.jump_residual,
        "bie_time": bie_time, "dst_time": pressure_dst_time + velocity_dst_time,
        "total_time": time.perf_counter() - start_total,
        "pressure_error_field": p_error, "velocity_error_field": u_error,
        "c_delta_mask": c_delta, "c_gradient_mask": c_gradient,
        "e_p_mask": e_p, "e_u_mask": e_u, "xx": xx, "yy": yy,
    }


def _write_csv(path, rows, excluded=()):
    clean = [{k: v for k, v in row.items() if k not in excluded} for row in rows]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=clean[0].keys())
        writer.writeheader(); writer.writerows(clean)


def _format_rate(value):
    return "--" if not np.isfinite(value) else f"{value:.3f}"


def write_tables(output, laplace_rows, stokes_rows, pressure_rows):
    finest_delta = min(row["delta"] for row in pressure_rows)
    lines = [r"\begin{table}[htbp]\centering", r"\small",
             r"\begin{tabular}{rrrr}", r"\toprule",
             r"order & $b/\delta$ & error at $\delta=0.02$ & rate\\", r"\midrule"]
    for row in pressure_rows:
        if row["delta"] == finest_delta:
            lines.append(f"{int(row['order'])} & {row['b_over_delta']:+.1f} & "
                         f"${row['error_inf']:.3e}$ & {row['rate_delta']:.3f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Direct pressure validation on the three-lobed curve. Here $b<0$ is the interior branch, $b=0$ is the average trace, and $b>0$ is the exterior branch. The rate is measured from $\delta=0.04$ to $0.02$ using 8192 boundary nodes.}",
              r"\label{tab:pressuredirect}", r"\end{table}", "",
              r"\begin{table}[htbp]\centering", r"\small",
             r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$M$ & $N_\Gamma$ & $|\mathcal E_L|$ & $\|e\|_\infty$ & rate & $\|e\|_{2,h_c}$ & rate\\",
             r"\midrule"]
    for row in laplace_rows:
        lines.append(f"{int(row['m'])} & {int(row['n_boundary'])} & {int(row['e_l'])} & "
                     f"${row['error_inf']:.3e}$ & {_format_rate(row['error_inf_rate'])} & "
                     f"${row['error_l2']:.3e}$ & {_format_rate(row['error_l2_rate'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Stencil-aware Laplace all-grid reconstruction. The seventh-order boundary regularizer uses $\delta=h_c^{5/7}$ and $N_\Gamma=8M$.}",
              r"\label{tab:allgridlap}", r"\end{table}", "",
              r"\begin{table}[htbp]\centering", r"\small", r"\begin{tabular}{rrrrrrrr}", r"\toprule",
              r"$M$ & $N_\Gamma$ & $|\mathcal E_p|$ & $|\mathcal E_u|$ & $\|e_p\|_\infty$ & rate & $\|e_u\|_\infty$ & rate\\",
              r"\midrule"]
    for row in stokes_rows:
        lines.append(f"{int(row['m'])} & {int(row['n_boundary'])} & {int(row['e_p'])} & {int(row['e_u'])} & "
                     f"${row['pressure_error_inf']:.3e}$ & {_format_rate(row['pressure_error_inf_rate'])} & "
                     f"${row['velocity_error_inf']:.3e}$ & {_format_rate(row['velocity_error_inf_rate'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Stencil-aware Stokes all-grid reconstruction. Pressure is solved first; velocity uses the pressure gradient only where its complete fourth-order stencil is safe.}",
              r"\label{tab:allgridstokes}", r"\end{table}", "",
              r"\begin{table}[htbp]\centering", r"\small",
              r"\begin{tabular}{rrrrr}", r"\toprule",
              r"$M$ & $\|e_p\|_{2,h_c}$ & rate & $\|e_u\|_{2,h_c}$ & rate\\", r"\midrule"]
    for row in stokes_rows:
        lines.append(f"{int(row['m'])} & ${row['pressure_error_l2']:.3e}$ & "
                     f"{_format_rate(row['pressure_error_l2_rate'])} & "
                     f"${row['velocity_error_l2']:.3e}$ & "
                     f"{_format_rate(row['velocity_error_l2_rate'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Discrete $L^2$ errors for the pressure-first Stokes reconstruction in Table~\ref{tab:allgridstokes}.}",
              r"\label{tab:allgridstokesl2}", r"\end{table}", "",
              r"\begin{table}[htbp]\centering", r"\small",
              r"\begin{tabular}{rrrrr}", r"\toprule",
              r"$M$ & $\|e_{\nabla p}\|_\infty$ & rate & $\|e_{\nabla p}\|_{2,h_c}$ & rate\\", r"\midrule"]
    for row in stokes_rows:
        lines.append(f"{int(row['m'])} & ${row['gradient_error_inf']:.3e}$ & "
                     f"{_format_rate(row['gradient_error_inf_rate'])} & "
                     f"${row['gradient_error_l2']:.3e}$ & "
                     f"{_format_rate(row['gradient_error_l2_rate'])}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Pressure-gradient errors over centers whose complete fourth-order gradient stencil is safe.}",
              r"\label{tab:allgridgradient}", r"\end{table}"]
    (output / "tables.tex").write_text("\n".join(lines) + "\n")


def write_fitted_metrics(output, laplace_rows, stokes_rows,
                         jump_residual, normal_jump_residual):
    metrics = {
        "fitLaplaceMax": _fitted_slope(laplace_rows, "error_inf"),
        "fitLaplaceLTwo": _fitted_slope(laplace_rows, "error_l2"),
        "fitPressureMax": _fitted_slope(stokes_rows, "pressure_error_inf"),
        "fitPressureLTwo": _fitted_slope(stokes_rows, "pressure_error_l2"),
        "fitVelocityMax": _fitted_slope(stokes_rows, "velocity_error_inf"),
        "fitVelocityLTwo": _fitted_slope(stokes_rows, "velocity_error_l2"),
        "fitGradientMax": _fitted_slope(stokes_rows, "gradient_error_inf"),
        "fitGradientLTwo": _fitted_slope(stokes_rows, "gradient_error_l2"),
    }
    lines = [f"\\newcommand{{\\{name}}}{{{value:.2f}}}"
             for name, value in metrics.items()]
    lines += [f"\\newcommand{{\\pressureJumpResidual}}{{{jump_residual:.2e}}}",
              f"\\newcommand{{\\pressureNormalJumpResidual}}{{{normal_jump_residual:.2e}}}"]
    (output / "fitted_metrics.tex").write_text("\n".join(lines) + "\n")
    return metrics


def copy_submission_assets(output, submission_root):
    """Copy generated figures and tables below an output root."""
    figure_dir = submission_root / "figures"
    table_dir = submission_root / "generated_tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
            ("classification.png", "all_grid_classification.png"),
            ("convergence.png", "all_grid_convergence.png"),
            ("error_heatmaps.png", "all_grid_error_heatmaps.png")):
        shutil.copy2(output / source_name, figure_dir / target_name)
    shutil.copy2(output / "tables.tex", table_dir / "all_grid_tables.tex")
    shutil.copy2(output / "fitted_metrics.tex", table_dir / "all_grid_metrics.tex")


def make_figures(output, laplace_rows, stokes_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    representative = next((row for row in stokes_rows if row["m"] == 256),
                          stokes_rows[-1])
    xx, yy = representative["xx"], representative["yy"]
    c_delta = representative["c_delta_mask"]
    c_gradient = representative["c_gradient_mask"]
    fig, ax = plt.subplots(figsize=(6.4, 5.7), constrained_layout=True)
    t = np.linspace(0, 2 * np.pi, 800)
    boundary, _ = curve(t)
    ax.plot(boundary[:, 0], boundary[:, 1], "k-", lw=1.3, label=r"$\Gamma$")
    centers = np.stack((xx[1:-1, 1:-1], yy[1:-1, 1:-1]), axis=-1)
    only_delta = c_delta & ~c_gradient
    both = c_delta & c_gradient
    only_gradient = c_gradient & ~c_delta
    for mask, marker, label, color in [
        (only_delta, "s", r"$\mathcal{C}_\Delta\setminus\mathcal{C}_\nabla$", "tab:blue"),
        (both, ".", r"$\mathcal{C}_\Delta\cap\mathcal{C}_\nabla$", "tab:purple"),
        (only_gradient, "x", r"$\mathcal{C}_\nabla\setminus\mathcal{C}_\Delta$", "tab:red")]:
        ax.scatter(centers[..., 0][mask], centers[..., 1][mask], s=12,
                   marker=marker, color=color, label=label)
    ax.set(aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title=f"Stencil-aware classification, M={representative['m']}")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(output / "classification.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, rows, keys, title in [
        (axes[0], laplace_rows, ("error_inf", "error_l2"), "Laplace"),
        (axes[1], stokes_rows,
         ("pressure_error_inf", "velocity_error_inf", "gradient_error_inf"),
         "Stokes")]:
        hs = np.array([row["h"] for row in rows])
        for key in keys:
            values = np.array([row[key] for row in rows])
            ax.loglog(hs, values, "o-", label=key.replace("_error_inf", ""))
        reference = np.array([row[keys[0]] for row in rows])[-1] * (hs / hs[-1]) ** 4
        ax.loglog(hs, reference, "k--", alpha=0.6, label=r"$h_c^4$")
        ax.set(title=title, xlabel="$h_c$", ylabel="error", xscale="log", yscale="log")
        ax.grid(which="both", alpha=0.25); ax.legend()
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)

    lap = laplace_rows[-1]
    sto = stokes_rows[-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    images = [np.abs(lap["error_field"]), np.abs(sto["pressure_error_field"]),
              np.linalg.norm(sto["velocity_error_field"], axis=-1)]
    titles = ["Laplace error", "Stokes pressure error", "Stokes velocity error"]
    for ax, image, title in zip(axes, images, titles):
        shown = ax.imshow(np.log10(np.maximum(image, 1e-16)), origin="lower",
                          extent=BOX, cmap="magma")
        ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$")
        fig.colorbar(shown, ax=ax, label=r"$\log_{10}|e|$")
    fig.savefig(output / "error_heatmaps.png", dpi=180)
    plt.close(fig)


def run(output=Path("results/all_grid"), resolutions=(32, 64, 128, 256),
        boundary_ratio=8, kappa=1.0, pressure_n=8192,
        submission_root=Path("docs")):
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_cartesian_operators()
    pressure_rows, jump_residual, normal_jump_residual = validate_pressure(
        output, n_boundary=pressure_n)
    laplace_rows = []
    stokes_rows = []
    for m in resolutions:
        print(f"all-grid M={m}", flush=True)
        laplace_rows.append(all_grid_laplace(m, boundary_ratio, kappa))
        stokes_rows.append(all_grid_stokes(m, boundary_ratio, kappa))
    _rates(laplace_rows, ("error_inf", "error_l2"))
    _rates(stokes_rows, ("pressure_error_inf", "pressure_error_l2",
                         "velocity_error_inf", "velocity_error_l2",
                         "gradient_error_inf", "gradient_error_l2"))
    large_keys = ("error_field", "classification", "halo", "xx", "yy")
    _write_csv(output / "laplace.csv", laplace_rows, large_keys)
    large_keys = ("pressure_error_field", "velocity_error_field", "c_delta_mask",
                  "c_gradient_mask", "e_p_mask", "e_u_mask", "xx", "yy")
    _write_csv(output / "stokes.csv", stokes_rows, large_keys)
    write_tables(output, laplace_rows, stokes_rows, pressure_rows)
    fitted = write_fitted_metrics(output, laplace_rows, stokes_rows,
                                  jump_residual, normal_jump_residual)
    make_figures(output, laplace_rows, stokes_rows)
    copy_submission_assets(output, submission_root)
    representative = next((row for row in stokes_rows if row["m"] == 256),
                          stokes_rows[-1])
    report = ["# Stencil-aware all-grid reconstruction", "",
              f"Boundary ratio: N_Gamma/M={boundary_ratio}; delta={kappa:g} h_c^(5/7).",
              f"Pressure jump-density residual: {jump_residual:.3e}.",
              f"Pressure normal-jump residual: {normal_jump_residual:.3e}.",
              f"DST/direct residual: {validation['dst_residual']:.3e}.",
              "Harmonic nine-point rates: " + ", ".join(f"{x:.3f}" for x in validation["harmonic_rates"]),
              "Last-three-level fitted slopes: " + ", ".join(
                  f"{key}={value:.3f}" for key, value in fitted.items()),
              "", f"Representative M={representative['m']} counts:",
              f"C_Delta={representative['c_delta']}, C_nabla={representative['c_gradient']}, "
              f"C_nabla\\C_Delta={representative['gradient_only']}, E_p={representative['e_p']}, E_u={representative['e_u']}.",
              "", "Full numerical data and timings are in laplace.csv, stokes.csv, and pressure_direct.csv."]
    (output / "summary.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return laplace_rows, stokes_rows, pressure_rows, validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/all_grid"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--boundary-ratio", type=int, default=8)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--pressure-n", type=int, default=8192)
    parser.add_argument("--submission-root", type=Path, default=Path("docs"))
    args = parser.parse_args()
    run(args.output, tuple(args.resolutions), args.boundary_ratio, args.kappa,
        args.pressure_n, args.submission_root)
