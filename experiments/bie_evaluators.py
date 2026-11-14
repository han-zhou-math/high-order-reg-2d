"""Thin generic wrappers around the repository's existing regularized kernels."""
import numpy as np

from regbie.regularized_dlp import double_layer
from regbie.regularized_slp import regularized_kernel, single_layer
from regbie.surface_operators import radial_factors
from regbie.verify_stokes_derivation import scaled_moment_n0

from .boundary_data import pressure_densities


class PeriodicSamples:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.n = len(values)
        self.coefficients = np.fft.fft(self.values, axis=0) / self.n
        self.modes = np.fft.fftfreq(self.n, d=1 / self.n)

    def __call__(self, t, chunk_size=256):
        t = np.asarray(t, dtype=float)
        if t.ndim == 1 and len(t) == self.n:
            grid = 2 * np.pi * np.arange(self.n) / self.n
            if np.array_equal(t, grid) or np.allclose(t, grid, rtol=0, atol=2e-14):
                return self.values.copy()
        flat = t.reshape(-1)
        result = np.empty((len(flat),) + self.values.shape[1:], dtype=float)
        for start in range(0, len(flat), chunk_size):
            stop = min(start + chunk_size, len(flat))
            phase = np.exp(1j * flat[start:stop, None] * self.modes)
            result[start:stop] = np.einsum(
                "mk,k...->m...", phase, self.coefficients).real
        return result.reshape(t.shape + self.values.shape[1:])


def curve_adapter(geometry):
    def curve(t):
        x, speed, tau = geometry(t)[:3]
        return x, speed[..., None] * tau
    return curve


def eval_laplace_regularized(geometry, f, g, targets, closest_t, b, delta,
                             n_gamma, order):
    curve = curve_adapter(geometry)
    sl = single_layer(curve, f, targets, delta, n_gamma, order, b)
    dl = double_layer(curve, g, targets, delta, n_gamma, order,
                      closest_t=closest_t, b=b)
    return sl + dl


def eval_pressure_regularized(geometry, q_values, f_values, targets, closest_t,
                              b, delta, order, mu=1.0):
    data = pressure_densities(geometry, q_values, f_values, mu)
    curve = curve_adapter(geometry)
    phi, psi = PeriodicSamples(data.phi), PeriodicSamples(data.psi)
    sl = single_layer(curve, phi, targets, delta, len(q_values), order, b)
    dl = double_layer(curve, psi, targets, delta, len(q_values), order,
                      closest_t=closest_t, b=b)
    return sl - dl


def eval_stokes_regularized(geometry, q_values, f_values, targets, closest_t,
                            b, delta, order, mu=1.0, chunk_size=64):
    """Corrected Stokes S f + D q for generic smooth periodic data."""
    targets = np.asarray(targets); b = np.asarray(b); closest_t = np.asarray(closest_t)
    n_gamma = len(q_values)
    t = 2 * np.pi * np.arange(n_gamma) / n_gamma
    x, speed, _, normal, _, _ = geometry(t)
    pressure_data = pressure_densities(geometry, q_values, f_values, mu)
    q_interp, f_interp = PeriodicSamples(q_values), PeriodicSamples(f_values)
    qs_interp, qss_interp = (PeriodicSamples(pressure_data.q_s),
                             PeriodicSamples(pressure_data.q_ss))
    x0, _, tau0, n0, kappa, kappa_s = geometry(closest_t)
    q0, f0 = q_interp(closest_t), f_interp(closest_t)
    q_s, q_ss = qs_interp(closest_t), qss_interp(closest_t)
    weights = 2 * np.pi * speed / n_gamma
    answer = np.empty((len(targets), 2))
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets)); slc = slice(start, stop)
        r = x[None] - targets[slc, None]
        r2 = np.sum(r * r, axis=-1); xi = r2 / delta**2
        a, ap, _ = radial_factors(xi, order, np.abs(b[slc, None]) / delta)
        gf = regularized_kernel(np.sqrt(r2), delta, order, b[slc, None])
        rf = np.sum(r * f_values[None], axis=-1)
        sl = (r * rf[..., None] * a[..., None] / (4 * np.pi * delta**2)
              - gf[..., None] * f_values[None] / 2) / mu
        rq = np.sum(r * (q_values[None] - q0[slc, None]), axis=-1)
        rn = np.sum(r * normal[None], axis=-1)
        dl = r * (rq * rn * (-ap) / (np.pi * delta**4))[..., None]
        value = np.einsum("mni,n->mi", sl + dl, weights)
        chi = np.where(b[slc] < 0, 1.0, np.where(b[slc] > 0, 0.0, 0.5))
        value += chi[:, None] * q0[slc]
        alpha = np.sqrt(1 + b[slc] * kappa[slc])
        scale = np.abs(b[slc]) * np.array([
            scaled_moment_n0(order, abs(value_b) / delta) for value_b in b[slc]])
        b0 = n0[slc] * np.sum(n0[slc] * f0[slc], axis=-1)[:, None] / alpha[:, None]
        sym = (n0[slc] * np.sum(tau0[slc] * q_s[slc], axis=-1)[:, None]
               + tau0[slc] * np.sum(n0[slc] * q_s[slc], axis=-1)[:, None])
        c2 = sym / alpha[:, None]**3
        c2 -= (b[slc] / (2 * alpha**3))[:, None] * n0[slc] * np.sum(
            n0[slc] * q_ss[slc], axis=-1)[:, None]
        c2 += (b[slc]**2 * kappa_s[slc] / (2 * alpha**5))[:, None] * n0[slc] * np.sum(
            n0[slc] * q_s[slc], axis=-1)[:, None]
        answer[slc] = (value + scale[:, None] * b0 / (4 * np.pi * mu)
                       + scale[:, None] * c2 / (2 * np.pi))
    return answer
