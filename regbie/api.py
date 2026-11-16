"""Laplace and Stokes layer potentials at arbitrary targets: the user interface.

    from regbie.api import Curve, LayerPotentials
    curve = Curve.from_points(points)          # (N, 2), counterclockwise
    layers = LayerPotentials(curve, n=len(points))
    u = layers.single_layer(f, targets) + layers.double_layer(g, targets)

A Curve is a smooth closed counterclockwise curve parameterized by t in
[0, 2*pi). Build it from an analytic frame (from_function), from x(t) and its
first three derivatives (from_derivatives), or from N points x(2*pi*k/N)
(from_points; the parameterization is the trigonometric interpolant).

LayerPotentials discretizes the curve with n equally spaced parameters
t_k = 2*pi*k/n and evaluates layer potentials of densities given at those
nodes. Targets within `near * delta` of the curve use the regularized kernels
of order p with the local corrections (error O(delta^p) plus a negligible
quadrature error); the others use the trapezoidal rule with the original
kernels. Closest points and signed distances are computed here. By default
delta = 3 times the largest node spacing.

Conventions: G(r) = log(r)/(2*pi); r = x - y with x on the curve and y the
target; outward normal n; densities per unit arclength. For u harmonic inside
the curve, S(-du/dn) + D(u) equals u inside and 0 outside. For a Stokes flow
(u, p) inside with viscosity mu, velocity q = u and traction f = sigma n on the
curve, S f + D q equals u inside and 0 outside, and stokes_pressure(f, q)
equals p inside and 0 outside. On the curve the double layers return the
average of the two one-sided limits.

The C++ library in cpp/ has the same interface (regbie::Curve, regbie::LayerPotentials).
"""
import numpy as np

from .regularized_dlp import smoothing_over_r2
from .regularized_slp import coefficients, regularized_kernel
from .surface_operators import radial_factors
from .verify_stokes_derivation import scaled_moment_n0

_SAMPLES = 4096
_CHUNK = 128


def _modes(n):
    return np.fft.fftfreq(n, d=1 / n)


def _frame(x, dx, ddx, dddx):
    """x, |x'|, tangent, outward normal, curvature and d(kappa)/ds."""
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    speed = np.linalg.norm(dx, axis=-1)
    tau = dx / speed[..., None]
    normal = np.stack((tau[..., 1], -tau[..., 0]), axis=-1)
    kappa = cross(dx, ddx) / speed**3
    kappa_s = cross(dx, dddx) / speed**4 - 3 * cross(dx, ddx) * np.sum(dx * ddx, axis=-1) / speed**6
    return x, speed, tau, normal, kappa, kappa_s


class _Periodic:
    """Trigonometric interpolant of samples at t_k = 2*pi*k/n (values of shape (n, ...))."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.n = len(self.values)
        self.coefficients = np.fft.fft(self.values, axis=0) / self.n
        self.modes = _modes(self.n)

    def __call__(self, t, order=0):
        t = np.asarray(t, dtype=float)
        factor = (1j * self.modes) ** order
        out = np.empty(t.shape + self.values.shape[1:])
        flat, out_flat = t.reshape(-1), out.reshape((-1,) + self.values.shape[1:])
        for start in range(0, len(flat), 256):
            phase = np.exp(1j * flat[start:start + 256, None] * self.modes) * factor
            out_flat[start:start + 256] = np.einsum("mk,k...->m...", phase, self.coefficients).real
        return out

    def derivative_at_nodes(self, order):
        shape = (self.n,) + (1,) * (self.values.ndim - 1)
        return np.fft.ifft(self.coefficients * self.n * ((1j * self.modes) ** order).reshape(shape),
                           axis=0).real


def _spectral_derivative(values, order=1):
    return _Periodic(values).derivative_at_nodes(order)


class Curve:
    """A smooth closed counterclockwise curve; see the module docstring."""

    def __init__(self, geometry, second_derivative):
        self._geometry, self._second = geometry, second_derivative
        self.sample_t = 2 * np.pi * np.arange(_SAMPLES) / _SAMPLES
        self.sample_x = self._geometry(self.sample_t)[0]
        a, b = self.sample_x, np.roll(self.sample_x, -1, axis=0)
        if not np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 0:
            raise ValueError("the curve must be counterclockwise")

    @classmethod
    def from_function(cls, geometry):
        """geometry(t) -> (x, |x'|, tangent, outward normal, kappa, d(kappa)/ds), arrays over t."""
        def second(t, eps=2e-6):
            _, sp, tp = geometry(t + eps)[:3]
            _, sm, tm = geometry(t - eps)[:3]
            return (sp[..., None] * tp - sm[..., None] * tm) / (2 * eps)
        return cls(geometry, second)

    @classmethod
    def from_derivatives(cls, derivatives):
        """derivatives(t) -> (x, x', x'', x'''), each of shape t.shape + (2,)."""
        return cls(lambda t: _frame(*derivatives(np.asarray(t, dtype=float))),
                   lambda t: derivatives(np.asarray(t, dtype=float))[2])

    @classmethod
    def from_points(cls, points):
        """N >= 8 points x(2*pi*k/N) of a smooth closed curve, counterclockwise."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 8:
            raise ValueError("points must have shape (N, 2) with N >= 8")
        series = _Periodic(points)
        nodes = [points] + [series.derivative_at_nodes(j) for j in (1, 2, 3)]
        n = len(points)

        def derivative(t, order):
            t = np.asarray(t, dtype=float)
            s = np.mod(t, 2 * np.pi) * n / (2 * np.pi)
            k = np.rint(s)
            at_node = np.abs(s - k) < 1e-11
            out = np.empty(t.shape + (2,))
            out[at_node] = nodes[order][k[at_node].astype(int) % n]
            if np.any(~at_node):
                out[~at_node] = series(t[~at_node], order)
            return out

        return cls(lambda t: _frame(*(derivative(t, j) for j in range(4))),
                   lambda t: derivative(t, 2))

    def __call__(self, t):
        """(x, |x'|, tangent, outward normal, kappa, d(kappa)/ds) at t."""
        return self._geometry(np.asarray(t, dtype=float))

    def second_derivative(self, t):
        return self._second(np.asarray(t, dtype=float))

    def inside(self, points):
        """Ray-crossing test against a polygon through 4096 points of the curve."""
        points = np.asarray(points, dtype=float)
        a, b = self.sample_x, np.roll(self.sample_x, -1, axis=0)
        flat = points.reshape(-1, 2)
        inside = np.zeros(len(flat), dtype=bool)
        for start in range(0, len(flat), 512):
            p = flat[start:start + 512]
            py = p[:, 1, None]
            crossing = (a[None, :, 1] > py) != (b[None, :, 1] > py)
            xhit = (b[None, :, 0] - a[None, :, 0]) * (py - a[None, :, 1]) \
                / (b[None, :, 1] - a[None, :, 1] + 1e-300) + a[None, :, 0]
            inside[start:start + 512] = np.count_nonzero(crossing & (p[:, 0, None] < xhit), axis=1) % 2 == 1
        return inside.reshape(points.shape[:-1])

    def _nearest_sample(self, points):
        index = np.empty(len(points), dtype=int)
        distance = np.empty(len(points))
        for start in range(0, len(points), 256):
            d2 = np.sum((self.sample_x[None] - points[start:start + 256, None]) ** 2, axis=-1)
            index[start:start + 256] = np.argmin(d2, axis=1)
            distance[start:start + 256] = np.sqrt(np.min(d2, axis=1))
        return index, distance

    def sample_distance(self, points):
        """Distance to the nearest of the 4096 samples (an upper bound on the distance to the curve)."""
        return self._nearest_sample(np.asarray(points, dtype=float).reshape(-1, 2))[1]

    def project(self, points):
        """Closest parameters t and signed distances b (negative inside) of points near the curve."""
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(points) == 0:
            return np.empty(0), np.empty(0)
        t = self.sample_t[self._nearest_sample(points)[0]]
        for _ in range(30):
            x, speed, tau = self(t)[:3]
            dx = speed[:, None] * tau
            r = x - points
            step = np.sum(r * dx, axis=-1) / (np.sum(dx * dx, axis=-1) + np.sum(r * self.second_derivative(t), axis=-1))
            t = np.mod(t - step, 2 * np.pi)
            if np.max(np.abs(step)) < 1e-14:
                break
        x, _, _, normal = self(t)[:4]
        b = np.sum((points - x) * normal, axis=-1)
        b[np.abs(b) <= 5e-13 * np.maximum(1.0, np.linalg.norm(points, axis=-1))] = 0.0
        return t, b


class LayerPotentials:
    """Layer potentials of densities at n equally spaced parameters; see the module docstring."""

    def __init__(self, curve, n, order=7, delta=None, near=6.0, mu=1.0):
        if order not in (3, 5, 7):
            raise ValueError("order must be 3, 5 or 7")
        if n < 8 or near <= 0 or mu <= 0:
            raise ValueError("require n >= 8 and positive near and mu")
        self.curve, self.n, self.order, self.near, self.mu = curve, int(n), order, near, mu
        self.parameters = 2 * np.pi * np.arange(self.n) / self.n
        self.nodes = curve(self.parameters)
        x = self.nodes[0]
        self.spacing = float(np.max(np.linalg.norm(np.roll(x, -1, axis=0) - x, axis=1)))
        self.delta = float(delta) if delta else 3 * self.spacing

    # ------------------------------------------------------------ helpers
    def _check(self, density, name, vector):
        density = np.asarray(density, dtype=float)
        if density.shape != ((self.n, 2) if vector else (self.n,)):
            raise ValueError(f"{name} must have shape {(self.n, 2) if vector else (self.n,)}")
        return density

    def near_targets(self, targets):
        """Indices of the near targets, their closest parameters and signed distances."""
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        index = np.flatnonzero(self.curve.sample_distance(targets) < self.near * self.delta)
        t, b = self.curve.project(targets[index])
        return index, t, b

    def _alpha(self, b, kappa):
        a2 = 1 + b * kappa
        if np.any(a2 <= 0):
            raise ValueError("a near target lies beyond the curvature radius of the curve "
                             "(1 + b*kappa <= 0); increase n or decrease delta")
        return np.sqrt(a2)

    def _far_sum(self, targets, integrand):
        """sum_k integrand(r, k) |x'_k| over the nodes, in chunks of targets; r = x_k - y."""
        out = []
        for start in range(0, len(targets), _CHUNK):
            r = self.nodes[0][None] - targets[start:start + _CHUNK, None]
            out.append(integrand(r))
        return np.concatenate(out) if out else np.empty(0)

    # ------------------------------------------------------------ Laplace
    def single_layer(self, f, targets):
        """S f = int log(r)/(2 pi) f ds."""
        f = self._check(f, "f", False)
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        x, speed = self.nodes[:2]
        out = self._far_sum(targets, lambda r: 0.5 * np.log(np.sum(r * r, axis=-1)) @ (f * speed) / self.n)
        index, _, b = self.near_targets(targets)
        for start in range(0, len(index), _CHUNK):
            sl = slice(start, start + _CHUNK)
            r = np.linalg.norm(targets[index[sl], None] - x[None], axis=2)
            out[index[sl]] = regularized_kernel(r, self.delta, self.order, b[sl, None]) @ (f * speed * 2 * np.pi / self.n)
        return out

    def double_layer(self, g, targets):
        """D g = int n.(x - y)/(2 pi r^2) g ds, with density subtraction near the curve."""
        g = self._check(g, "g", False)
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        x, speed, _, normal = self.nodes[:4]
        out = self._far_sum(targets, lambda r: (np.sum(r * normal[None], axis=-1) / np.sum(r * r, axis=-1))
                            @ (g * speed) / self.n)
        index, t, b = self.near_targets(targets)
        g0 = _Periodic(g)(t)
        chi = np.where(b < 0, 1.0, np.where(b > 0, 0.0, 0.5))
        for start in range(0, len(index), _CHUNK):
            sl = slice(start, start + _CHUNK)
            r = x[None] - targets[index[sl], None]
            factor = smoothing_over_r2(np.sum(r * r, axis=-1), self.delta, self.order, b[sl, None])
            numerator = np.sum(r * normal[None], axis=-1)
            out[index[sl]] = chi[sl] * g0[sl] + np.sum(numerator * factor * (g[None] - g0[sl, None]) * speed / self.n, axis=1)
        return out

    # ------------------------------------------------------------ Stokes
    def stokeslet(self, f, targets):
        """S f = int G f ds, G = (-log(r) I + r r^T/r^2)/(4 pi mu)."""
        f = self._check(f, "f", True)
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        x, speed = self.nodes[:2]
        w = 2 * np.pi * speed / self.n

        def far(r):
            r2 = np.sum(r * r, axis=-1)
            rf = np.sum(r * f[None], axis=-1)
            kernel = (-0.5 * np.log(r2)[..., None] * f[None] + r * (rf / r2)[..., None]) / (4 * np.pi * self.mu)
            return np.einsum("mni,n->mi", kernel, w)

        out = self._far_sum(targets, far)
        index, t, b = self.near_targets(targets)
        if len(index) == 0:
            return out
        _, _, _, n0, kappa, _ = self.curve(t)
        f0 = _Periodic(f)(t)
        alpha = self._alpha(b, kappa)
        scale = np.abs(b) * np.array([scaled_moment_n0(self.order, abs(bb) / self.delta) for bb in b])
        correction = scale[:, None] * n0 * (np.sum(n0 * f0, axis=-1) / alpha)[:, None] / (4 * np.pi * self.mu)
        d2 = self.delta**2
        for start in range(0, len(index), _CHUNK):
            sl = slice(start, start + _CHUNK)
            r = x[None] - targets[index[sl], None]
            r2 = np.sum(r * r, axis=-1)
            a = radial_factors(r2 / d2, self.order, np.abs(b[sl, None]) / self.delta)[0]
            gf = regularized_kernel(np.sqrt(r2), self.delta, self.order, b[sl, None])
            rf = np.sum(r * f[None], axis=-1)
            kernel = (r * (rf * a)[..., None] / (4 * np.pi * d2) - gf[..., None] * f[None] / 2) / self.mu
            out[index[sl]] = np.einsum("mni,n->mi", kernel, w) + correction[sl]
        return out

    def stresslet(self, q, targets):
        """D q = int r (r.q)(r.n)/(pi r^4) ds, r = x - y, with density subtraction near the curve."""
        q = self._check(q, "q", True)
        targets = np.asarray(targets, dtype=float).reshape(-1, 2)
        x, speed, _, normal = self.nodes[:4]
        w = 2 * np.pi * speed / self.n

        def far(r):
            r2 = np.sum(r * r, axis=-1)
            coefficient = np.sum(r * q[None], axis=-1) * np.sum(r * normal[None], axis=-1) / (np.pi * r2 * r2)
            return np.einsum("mni,n->mi", r * coefficient[..., None], w)

        out = self._far_sum(targets, far)
        index, t, b = self.near_targets(targets)
        if len(index) == 0:
            return out
        q_s = _spectral_derivative(q) / speed[:, None]
        q_ss = _spectral_derivative(q_s) / speed[:, None]
        q0, qs0, qss0 = _Periodic(q)(t), _Periodic(q_s)(t), _Periodic(q_ss)(t)
        _, _, tau0, n0, kappa, kappa_s = self.curve(t)
        alpha = self._alpha(b, kappa)
        scale = np.abs(b) * np.array([scaled_moment_n0(self.order, abs(bb) / self.delta) for bb in b])
        dot = lambda u, v: np.sum(u * v, axis=-1)[:, None]
        c2 = (n0 * dot(tau0, qs0) + tau0 * dot(n0, qs0)) / alpha[:, None] ** 3
        c2 -= (b / (2 * alpha**3))[:, None] * n0 * dot(n0, qss0)
        c2 += (b * b * kappa_s / (2 * alpha**5))[:, None] * n0 * dot(n0, qs0)
        chi = np.where(b < 0, 1.0, np.where(b > 0, 0.0, 0.5))
        d2 = self.delta**2
        for start in range(0, len(index), _CHUNK):
            sl = slice(start, start + _CHUNK)
            r = x[None] - targets[index[sl], None]
            ap = radial_factors(np.sum(r * r, axis=-1) / d2, self.order, np.abs(b[sl, None]) / self.delta)[1]
            rq = np.sum(r * (q[None] - q0[sl, None]), axis=-1)
            rn = np.sum(r * normal[None], axis=-1)
            kernel = r * (rq * rn * (-ap) / (np.pi * d2 * d2))[..., None]
            out[index[sl]] = (np.einsum("mni,n->mi", kernel, w) + chi[sl, None] * q0[sl]
                              + scale[sl, None] * c2[sl] / (2 * np.pi))
        return out

    def stokes_pressure(self, f, q, targets):
        """Pressure of S f + D q; pass None for a density that is zero."""
        f = np.zeros((self.n, 2)) if f is None else self._check(f, "f", True)
        q = np.zeros((self.n, 2)) if q is None else self._check(q, "q", True)
        _, speed, tau, normal = self.nodes[:4]
        # p = S phi - D psi with phi = d/ds(f.tau) - 2 mu d/ds(q_s.n), psi = f.n + 2 mu q_s.tau.
        q_s = _spectral_derivative(q) / speed[:, None]
        phi = (_spectral_derivative(np.sum(f * tau, axis=-1))
               - 2 * self.mu * _spectral_derivative(np.sum(q_s * normal, axis=-1))) / speed
        psi = np.sum(f * normal, axis=-1) + 2 * self.mu * np.sum(q_s * tau, axis=-1)
        return self.single_layer(phi, targets) - self.double_layer(psi, targets)
