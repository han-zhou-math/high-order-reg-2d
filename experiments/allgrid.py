"""Problem-agnostic wrapper for the validated stencil-aware all-grid solver."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial import cKDTree

from regbie.all_grid_reconstruction import (GRADIENT_OFFSETS, LAPLACE_OFFSETS,
                                     dst_solve_nine_point, grad4,
                                     nine_point_laplacian, stencil_halo)
from .bie_evaluators import (PeriodicSamples, eval_laplace_regularized,
                             eval_pressure_regularized,
                             eval_stokes_regularized)
from .boundary_data import pressure_densities


@dataclass(frozen=True)
class LaplaceProblem:
    name: str
    geometry: callable
    inside: callable
    boundary_densities: callable
    exact: callable
    box: tuple


@dataclass(frozen=True)
class StokesProblem:
    name: str
    geometry: callable
    inside: callable
    boundary_data: callable
    exact: callable
    exact_gradient: callable
    box: tuple
    mu: float = 1.0


def grid_coordinates(m, box):
    xmin, xmax, ymin, ymax = box
    if abs((xmax - xmin) - (ymax - ymin)) > 1e-13:
        raise ValueError("the DST implementation requires a square box")
    h = (xmax - xmin) / (m + 1)
    x = np.linspace(xmin, xmax, m + 2); y = np.linspace(ymin, ymax, m + 2)
    return (*np.meshgrid(x, y), h)


def closest_points(targets, geometry, inside, samples=4096):
    targets = np.asarray(targets, dtype=float)
    ts = 2 * np.pi * np.arange(samples) / samples
    _, index = cKDTree(geometry(ts)[0]).query(targets)
    t = ts[index]
    for _ in range(12):
        x, speed, tau, _, kappa, _ = geometry(t)
        dx = speed[..., None] * tau
        # x_tt=(speed)_t tau-speed^2 kappa n is avoided by requesting the
        # analytic geometry at two nearby parameters for a stable Newton step.
        eps = 2e-6
        dxp = geometry(t + eps)[1][..., None] * geometry(t + eps)[2]
        dxm = geometry(t - eps)[1][..., None] * geometry(t - eps)[2]
        ddx = (dxp - dxm) / (2 * eps)
        r = x - targets
        step = np.sum(r * dx, axis=-1) / (np.sum(dx * dx, axis=-1)
                                             + np.sum(r * ddx, axis=-1))
        t = np.mod(t - step, 2 * np.pi)
        if np.max(np.abs(step)) < 2e-13:
            break
    x0, _, _, normal, _, _ = geometry(t)
    b = np.sum((targets - x0) * normal, axis=-1)
    # Cartesian nodes can lie on the curve to roundoff.  Their phase label is
    # arbitrary, whereas the closest-point offset should be exactly zero.
    on_surface = np.abs(b) <= 5e-13 * np.maximum(
        1.0, np.linalg.norm(targets, axis=-1))
    b = np.where(on_surface, 0.0, b)
    if np.any(((b < 0) != inside(targets)) & ~on_surface):
        raise RuntimeError("closest-point sign disagrees with phase classifier")
    return t, b


def classify_crossing(xx, yy, h, offsets, inside):
    center = np.stack((xx[1:-1, 1:-1], yy[1:-1, 1:-1]), axis=-1)
    center_inside = inside(center)
    crossing = np.zeros(center_inside.shape, dtype=bool)
    for di, dj in offsets:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            point = center + fraction * np.array((di * h, dj * h))
            crossing |= inside(point) != center_inside
    return crossing


def _boundary_points(xx, yy):
    mask = np.zeros(xx.shape, dtype=bool)
    mask[[0, -1], :] = True; mask[:, [0, -1]] = True
    return mask, np.stack((xx[mask], yy[mask]), axis=-1)


def _unregularized_laplace(geometry, f, g, targets, n, d_sign=1.0):
    t = 2 * np.pi * np.arange(n) / n
    x, speed, _, normal, _, _ = geometry(t); fv, gv = f(t), g(t)
    out = np.empty(len(targets))
    for start in range(0, len(targets), 128):
        stop = min(start + 128, len(targets)); r = x[None] - targets[start:stop, None]
        r2 = np.sum(r * r, axis=-1)
        sl = np.sum(np.log(np.sqrt(r2)) * (fv * speed)[None], axis=1) / n
        dl = np.sum(np.sum(r * normal[None], axis=-1) / r2
                    * (gv * speed)[None], axis=1) / n
        out[start:stop] = sl + d_sign * dl
    return out


def _unregularized_stokes(geometry, q, f, targets, mu=1.0):
    n = len(q); t = 2 * np.pi * np.arange(n) / n
    x, speed, _, normal, _, _ = geometry(t); weights = 2 * np.pi * speed / n
    out = np.empty((len(targets), 2))
    for start in range(0, len(targets), 64):
        stop = min(start + 64, len(targets)); r = x[None] - targets[start:stop, None]
        r2 = np.sum(r * r, axis=-1)
        rf = np.sum(r * f[None], axis=-1)
        sl = (-np.log(np.sqrt(r2))[..., None] * f[None]
              + r * (rf / r2)[..., None]) / (4 * np.pi * mu)
        rq = np.sum(r * q[None], axis=-1); rn = np.sum(r * normal[None], axis=-1)
        dl = r * (rq * rn / (np.pi * r2**2))[..., None]
        out[start:stop] = np.einsum("mni,n->mi", sl + dl, weights)
    return out


def solve_laplace(problem, m, n_gamma, delta, order=7, oversample_factor=2):
    start = time.perf_counter(); xx, yy, h = grid_coordinates(m, problem.box)
    points = np.stack((xx, yy), axis=-1)
    c_delta = classify_crossing(xx, yy, h, LAPLACE_OFFSETS, problem.inside)
    e_l = stencil_halo(c_delta)
    t = 2 * np.pi * np.arange(n_gamma) / n_gamma
    f, g = problem.boundary_densities(t); fi, gi = PeriodicSamples(f), PeriodicSamples(g)
    closest_t, b = closest_points(points[e_l], problem.geometry, problem.inside)
    bie_start = time.perf_counter()
    direct = eval_laplace_regularized(problem.geometry, fi, gi, points[e_l],
                                      closest_t, b, delta, n_gamma, order)
    n_check = oversample_factor * n_gamma
    t_check = 2 * np.pi * np.arange(n_check) / n_check
    f_check, g_check = problem.boundary_densities(t_check)
    direct_check = eval_laplace_regularized(
        problem.geometry, PeriodicSamples(f_check), PeriodicSamples(g_check),
        points[e_l], closest_t, b, delta, n_check, order)
    oversampled_error = float(np.max(np.abs(direct_check - direct)))
    boundary_mask, boundary_points = _boundary_points(xx, yy)
    boundary_values = _unregularized_laplace(problem.geometry, fi, gi,
                                             boundary_points, n_gamma)
    bie_seconds = time.perf_counter() - bie_start
    direct_full = np.full(xx.shape, np.nan); direct_full[e_l] = direct
    rhs = np.zeros((m, m)); rhs[c_delta] = nine_point_laplacian(direct_full, h)[c_delta]
    boundary = np.zeros(xx.shape); boundary[boundary_mask] = boundary_values
    dst_start = time.perf_counter(); solved = dst_solve_nine_point(rhs, boundary, h)
    dst_seconds = time.perf_counter() - dst_start
    solved[e_l] = direct
    exact = problem.exact(points); error = solved - exact
    return {"values": solved, "error": error, "xx": xx, "yy": yy,
            "c_delta": c_delta, "halo": e_l, "h": h,
            "error_inf": float(np.max(np.abs(error))),
            "error_l2": float(h * np.sqrt(np.sum(error**2))),
            "oversampled_error_inf": oversampled_error,
            "direct_targets": int(np.count_nonzero(e_l)),
            "bie_seconds": bie_seconds, "dst_seconds": dst_seconds,
            "total_seconds": time.perf_counter() - start}


def solve_stokes(problem, m, n_gamma, delta, order=7, oversample_factor=2):
    start = time.perf_counter(); xx, yy, h = grid_coordinates(m, problem.box)
    points = np.stack((xx, yy), axis=-1)
    c_delta = classify_crossing(xx, yy, h, LAPLACE_OFFSETS, problem.inside)
    c_gradient = classify_crossing(xx, yy, h, GRADIENT_OFFSETS, problem.inside)
    c_union = c_delta | c_gradient; halo = stencil_halo(c_union)
    t = 2 * np.pi * np.arange(n_gamma) / n_gamma
    q, f = problem.boundary_data(t, problem.mu)
    pdata = pressure_densities(problem.geometry, q, f, problem.mu)
    phi, psi = PeriodicSamples(pdata.phi), PeriodicSamples(pdata.psi)
    closest_t, b = closest_points(points[halo], problem.geometry, problem.inside)
    bie_start = time.perf_counter()
    p_direct = eval_pressure_regularized(problem.geometry, q, f, points[halo],
                                         closest_t, b, delta, order, problem.mu)
    n_check = oversample_factor * n_gamma
    t_check = 2 * np.pi * np.arange(n_check) / n_check
    q_check, f_check = problem.boundary_data(t_check, problem.mu)
    p_direct_check = eval_pressure_regularized(
        problem.geometry, q_check, f_check, points[halo], closest_t, b,
        delta, order, problem.mu)
    boundary_mask, boundary_points = _boundary_points(xx, yy)
    p_boundary = _unregularized_laplace(problem.geometry, phi, psi,
                                       boundary_points, n_gamma, -1.0)
    p_full = np.full(xx.shape, np.nan); p_full[halo] = p_direct
    p_rhs = np.zeros((m, m)); p_rhs[c_delta] = nine_point_laplacian(p_full, h)[c_delta]
    p_bc = np.zeros(xx.shape); p_bc[boundary_mask] = p_boundary
    dst_start = time.perf_counter(); p_solved = dst_solve_nine_point(p_rhs, p_bc, h)
    p_dst = time.perf_counter() - dst_start
    p_all = p_solved.copy(); p_all[halo] = p_direct
    u_direct = eval_stokes_regularized(problem.geometry, q, f, points[halo],
                                       closest_t, b, delta, order, problem.mu)
    u_direct_check = eval_stokes_regularized(
        problem.geometry, q_check, f_check, points[halo], closest_t, b,
        delta, order, problem.mu)
    p_check = np.max(np.abs(p_direct_check - p_direct))
    u_check = np.max(np.linalg.norm(u_direct_check - u_direct, axis=-1))
    u_boundary = _unregularized_stokes(problem.geometry, q, f, boundary_points,
                                       problem.mu)
    bie_seconds = time.perf_counter() - bie_start
    u_full = np.full(xx.shape + (2,), np.nan); u_full[halo] = u_direct
    l_u = np.stack([nine_point_laplacian(u_full[..., k], h) for k in range(2)], -1)
    p_gradient = grad4(p_solved, h); u_rhs = np.empty((m, m, 2))
    use_gradient = ~c_union
    assert not np.any(use_gradient & c_gradient)
    u_rhs[use_gradient] = p_gradient[use_gradient] / problem.mu
    u_rhs[c_union] = l_u[c_union]
    u_bc = np.zeros(xx.shape + (2,)); u_bc[boundary_mask] = u_boundary
    dst_start = time.perf_counter()
    u_solved = np.stack([dst_solve_nine_point(u_rhs[..., k], u_bc[..., k], h)
                         for k in range(2)], axis=-1)
    u_dst = time.perf_counter() - dst_start; u_solved[halo] = u_direct
    u_exact, p_exact = problem.exact(points); pe = p_all - p_exact; ue = u_solved - u_exact
    safe = ~c_gradient; grad_exact = problem.exact_gradient(points[1:-1, 1:-1])
    ge = np.linalg.norm(p_gradient - grad_exact, axis=-1)
    return {"pressure": p_all, "velocity": u_solved, "pressure_error": pe,
            "velocity_error": ue, "xx": xx, "yy": yy, "h": h,
            "c_delta": c_delta, "c_gradient": c_gradient, "halo": halo,
            "pressure_error_inf": float(np.max(np.abs(pe))),
            "pressure_error_l2": float(h * np.sqrt(np.sum(pe**2))),
            "velocity_error_inf": float(np.max(np.linalg.norm(ue, axis=-1))),
            "velocity_error_l2": float(h * np.sqrt(np.sum(ue**2))),
            "gradient_error_inf": float(np.max(ge[safe])),
            "gradient_error_l2": float(h * np.sqrt(np.sum(ge[safe]**2))),
            "oversampled_error_inf": float(max(p_check, u_check)),
            "pressure_oversampled_error_inf": float(p_check),
            "velocity_oversampled_error_inf": float(u_check),
            "direct_targets": int(np.count_nonzero(halo)),
            "bie_seconds": bie_seconds, "dst_seconds": p_dst + u_dst,
            "total_seconds": time.perf_counter() - start}
