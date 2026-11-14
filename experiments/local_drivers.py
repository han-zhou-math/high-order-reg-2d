"""Production local-layer drivers. They run only when called explicitly."""
import numpy as np

from regbie.regularized_dlp import double_layer
from regbie.regularized_slp import single_layer

from .bie_evaluators import (PeriodicSamples, curve_adapter,
                             eval_laplace_regularized,
                             eval_pressure_regularized,
                             eval_stokes_regularized)
from .boundary_data import pressure_densities
from .config import (BOUNDARY_LEVELS, KAPPA, ORDER_TO_M, ORDER_TO_Q, ORDERS,
                     OVERSAMPLE_FACTOR, ROBUSTNESS_LAMBDAS, TARGET_COUNT)
from .geometries import (circle_geometry, laplace_starfish_geometry,
                         stokes_kite_geometry)
from .manufactured_laplace import (circle_density, exact_circle_double,
                                   exact_circle_single,
                                   exact_piecewise_laplace,
                                   laplace_boundary_densities)
from .manufactured_stokes import (exact_piecewise_stokes, grad_pressure, pressure,
                                  stokes_boundary_data)
from .result_io import LOCAL_FIELDS, write_metadata, write_rows
from .validation import validate_all


def _targets(geometry, delta, signed_lambdas, count=TARGET_COUNT):
    theta = 2 * np.pi * (np.arange(count) + 0.37) / count
    x0, _, _, normal, _, _ = geometry(theta)
    for lam in signed_lambdas:
        b = lam * delta
        yield theta, x0 + b * normal, np.full(count, b), lam


def run_circle_layers(output):
    validate_all()
    rows = []
    curve = curve_adapter(circle_geometry)
    for n in BOUNDARY_LEVELS:
        h = 2 * np.pi / n
        for order in ORDERS:
            q = ORDER_TO_Q[order]; delta = KAPPA * h**q
            for theta, targets, b, lam in _targets(circle_geometry, delta,
                                                    (-0.5, 0.0, 0.5)):
                radius = np.linalg.norm(targets, axis=-1)
                angle = np.arctan2(targets[:, 1], targets[:, 0])
                sn = single_layer(curve, circle_density, targets, delta, n, order, b)
                dn = double_layer(curve, circle_density, targets, delta, n, order,
                                  closest_t=theta, b=b, surface=(lam == 0))
                sn_hi = single_layer(curve, circle_density, targets, delta,
                                     OVERSAMPLE_FACTOR * n, order, b)
                dn_hi = double_layer(curve, circle_density, targets, delta,
                                     OVERSAMPLE_FACTOR * n, order,
                                     closest_t=theta, b=b, surface=(lam == 0))
                side = "average" if lam == 0 else ("inside" if lam < 0 else "outside")
                se = np.max(np.abs(sn - exact_circle_single(radius, angle)))
                de = np.max(np.abs(dn - exact_circle_double(radius, angle, side)))
                for layer, error, check in (
                        ("circle_single", se, np.max(np.abs(sn_hi - sn))),
                        ("circle_double", de, np.max(np.abs(dn_hi - dn)))):
                    rows.append(dict(geometry="unit_circle", solution=layer,
                        n_gamma=n, h_gamma=h, order=order, m=ORDER_TO_M[order], q=q,
                        delta=delta, kappa=KAPPA, **{"lambda": lam}, side=side,
                        error_inf=error, error_l2="", oversampled_error_inf=check))
    write_rows(output / "errors.csv", rows, LOCAL_FIELDS)
    write_metadata(output / "metadata.json", experiment="circle_layers",
                   configuration={"levels": BOUNDARY_LEVELS, "orders": ORDERS,
                                  "q_by_order": ORDER_TO_Q, "kappa": KAPPA,
                                  "target_count": TARGET_COUNT,
                                  "signed_lambdas": (-0.5, 0.0, 0.5),
                                  "oversample_factor": OVERSAMPLE_FACTOR})


def run_laplace_starfish(output):
    validate_all(); rows = []
    for n in BOUNDARY_LEVELS:
        h = 2 * np.pi / n
        f, g = laplace_boundary_densities(2 * np.pi * np.arange(n) / n)
        fi, gi = PeriodicSamples(f), PeriodicSamples(g)
        n_hi = OVERSAMPLE_FACTOR * n
        f_hi, g_hi = laplace_boundary_densities(2 * np.pi * np.arange(n_hi) / n_hi)
        fi_hi, gi_hi = PeriodicSamples(f_hi), PeriodicSamples(g_hi)
        for order in ORDERS:
            delta = KAPPA * h**ORDER_TO_Q[order]
            lambdas = (-0.5, 0.0, 0.5)
            if order == 7:
                lambdas = tuple(sorted(set(lambdas + tuple(
                    sign * lam for lam in ROBUSTNESS_LAMBDAS for sign in (-1, 1)))))
            for theta, targets, b, lam in _targets(laplace_starfish_geometry,
                                                    delta, lambdas):
                numerical = eval_laplace_regularized(
                    laplace_starfish_geometry, fi, gi, targets, theta, b, delta, n, order)
                oversampled = eval_laplace_regularized(
                    laplace_starfish_geometry, fi_hi, gi_hi, targets, theta, b,
                    delta, n_hi, order)
                side = "average" if lam == 0 else ("inside" if lam < 0 else "outside")
                exact = exact_piecewise_laplace(targets, side)
                rows.append(dict(geometry="five_lobed_starfish",
                    solution="combined_laplace", n_gamma=n, h_gamma=h, order=order,
                    m=ORDER_TO_M[order], q=ORDER_TO_Q[order], delta=delta,
                    kappa=KAPPA, **{"lambda": abs(lam)}, side=side,
                    error_inf=np.max(np.abs(numerical - exact)), error_l2="",
                    oversampled_error_inf=np.max(np.abs(oversampled - numerical))))
    write_rows(output / "errors.csv", rows, LOCAL_FIELDS)
    write_metadata(output / "metadata.json", experiment="laplace_starfish",
                   configuration={"levels": BOUNDARY_LEVELS, "orders": ORDERS,
                                  "q_by_order": ORDER_TO_Q, "kappa": KAPPA,
                                  "target_count": TARGET_COUNT,
                                  "robustness_lambdas": ROBUSTNESS_LAMBDAS,
                                  "oversample_factor": OVERSAMPLE_FACTOR})


def run_stokes_kite(output, mu=1.0):
    validate_all(); rows = []
    for n in BOUNDARY_LEVELS:
        h = 2 * np.pi / n; t = 2 * np.pi * np.arange(n) / n
        q_values, f_values = stokes_boundary_data(t, mu)
        n_hi = OVERSAMPLE_FACTOR * n
        q_hi, f_hi = stokes_boundary_data(2 * np.pi * np.arange(n_hi) / n_hi, mu)
        for order in ORDERS:
            delta = KAPPA * h**ORDER_TO_Q[order]
            for theta, targets, b, lam in _targets(stokes_kite_geometry, delta,
                                                    (-0.5, 0.0, 0.5)):
                side = "average" if lam == 0 else ("inside" if lam < 0 else "outside")
                un = eval_stokes_regularized(stokes_kite_geometry, q_values, f_values,
                                             targets, theta, b, delta, order, mu)
                pn = eval_pressure_regularized(stokes_kite_geometry, q_values, f_values,
                                               targets, theta, b, delta, order, mu)
                un_hi = eval_stokes_regularized(stokes_kite_geometry, q_hi, f_hi,
                                                targets, theta, b, delta, order, mu)
                pn_hi = eval_pressure_regularized(stokes_kite_geometry, q_hi, f_hi,
                                                  targets, theta, b, delta, order, mu)
                ue, pe = exact_piecewise_stokes(targets, side, mu)
                for name, error, check in (("combined_stokes_velocity",
                         np.max(np.linalg.norm(un - ue, axis=-1)),
                         np.max(np.linalg.norm(un_hi - un, axis=-1))),
                        ("combined_stokes_pressure", np.max(np.abs(pn - pe)),
                         np.max(np.abs(pn_hi - pn)))):
                    rows.append(dict(geometry="nonconvex_kite", solution=name,
                        n_gamma=n, h_gamma=h, order=order, m=ORDER_TO_M[order],
                        q=ORDER_TO_Q[order], delta=delta, kappa=KAPPA,
                        **{"lambda": lam}, side=side, error_inf=error, error_l2="",
                        oversampled_error_inf=check))
    write_rows(output / "errors.csv", rows, LOCAL_FIELDS)
    check_n = 1024
    check_t = 2 * np.pi * np.arange(check_n) / check_n
    x, _, _, normal, _, _ = stokes_kite_geometry(check_t)
    q_check, f_check = stokes_boundary_data(check_t, mu)
    pdata = pressure_densities(stokes_kite_geometry, q_check, f_check, mu)
    pressure_jump_residual = float(np.max(np.abs(pdata.psi + pressure(x, mu))))
    normal_jump_residual = float(np.max(np.abs(
        pdata.phi + np.sum(grad_pressure(x, mu) * normal, axis=-1))))
    write_metadata(output / "metadata.json", experiment="stokes_kite",
                   configuration={"levels": BOUNDARY_LEVELS, "orders": ORDERS,
                                  "q_by_order": ORDER_TO_Q, "kappa": KAPPA,
                                  "target_count": TARGET_COUNT,
                                  "signed_lambdas": (-0.5, 0.0, 0.5),
                                  "oversample_factor": OVERSAMPLE_FACTOR,
                                  "mu": mu, "jump_check_n_gamma": check_n,
                                  "pressure_jump_residual": pressure_jump_residual,
                                  "normal_jump_residual": normal_jump_residual})
