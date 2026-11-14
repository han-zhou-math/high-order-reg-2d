"""Production local and all-grid runs for Laplace-kite/Stokes-starfish tests."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .bie_evaluators import (PeriodicSamples, eval_laplace_regularized,
                             eval_pressure_regularized,
                             eval_stokes_regularized)
from .config import ORDER_TO_M, ORDER_TO_Q
from .local_drivers import _targets
from .matched_allgrid import run, spacing
from .swapped_low_order import LAPLACE_KITE, STOKES_STARFISH


ROOT = Path("results/swapped_low_order")
LOCAL_LEVELS = (256, 512, 1024, 2048, 4096, 8192, 16384)
GRID_LEVELS = (64, 128, 256, 512, 768, 1024)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_local(levels=LOCAL_LEVELS, orders=(3, 5, 7), filename="errors.csv"):
    rows = []
    for label, problem in (("laplace_kite", LAPLACE_KITE),
                           ("stokes_starfish", STOKES_STARFISH)):
        for n in levels:
            t = 2 * np.pi * np.arange(n) / n
            t2 = 2 * np.pi * np.arange(2 * n) / (2 * n)
            data = (problem.boundary_densities(t) if label == "laplace_kite"
                    else problem.boundary_data(t))
            data2 = (problem.boundary_densities(t2) if label == "laplace_kite"
                     else problem.boundary_data(t2))
            for order in orders:
                h_gamma = 2 * np.pi / n
                delta = h_gamma**ORDER_TO_Q[order]
                lambdas = (-0.5, 0.0, 0.5)
                if label == "laplace_kite" and order == 7:
                    lambdas = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
                for theta, targets, b, lam in _targets(problem.geometry, delta,
                                                        lambdas):
                    side = "average" if lam == 0 else ("inside" if lam < 0 else "outside")
                    if label == "laplace_kite":
                        f, g = data; f2, g2 = data2
                        value = eval_laplace_regularized(
                            problem.geometry, PeriodicSamples(f), PeriodicSamples(g),
                            targets, theta, b, delta, n, order)
                        check = eval_laplace_regularized(
                            problem.geometry, PeriodicSamples(f2), PeriodicSamples(g2),
                            targets, theta, b, delta, 2 * n, order)
                        exact = problem.exact(targets, side)
                        errors = (("laplace", np.abs(value - exact),
                                   np.abs(check - value)),)
                    else:
                        q, f = data; q2, f2 = data2
                        un = eval_stokes_regularized(problem.geometry, q, f, targets,
                                                     theta, b, delta, order)
                        pn = eval_pressure_regularized(problem.geometry, q, f, targets,
                                                       theta, b, delta, order)
                        un2 = eval_stokes_regularized(problem.geometry, q2, f2, targets,
                                                      theta, b, delta, order)
                        pn2 = eval_pressure_regularized(problem.geometry, q2, f2, targets,
                                                        theta, b, delta, order)
                        ue, pe = problem.exact(targets, side)
                        errors = (
                            ("velocity", np.linalg.norm(un - ue, axis=-1),
                             np.linalg.norm(un2 - un, axis=-1)),
                            ("pressure", np.abs(pn - pe), np.abs(pn2 - pn)),
                        )
                    for field, error, difference in errors:
                        rows.append({"problem": label, "field": field,
                            "n_gamma": n, "h_gamma": h_gamma, "order": order,
                            "m": ORDER_TO_M[order], "q": ORDER_TO_Q[order],
                            "delta": delta, "b_over_delta": lam, "side": side,
                            "error_inf": float(np.max(error)),
                            "quadrature_difference_inf": float(np.max(difference))})
            print(f"local {label} N={n}", flush=True)
    _write_csv(ROOT / "local" / filename, rows)


def _grid_parameters(problem, m):
    h_c = (problem.box[1] - problem.box[0]) / (m + 1)
    match = spacing(problem.geometry, h_c)
    h_gamma = 2 * np.pi / match["n_gamma"]
    return match, h_gamma, h_gamma**(5 / 7)


def _maximum_node(fields, key, vector=False):
    value = fields[key]
    magnitude = np.linalg.norm(value, axis=-1) if vector else np.abs(value)
    index = tuple(map(int, np.unravel_index(np.argmax(magnitude), magnitude.shape)))
    return {"index": list(index),
            "point": [float(fields["xx"][index]), float(fields["yy"][index])],
            "error": float(magnitude[index]),
            "in_direct_halo": bool(fields["halo"][index])}


def run_allgrid(levels):
    for label, problem in (("laplace", LAPLACE_KITE),
                           ("stokes", STOKES_STARFISH)):
        output = ROOT / "allgrid" / label
        output.mkdir(parents=True, exist_ok=True)
        for m in levels:
            destination = output / f"M{m}.json"
            if destination.exists():
                print(f"allgrid {label} M={m} already complete", flush=True)
                continue
            match, h_gamma, delta = _grid_parameters(problem, m)
            report, fields = run(problem, m, mode="bie", rule="matched",
                                 delta_override=delta,
                                 n_override=match["n_gamma"])
            report["h_gamma"] = h_gamma
            if label == "laplace":
                report["maximum_node"] = _maximum_node(fields, "error")
            else:
                report["pressure_maximum_node"] = _maximum_node(
                    fields, "pressure_error")
                report["velocity_maximum_node"] = _maximum_node(
                    fields, "velocity_error", vector=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(output / f"fields_M{m}.npz", **fields)
            print(json.dumps({"problem": label, "m": m, "n_gamma": match["n_gamma"],
                              "delta": delta, "seconds": report["total_seconds"],
                              "errors": report["laplace"] if label == "laplace" else
                              {"pressure": report["pressure"],
                               "velocity": report["velocity"],
                               "gradient": report["gradient"]}}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--allgrid", action="store_true")
    parser.add_argument("--levels", nargs="+", type=int, default=GRID_LEVELS)
    args = parser.parse_args(argv)
    if not args.local and not args.allgrid:
        args.local = args.allgrid = True
    if args.local:
        run_local()
    if args.allgrid:
        run_allgrid(args.levels)


if __name__ == "__main__":
    main()
