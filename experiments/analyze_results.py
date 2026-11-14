"""Compute documented local orders and least-squares rates from saved CSV data."""
import json
from pathlib import Path
import numpy as np

from .config import RESULT_ROOT
from .result_io import read_rows


def _numeric(rows):
    for row in rows:
        for key, value in tuple(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _fit(points, fit_levels):
    selected = [point for point in points if int(point[0]) in fit_levels]
    resolution = [int(point[0]) for point in selected]
    h = np.asarray([point[1] for point in selected])
    error = np.asarray([point[2] for point in selected])
    local = [float(np.log(error[j] / error[j + 1])
                   / np.log(h[j] / h[j + 1])) for j in range(len(error) - 1)]
    slope, intercept = np.polyfit(np.log(h), np.log(error), 1)
    return {"fit_levels": resolution, "fit_slope": float(slope),
            "fit_intercept": float(intercept), "h": h.tolist(),
            "error": error.tolist(), "local_orders_within_fit": local}


def _local_summary(directory, specifications):
    rows = _numeric(read_rows(directory / "errors.csv")); report = {}
    for label, solution, order, lambdas, fit_levels in specifications:
        selected = [row for row in rows if row["solution"] == solution
                    and int(row["order"]) == order
                    and (lambdas is None or row["lambda"] in lambdas)]
        grouped = {}
        for row in selected:
            key = int(row["n_gamma"])
            grouped[key] = max(grouped.get(key, 0.0), row["error_inf"])
        points = [(n, 2 * np.pi / n, error) for n, error in sorted(grouped.items())]
        item = _fit(points, fit_levels)
        item["all_levels"] = [point[0] for point in points]
        item["all_errors"] = [point[2] for point in points]
        item["all_local_orders"] = [float(np.log(points[j][2] / points[j + 1][2])
                                          / np.log(points[j][1] / points[j + 1][1]))
                                     for j in range(len(points) - 1)]
        report[label] = item
    (directory / "rates.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _allgrid_summary(directory, fields):
    rows = sorted(_numeric(read_rows(directory / "errors.csv")),
                  key=lambda row: int(row["m_grid"])); report = {}
    for field in fields:
        points = [(int(row["m_grid"]), row["h_c"], row[field]) for row in rows]
        item = _fit(points, {512, 768, 1024})
        item["all_levels"] = [point[0] for point in points]
        item["all_errors"] = [point[2] for point in points]
        item["all_local_orders"] = [float(np.log(points[j][2] / points[j + 1][2])
                                          / np.log(points[j][1] / points[j + 1][1]))
                                     for j in range(len(points) - 1)]
        report[field] = item
    (directory / "rates.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    circle_specs = []
    for solution in ("circle_single", "circle_double"):
        for order in (3, 5, 7):
            levels = {512, 1024, 2048, 4096} if order == 7 else {2048, 4096, 8192, 16384}
            circle_specs.append((f"{solution}_p{order}", solution, order, None, levels))
    starfish_specs = [(f"combined_laplace_p{order}", "combined_laplace", order,
                       {0.0, 0.5}, {4096, 8192, 16384}) for order in (3, 5, 7)]
    for lam in (0.0, 0.5, 1.0, 2.0):
        starfish_specs.append((f"robustness_p7_lambda{lam:g}", "combined_laplace",
                               7, {lam}, {4096, 8192, 16384}))
    stokes_specs = [(f"{solution}_p{order}", solution, order, None,
                     {4096, 8192, 16384})
                    for solution in ("combined_stokes_velocity",
                                     "combined_stokes_pressure")
                    for order in (3, 5, 7)]
    reports = {
        "circle_layers": _local_summary(RESULT_ROOT / "circle_layers", circle_specs),
        "laplace_starfish": _local_summary(RESULT_ROOT / "laplace_starfish", starfish_specs),
        "stokes_kite": _local_summary(RESULT_ROOT / "stokes_kite", stokes_specs),
        "allgrid_laplace": _allgrid_summary(RESULT_ROOT / "allgrid_laplace",
                                             ("error_inf", "error_l2")),
        "allgrid_stokes": _allgrid_summary(
            RESULT_ROOT / "allgrid_stokes",
            ("pressure_error_inf", "pressure_error_l2", "velocity_error_inf",
             "velocity_error_l2", "gradient_error_inf", "gradient_error_l2")),
    }
    print(json.dumps({name: {key: round(value["fit_slope"], 4)
                                  for key, value in report.items()}
                      for name, report in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
