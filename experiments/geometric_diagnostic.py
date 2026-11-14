"""Audit geometry of the current all-grid data; reproduce coarse direct errors."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from experiments.allgrid import closest_points, grid_coordinates
from experiments.bie_evaluators import eval_stokes_regularized, PeriodicSamples
from experiments.boundary_data import pressure_densities
from experiments.matched_allgrid import spacing
from experiments.swapped_low_order import STOKES_STARFISH
from regbie.verify_stokes_derivation import scaled_moment_n0

ROOT = Path("results/paper_revision")
problem = STOKES_STARFISH
geometry = problem.geometry
levels = (64, 128, 256, 512, 768, 1024)
t_dense = 2 * np.pi * np.arange(65536) / 65536
kappa_inf = float(np.max(np.abs(geometry(t_dense)[4])))
rows = []
fresh = {}
for m in levels:
    original = Path(f"results/swapped_low_order/allgrid/stokes/fields_M{m}.npz")
    base_report = json.loads(original.with_name(f"M{m}.json").read_text())
    data = np.load(original)
    halo = data["halo"]
    targets = np.stack((data["xx"][halo], data["yy"][halo]), axis=-1)
    theta, b = closest_points(targets, geometry, problem.inside)
    kappa = geometry(theta)[4]
    alpha2 = 1 + b * kappa
    ia = int(np.argmin(alpha2))
    for rule in ("separated", "2hc"):
        report_path = (original.with_name(f"M{m}.json") if rule == "separated"
                       else Path(f"results/fixed_width/allgrid/starfish_stokes_k2_M{m}.json"))
        field_path = (original if rule == "separated" else report_path.with_suffix(".npz"))
        if rule == "2hc" and not field_path.exists():
            report_path = Path(f"results/paper_revision/allgrid_practical/stokes/M{m}.json")
            field_path = report_path.with_name(f"fields_M{m}.npz")
        delta = ((2 * np.pi / base_report["n_gamma"]) ** (5 / 7)
                 if rule == "separated" else 2 * base_report["h_c"])
        row = dict(m_grid=m, width_rule=rule, h_c=base_report["h_c"],
                   n_gamma=base_report["n_gamma"], delta=delta,
                   curvature_inf=kappa_inf, delta_curvature_inf=delta * kappa_inf,
                   min_alpha_squared=float(alpha2[ia]),
                   min_alpha_point=targets[ia].tolist(), min_alpha_b=float(b[ia]),
                   direct_targets=len(targets), max_abs_b_over_delta=float(np.max(abs(b)) / delta))
        if field_path.exists():
            fields = np.load(field_path)
            magnitude = np.linalg.norm(fields["velocity_error"], axis=-1)
            ij = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            y = np.array([fields["xx"][ij], fields["yy"][ij]])
            th, bv = closest_points(y[None], geometry, problem.inside)
            a2 = float(1 + bv[0] * geometry(th)[4][0])
            row.update(velocity_error_inf=float(magnitude[ij]),
                       velocity_max_point=y.tolist(), velocity_max_index=list(map(int, ij)),
                       velocity_max_is_direct=bool(fields["halo"][ij]),
                       velocity_max_alpha_squared=a2, velocity_max_b=float(bv[0]),
                       velocity_max_alpha_minus5=float(a2 ** (-2.5)))
        if m == 128:
            n = base_report["n_gamma"]
            ts = 2 * np.pi * np.arange(n) / n
            q, f = problem.boundary_data(ts)
            computed = eval_stokes_regularized(geometry, q, f, targets, theta, b, delta, 7)
            exact = problem.exact(targets)[0]
            errors = np.linalg.norm(computed - exact, axis=-1)
            ie = int(np.argmax(errors))
            y = targets[ie]
            dense_x = geometry(t_dense)[0]
            j = int(np.argmin(np.sum((dense_x - y) ** 2, axis=-1)))
            period_step = 2 * np.pi / len(t_dense)
            opt = minimize_scalar(lambda t: float(np.sum((geometry(np.array([t]))[0][0] - y) ** 2)),
                                  bounds=(t_dense[j] - period_step, t_dense[j] + period_step),
                                  method="bounded", options={"xatol": 1e-15})
            xo, _, _, no, ko, _ = geometry(np.array([opt.x]))
            independent_b = float(np.dot(y - xo[0], no[0]))
            fresh[rule] = dict(maximum_error=float(errors[ie]), maximum_point=y.tolist(),
                               computed_closest_t=float(theta[ie]),
                               computed_b=float(b[ie]), computed_alpha2=float(alpha2[ie]),
                               dense_refined_closest_t=float(opt.x),
                               dense_refined_b=independent_b,
                               dense_refined_alpha2=float(1 + independent_b * ko[0]),
                               max_saved_fresh_error_difference=float(abs(errors[ie] - row["velocity_error_inf"])))
            pd = pressure_densities(geometry, q, f)
            normal = geometry(theta[ie:ie+1])[3]
            tau = geometry(theta[ie:ie+1])[2]
            ks = geometry(theta[ie:ie+1])[5]
            aa = np.sqrt(alpha2[ie])
            f0 = PeriodicSamples(f)(theta[ie:ie+1])
            qs = PeriodicSamples(pd.q_s)(theta[ie:ie+1])
            qss = PeriodicSamples(pd.q_ss)(theta[ie:ie+1])
            bv = b[ie]
            scale = abs(bv) * scaled_moment_n0(7, abs(bv)/delta)
            sym = normal*np.sum(tau*qs,axis=-1)[:,None] + tau*np.sum(normal*qs,axis=-1)[:,None]
            c1 = sym/aa**3
            c2 = -bv/(2*aa**3)*normal*np.sum(normal*qss,axis=-1)[:,None]
            c3 = bv*bv*ks[:,None]/(2*aa**5)*normal*np.sum(normal*qs,axis=-1)[:,None]
            slcorr = scale*normal*np.sum(normal*f0,axis=-1)[:,None]/aa/(4*np.pi)
            dlcorr = scale*(c1+c2+c3)/(2*np.pi)
            q2, f2 = problem.boundary_data(2*np.pi*np.arange(2*n)/(2*n))
            w = eval_stokes_regularized(geometry,q2,f2,y[None],theta[ie:ie+1],b[ie:ie+1],delta,7)
            fresh[rule].update(single_layer_correction=slcorr[0].tolist(),
                double_layer_correction=dlcorr[0].tolist(),
                stresslet_correction_terms=[(scale*c/(2*np.pi))[0].tolist() for c in (c1,c2,c3)],
                source_doubling_difference=float(np.linalg.norm(computed[ie]-w[0])),
                raw_combined_error=float(np.linalg.norm(computed[ie]-slcorr[0]-dlcorr[0]-exact[ie])))
        rows.append(row)
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "geometric_diagnostic.json").write_text(json.dumps(dict(
    method="Closest-point geometry on current direct halo; saved full-grid fields; fresh M=128 direct evaluation",
    curvature_dense_samples=len(t_dense), rows=rows, fresh_M128=fresh), indent=2) + "\n")
keys = ["m_grid", "width_rule", "h_c", "n_gamma", "delta", "delta_curvature_inf",
        "min_alpha_squared", "velocity_error_inf", "velocity_max_alpha_squared", "velocity_max_alpha_minus5"]
with (ROOT / "geometric_diagnostic.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(dict(curvature_inf=kappa_inf, fresh_M128=fresh, rows=rows), indent=2))
