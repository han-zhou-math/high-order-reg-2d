"""Focused reproducible studies: Stokes corrections, near-curve limits, choice of delta, timing.

Uses current scaled kite/starfish geometry; no existing data are overwritten.
Run: OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/reg-bie-mpl python3 -m
     experiments.paper_revision --stage all

The independent comparator integrates ORIGINAL kernels on target-dependent,
geometrically graded Gauss--Legendre panels. It uses analytic densities and
geometry; timings compare these concrete Python implementations, not optimized
implementations of other published near-singular quadrature methods.
"""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from regbie.regularized_slp import coefficients, regularized_kernel
from regbie.surface_operators import radial_factors
from regbie.verify_stokes_derivation import scaled_moment_n0
from .allgrid import closest_points
from .bie_evaluators import (PeriodicSamples, eval_laplace_regularized,
                             eval_stokes_regularized, eval_pressure_regularized)
from .boundary_data import spectral_derivative
from .swapped_low_order import LAPLACE_KITE, STOKES_STARFISH

ROOT = Path("results/paper_revision")
STOKES_GEOMETRY = STOKES_STARFISH.geometry


@contextmanager
def coefficient_counter():
    """Time all existing coefficient calls; this is a subtotal of other stages."""
    from regbie import regularized_slp
    from regbie import regularized_dlp
    from regbie import surface_operators
    from regbie import verify_stokes_derivation
    modules = (regularized_slp, regularized_dlp, surface_operators,
               verify_stokes_derivation)
    originals = [module.coefficients for module in modules]
    counter = {"seconds": 0., "calls": 0}
    original = originals[0]
    def timed_coefficients(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        counter["seconds"] += time.perf_counter()-start
        counter["calls"] += 1
        return result
    for module in modules:
        module.coefficients = timed_coefficients
    try:
        yield counter
    finally:
        for module, function in zip(modules, originals):
            module.coefficients = function


def save(name, rows):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stokes_densities(t):
    """Return independent q,f arrays (...,2), unrelated by a PDE representation."""
    t = np.asarray(t)
    q = np.stack((np.sin(2*t) + .25*np.cos(5*t),
                  .6*np.cos(3*t) - .2*np.sin(t)), axis=-1)
    f = np.stack((.7*np.cos(2*t) + .15*np.sin(4*t),
                  .5*np.sin(3*t) + .2*np.cos(t)), axis=-1)
    return q, f


def max_error(value, reference):
    difference = value - reference
    if difference.ndim > 1:
        difference = np.linalg.norm(difference, axis=-1)
    return float(np.max(np.abs(difference)))


def target_set(problem, offset, count=49):
    theta = 2*np.pi*(np.arange(count)+.37)/count
    x0, _, _, normal, _, _ = problem.geometry(theta)
    b = np.full(count, offset)
    return theta, b, x0 + b[:, None]*normal


def physical_spacing(geometry, n):
    points = geometry(2*np.pi*np.arange(n)/n)[0]
    return float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).max())


def stokes_regularized(n, targets, theta, b, delta, order):
    """Return separate corrected/raw SLP,DLP and nonoverlapping stage timings.

    Density derivatives and off-node interpolation follow the actual FFT path.
    The returned total excludes projection; callers add it when applicable.
    """
    start = time.perf_counter()
    t = 2*np.pi*np.arange(n)/n
    x, speed, _, normal, _, _ = STOKES_GEOMETRY(t)
    q, f = stokes_densities(t)
    qs = spectral_derivative(q)/speed[:, None]
    qss = spectral_derivative(qs)/speed[:, None]
    qi, fi = PeriodicSamples(q), PeriodicSamples(f)
    qsi, qssi = PeriodicSamples(qs), PeriodicSamples(qss)
    prepare_seconds = time.perf_counter()-start
    local_start = time.perf_counter()
    _, _, tau0, n0, kappa, kappa_s = STOKES_GEOMETRY(theta)
    q0, f0, qs0, qss0 = qi(theta), fi(theta), qsi(theta), qssi(theta)
    alpha = np.sqrt(1+b*kappa)
    assert np.all(np.isfinite(alpha)) and np.all(alpha > 0)
    scale = np.abs(b)*np.array([scaled_moment_n0(order, abs(bb)/delta) for bb in b])
    b0 = n0*np.sum(n0*f0, axis=-1)[:, None]/alpha[:, None]
    sym = n0*np.sum(tau0*qs0, axis=-1)[:, None] + tau0*np.sum(n0*qs0, axis=-1)[:, None]
    c2 = sym/alpha[:, None]**3
    c2 -= (b/(2*alpha**3))[:, None]*n0*np.sum(n0*qss0, axis=-1)[:, None]
    c2 += (b*b*kappa_s/(2*alpha**5))[:, None]*n0*np.sum(n0*qs0, axis=-1)[:, None]
    correction_sl = scale[:, None]*b0/(4*np.pi)
    correction_dl = scale[:, None]*c2/(2*np.pi)
    local_seconds = time.perf_counter()-local_start
    kernel_start = time.perf_counter()
    r = x[None]-targets[:, None]
    r2 = np.sum(r*r, axis=-1)
    u = r2/delta**2
    a, ap, _ = radial_factors(u, order, np.abs(b[:, None])/delta)
    gf = regularized_kernel(np.sqrt(r2), delta, order, b[:, None])
    rf = np.sum(r*f[None], axis=-1)
    sl_integrand = r*rf[..., None]*a[..., None]/(4*np.pi*delta**2) - gf[..., None]*f[None]/2
    rq = np.sum(r*(q[None]-q0[:, None]), axis=-1)
    rn = np.sum(r*normal[None], axis=-1)
    dl_integrand = r*(rq*rn*(-ap)/(np.pi*delta**4))[..., None]
    weights = 2*np.pi*speed/n
    raw_sl = np.einsum("mni,n->mi", sl_integrand, weights)
    chi = np.where(b < 0, 1., np.where(b > 0, 0., .5))
    raw_dl = np.einsum("mni,n->mi", dl_integrand, weights)+chi[:, None]*q0
    kernel_seconds = time.perf_counter()-kernel_start
    return (raw_sl+correction_sl, raw_dl+correction_dl, raw_sl, raw_dl,
            dict(prepare_seconds=prepare_seconds, local_seconds=local_seconds,
                 kernel_seconds=kernel_seconds, evaluation_seconds=time.perf_counter()-start))


@lru_cache(maxsize=None)
def gauss_rule(order):
    return leggauss(order)


def graded_nodes(theta, b, speed, order, split=1):
    """GL panels in t-theta, with dyadically increasing width away from zero.

    Central width is |b|/|x'(theta)|. At b=0 use 32 equal smooth panels.
    split=2 bisects every panel for an independent refinement diagnostic.
    """
    if b == 0:
        edges = np.linspace(-np.pi, np.pi, 33)
    else:
        width = min(np.pi, abs(b)/speed)
        positive = [0., width]
        while positive[-1] < np.pi:
            positive.append(min(np.pi, 2*positive[-1]))
        edges = np.r_[-np.asarray(positive[:0:-1]), positive]
    # Bound the far-panel width as well: the five-lobed geometry has
    # significant variation away from the nearest point.
    edges = np.unique(np.r_[edges, np.linspace(-np.pi, np.pi, 33)])
    if split == 2:
        edges = np.sort(np.r_[edges, .5*(edges[:-1]+edges[1:])])
    nodes, weights = gauss_rule(order)
    mid = .5*(edges[1:]+edges[:-1])
    half = .5*np.diff(edges)
    t = theta+(mid[:, None]+half[:, None]*nodes).ravel()
    w = (half[:, None]*weights).ravel()
    return t, w


def panel_original(problem, targets, theta, b, order=32, split=1):
    """Independent original-kernel graded GL evaluator; returns pair and work."""
    stokes = problem is STOKES_STARFISH
    out_sl = np.empty((len(targets), 2) if stokes else len(targets))
    out_dl = np.empty_like(out_sl)
    total_nodes = 0
    for i, (y, t0, bb) in enumerate(zip(targets, theta, b)):
        speed0 = float(problem.geometry(np.array(t0))[1])
        t, dtw = graded_nodes(t0, bb, speed0, order, split)
        x, speed, _, normal, _, _ = problem.geometry(t)
        w = speed*dtw
        r = x-y
        r2 = np.sum(r*r, axis=-1)
        chi = 1. if bb < 0 else (0. if bb > 0 else .5)
        if stokes:
            q, f = stokes_densities(t)
            q0 = stokes_densities(np.array(t0))[0]
            rf = np.sum(r*f, axis=-1)
            sl = (-.5*np.log(r2)[:, None]*f+r*(rf/r2)[:, None])/(4*np.pi)
            rq = np.sum(r*(q-q0), axis=-1)
            rn = np.sum(r*normal, axis=-1)
            dl = r*(rq*rn/(np.pi*r2**2))[:, None]
            out_sl[i] = np.einsum("ni,n->i", sl, w)
            out_dl[i] = np.einsum("ni,n->i", dl, w)+chi*q0
        else:
            f, g = problem.boundary_densities(t)
            g0 = problem.boundary_densities(np.array(t0))[1]
            out_sl[i] = np.sum(.5*np.log(r2)*f*w)/(2*np.pi)
            out_dl[i] = np.sum(np.sum(r*normal, axis=-1)/r2*(g-g0)*w)/(2*np.pi)+chi*g0
        total_nodes += len(t)
    return out_sl, out_dl, total_nodes


def original_uniform_stokes(targets, n=65536):
    t = 2*np.pi*(np.arange(n)+.173)/n
    x, speed, _, normal, _, _ = STOKES_GEOMETRY(t)
    q, f = stokes_densities(t)
    w = 2*np.pi*speed/n
    sl = []
    dl = []
    for y in targets:
        r = x-y
        r2 = np.sum(r*r, axis=-1)
        rf = np.sum(r*f, axis=-1)
        sl.append(np.einsum("ni,n->i", (-.5*np.log(r2)[:, None]*f+r*(rf/r2)[:, None])/(4*np.pi), w))
        rq = np.sum(r*q, axis=-1)
        rn = np.sum(r*normal, axis=-1)
        dl.append(np.einsum("ni,n->i", r*(rq*rn/(np.pi*r2**2))[:, None], w))
    return np.array(sl), np.array(dl)


def ablation():
    rows = []
    reference_checks = []
    for delta in (.04, .02, .01, .005, .0025, .00125):
        n = max(8192, 2**int(np.ceil(np.log2(32/delta))))
        for lam in (-.5, .5):
            theta, b, targets = target_set(STOKES_STARFISH, lam*delta)
            ref_sl, ref_dl, work = panel_original(STOKES_STARFISH, targets, theta, b, 48)
            fine_sl, fine_dl, _ = panel_original(STOKES_STARFISH, targets, theta, b, 64, 2)
            ref_check = max(max_error(ref_sl, fine_sl), max_error(ref_dl, fine_dl))
            assert ref_check < 2e-11, ref_check
            if delta == .01:
                uniform = original_uniform_stokes(targets)
                independent = max(max_error(uniform[0], fine_sl), max_error(uniform[1], fine_dl))
                assert independent < 2e-11, independent
                reference_checks.append(dict(delta=delta, lam=lam, uniform_check=independent))
            for order in (3, 5, 7):
                sl, dl, raw_sl, raw_dl, _ = stokes_regularized(n, targets, theta, b, delta, order)
                finer = stokes_regularized(2*n, targets, theta, b, delta, order)
                for name, value, raw, reference, check in (
                    ("single", sl, raw_sl, fine_sl, finer[0]),
                    ("double", dl, raw_dl, fine_dl, finer[1])):
                    rows.append(dict(operator=name, p=order, delta=delta, lambda_signed=lam,
                                     n_gamma=n, target_count=len(targets), error_inf=max_error(value, reference),
                                     raw_error_inf=max_error(raw, reference),
                                     source_doubling_inf=max_error(value, check), reference_refinement_inf=ref_check))
            save("stokes_ablation.csv", rows)
            print("ablation", delta, lam, flush=True)
    (ROOT/"reference_checks.json").write_text(json.dumps(reference_checks, indent=2)+"\n")


def width():
    rows = []
    n = 2048
    hp = physical_spacing(STOKES_GEOMETRY, n)
    # Targets stay fixed throughout the sweep, independently of delta.
    theta, b, targets = target_set(STOKES_STARFISH, -.003)
    ref_sl, ref_dl, _ = panel_original(STOKES_STARFISH, targets, theta, b, 64, 2)
    for order in (3, 5, 7):
        for width_ratio in (.35, .5, .75, 1., 1.5, 2., 3., 4., 6., 8.):
            delta = width_ratio*hp
            sl, dl, _, _, _ = stokes_regularized(n, targets, theta, b, delta, order)
            fine_sl, fine_dl, _, _, _ = stokes_regularized(2*n, targets, theta, b, delta, order)
            for name, value, fine, reference in (("single", sl, fine_sl, ref_sl),
                                                  ("double", dl, fine_dl, ref_dl)):
                rows.append(dict(operator=name, p=order, n_gamma=n, h_phys_max=hp,
                                 delta=delta, delta_over_h_phys=width_ratio,
                                 b=-.003, target_count=len(targets), error_inf=max_error(value, reference),
                                 source_doubling_inf=max_error(value, fine)))
            save("width_sweep.csv", rows)
        print("width", order, flush=True)


def benchmark():
    rows = []
    repeats = 3
    for name, problem in (("laplace", LAPLACE_KITE), ("stokes", STOKES_STARFISH)):
        theta1, b1, y1 = target_set(problem, -.001, 25)
        theta2, b2, y2 = target_set(problem, .001, 24)
        theta, b, targets = np.r_[theta1,theta2], np.r_[b1,b2], np.r_[y1,y2]
        if name == "laplace":
            reference = np.r_[problem.exact(y1, "inside"),problem.exact(y2, "outside")]
        else:
            sl, dl, _ = panel_original(problem, targets, theta, b, 64, 2)
            reference = sl+dl
        for n in (256, 512, 1024, 2048, 4096, 8192, 16384):
            hp = physical_spacing(problem.geometry, n)
            delta = 2*hp
            for order in (3,5,7):
                values, times = [], []
                # Complete independent repeats: include target projection, density
                # samples, FFT differentiation/interpolation and coefficients.
                for repeat in range(repeats):
                    start = time.perf_counter()
                    ct, bb = closest_points(targets, problem.geometry, problem.inside)
                    projection_seconds = time.perf_counter()-start
                    with coefficient_counter() as counter:
                        if name == "stokes":
                            sl, dl, _, _, stage = stokes_regularized(n, targets, ct, bb, delta, order)
                            value = sl+dl
                        else:
                            prepare_start = time.perf_counter()
                            t = 2*np.pi*np.arange(n)/n
                            f, g = problem.boundary_densities(t)
                            fi, gi = PeriodicSamples(f), PeriodicSamples(g)
                            prepare_seconds = time.perf_counter()-prepare_start
                            kernel_start = time.perf_counter()
                            value = eval_laplace_regularized(problem.geometry, fi, gi, targets, ct, bb, delta, n, order)
                            stage = dict(prepare_seconds=prepare_seconds, local_seconds=0.,
                                         kernel_seconds=time.perf_counter()-kernel_start)
                    stage["coefficient_seconds"] = counter["seconds"]
                    values.append(value)
                    times.append(dict(total_seconds=time.perf_counter()-start,
                                      projection_seconds=projection_seconds, **stage))
                timing = {key:float(np.median([item[key] for item in times]))
                          for key in times[0] if key != "evaluation_seconds"}
                rows.append(dict(problem=name, method=f"E1-p{order}", n_gamma=n,
                                 total_kernel_nodes=n*len(targets), nodes_per_target=n,
                                 delta=delta, delta_over_h_phys=2., target_count=len(targets),
                                 error_inf=max_error(values[-1], reference), repetitions=repeats, **timing))
            save("accuracy_work.csv", rows)
            print("benchmark", name, n, flush=True)
        for gl_order in (4, 6, 8, 12, 16, 24, 32):
            timings = []
            for repeat in range(repeats):
                start = time.perf_counter()
                ct, bb = closest_points(targets, problem.geometry, problem.inside)
                projection_seconds = time.perf_counter()-start
                kernel_start = time.perf_counter()
                sl, dl, nodes = panel_original(problem, targets, ct, bb, gl_order)
                timings.append(dict(total_seconds=time.perf_counter()-start,
                                    projection_seconds=projection_seconds, prepare_seconds=0., local_seconds=0.,
                                    coefficient_seconds=0.,
                                    kernel_seconds=time.perf_counter()-kernel_start))
            timing = {key:float(np.median([item[key] for item in timings])) for key in timings[0]}
            rows.append(dict(problem=name, method=f"graded-GL-{gl_order}", n_gamma=0,
                             total_kernel_nodes=nodes, nodes_per_target=nodes/len(targets),
                             delta=0., delta_over_h_phys=0., target_count=len(targets),
                             error_inf=max_error(sl+dl, reference), repetitions=repeats, **timing))
        save("accuracy_work.csv", rows)


def limits():
    rows = []
    n = 8192
    delta = .01
    theta, zero, on_surface = target_set(STOKES_STARFISH, 0., 33)
    q0 = stokes_densities(theta)[0]
    for order in (3,5,7):
        surface_sl, surface_dl, _, _, _ = stokes_regularized(n, on_surface, theta, zero, delta, order)
        for lam_abs in (.5, .01, 1e-4, 1e-6, 1e-8, 1e-10):
            for sign in (-1,1):
                b = np.full(len(theta), sign*lam_abs*delta)
                normal = STOKES_GEOMETRY(theta)[3]
                targets = on_surface+b[:,None]*normal
                sl, dl, _, _, _ = stokes_regularized(n, targets, theta, b, delta, order)
                dl_limit = surface_dl+(-sign*.5)*q0
                assert np.isfinite(sl).all() and np.isfinite(dl).all()
                rows.append(dict(operator="single", p=order, lambda_signed=sign*lam_abs,
                                 delta=delta, n_gamma=n, trace_difference_inf=max_error(sl, surface_sl)))
                rows.append(dict(operator="double", p=order, lambda_signed=sign*lam_abs,
                                 delta=delta, n_gamma=n, trace_difference_inf=max_error(dl, dl_limit)))
    save("tiny_lambda.csv", rows)


def manufactured_limits():
    """True manufactured-field errors at very small nonzero signed offsets."""
    n, delta, order = 8192, .01, 7
    rows = []
    for problem in (LAPLACE_KITE, STOKES_STARFISH):
        theta, zero, on_surface = target_set(problem, 0., 33)
        normal = problem.geometry(theta)[3]
        offsets = [0.]
        for lam in (.5, .01, 1e-4, 1e-6, 1e-8, 1e-10):
            offsets.extend((-lam, lam))
        for lam in offsets:
            b = np.full(len(theta), lam*delta)
            targets = on_surface+b[:, None]*normal
            side = "inside" if lam < 0 else ("outside" if lam > 0 else "average")
            values = []
            for sources in (n, 2*n):
                t = 2*np.pi*np.arange(sources)/sources
                if problem is LAPLACE_KITE:
                    f, g = problem.boundary_densities(t)
                    values.append({"laplace":eval_laplace_regularized(problem.geometry,
                        PeriodicSamples(f), PeriodicSamples(g), targets, theta, b, delta,
                        sources, order)})
                else:
                    q, f = problem.boundary_data(t)
                    values.append({"velocity":eval_stokes_regularized(problem.geometry,
                        q, f, targets, theta, b, delta, order),
                        "pressure":eval_pressure_regularized(problem.geometry,
                        q, f, targets, theta, b, delta, order)})
            if problem is LAPLACE_KITE:
                exact = {"laplace":problem.exact(targets, side)}
            else:
                velocity, pressure = problem.exact(targets, side)
                exact = {"velocity":velocity,"pressure":pressure}
            for field, value in values[0].items():
                assert np.all(np.isfinite(value))
                rows.append(dict(geometry=problem.name,field=field,p=order,n_gamma=n,
                    delta=delta,lambda_signed=lam,target_count=len(targets),
                    error_inf=max_error(value,exact[field]),
                    source_doubling_inf=max_error(value,values[1][field])))
        print("manufactured small-lambda", problem.name, flush=True)
        save("manufactured_tiny_lambda.csv",rows)


def validate():
    import scipy
    theta,b,y=target_set(STOKES_STARFISH,-.01,17)
    t=2*np.pi*np.arange(1024)/1024
    q,f=stokes_densities(t)
    separate=stokes_regularized(1024,y,theta,b,.03,7)
    current=eval_stokes_regularized(STOKES_GEOMETRY,q,f,y,theta,b,.03,7)
    discrepancy=max_error(separate[0]+separate[1],current)
    assert discrepancy < 1e-13,discrepancy
    ROOT.mkdir(parents=True,exist_ok=True)
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    metadata=dict(python=platform.python_version(), numpy=np.__version__,scipy=scipy.__version__,
                  platform=platform.platform(),cpu_model=cpu_model,
                  curve_names=[LAPLACE_KITE.name,STOKES_STARFISH.name],
                  existing_evaluator_discrepancy=discrepancy,
                  note="Single-thread OPENBLAS. Timing medians over three complete evaluations, no cached projection.")
    (ROOT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


def summarize():
    """Save finite-range rates and fastest tested tolerance configurations."""
    def load(name):
        with (ROOT/name).open() as stream:
            return list(csv.DictReader(stream))
    rows = load("stokes_ablation.csv")
    rates = []
    for lam in (-.5, .5):
        for order in (3, 5, 7):
            for operator in ("single", "double"):
                selected = [row for row in rows if int(row["p"]) == order
                            and row["operator"] == operator
                            and float(row["lambda_signed"]) == lam]
                resolved = [row for row in selected if float(row["error_inf"])
                            > 20*max(float(row["source_doubling_inf"]),
                                     float(row["reference_refinement_inf"]))]
                coarse, fine = resolved[-2:]
                rates.append(dict(operator=operator, p=order, lambda_signed=lam,
                    delta_coarse=float(coarse["delta"]), delta_fine=float(fine["delta"]),
                    rate=float(np.log2(float(coarse["error_inf"])/float(fine["error_inf"]))),
                    raw_rate=float(np.log2(float(coarse["raw_error_inf"])/float(fine["raw_error_inf"]))),
                    error_fine=float(fine["error_inf"])))
    save("ablation_rates.csv", rates)
    work = load("accuracy_work.csv")
    threshold = []
    for problem in ("laplace", "stokes"):
        for tolerance in (1e-6, 1e-9, 1e-12):
            for family in ("E1", "graded"):
                achieved = [row for row in work if row["problem"] == problem
                            and row["method"].startswith(family)
                            and float(row["error_inf"]) <= tolerance]
                if achieved:
                    best = min(achieved, key=lambda row: float(row["total_seconds"]))
                    threshold.append(dict(tolerance=tolerance, **best))
    save("threshold.csv", threshold)
    summary = dict(ablation_rates=rates, threshold=threshold,
        reference_refinement_max=max(float(row["reference_refinement_inf"]) for row in rows),
        quadrature_difference_max=max(float(row["source_doubling_inf"]) for row in rows))
    (ROOT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def load(name):
        with (ROOT/name).open() as stream:
            return list(csv.DictReader(stream))
    colors={3:"#2878B5",5:"#E07A20",7:"#228C57"}
    rows=load("stokes_ablation.csv")
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
    for ax,operator in zip(axs,("single","double")):
        for order in (3,5,7):
            selected=[r for r in rows if r["operator"]==operator and int(r["p"])==order and float(r["lambda_signed"])==-.5]
            d=np.array([float(r["delta"]) for r in selected])
            ax.loglog(d,[float(r["error_inf"]) for r in selected],"o-",color=colors[order],label=f"corrected p={order}")
            ax.loglog(d,[float(r["raw_error_inf"]) for r in selected],"x--",color=colors[order],label=f"uncorrected p={order}")
        ax.set(xlabel=r"$\delta$",ylabel="maximum vector error",title=f"Stokes {operator} layer; b=-δ/2")
        ax.grid(True,which="both",alpha=.2)
    axs[0].legend(fontsize=7,ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT/"stokes_ablation.pdf")
    fig.savefig(ROOT/"stokes_ablation.png",dpi=160)
    plt.close(fig)
    rows=load("width_sweep.csv")
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
    for ax,operator in zip(axs,("single","double")):
        for order in (3,5,7):
            selected=[r for r in rows if r["operator"]==operator and int(r["p"])==order]
            d=[float(r["delta_over_h_phys"]) for r in selected]
            ax.loglog(d,[float(r["error_inf"]) for r in selected],"o-",color=colors[order],label=f"error p={order}")
            ax.loglog(d,[float(r["source_doubling_inf"]) for r in selected],"x--",color=colors[order],label=f"N to 2N, p={order}")
        ax.set(xlabel=r"$\delta/h_{\mathrm{phys}}$",ylabel="maximum vector difference",title=f"Stokes {operator} layer; N=2048, b=-0.003")
        ax.grid(True,which="both",alpha=.2)
    axs[0].legend(fontsize=7,ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT/"width_sweep.pdf")
    fig.savefig(ROOT/"width_sweep.png",dpi=160)
    plt.close(fig)
    rows=load("accuracy_work.csv")
    fig,axs=plt.subplots(1,2,figsize=(9.2,3.5))
    for ax,problem in zip(axs,("laplace","stokes")):
        for order in (3,5,7):
            selected=[r for r in rows if r["problem"]==problem and r["method"]==f"E1-p{order}"]
            ax.loglog([float(r["total_seconds"]) for r in selected],[float(r["error_inf"]) for r in selected],"o-",color=colors[order],label=f"regularized, $p={order}$")
        selected=[r for r in rows if r["problem"]==problem and r["method"].startswith("graded")]
        ax.loglog([float(r["total_seconds"]) for r in selected],[float(r["error_inf"]) for r in selected],"s--",color=".3",label="graded Gauss–Legendre")
        ax.set(xlabel="wall time (seconds, 49 targets)",ylabel="maximum error",title=f"{problem.capitalize()}: fixed b=±0.001")
        ax.grid(True,which="both",alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/"accuracy_work.pdf")
    fig.savefig(ROOT/"accuracy_work.png",dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["all","ablation","width","benchmark","limits",
                         "manufactured_limits","summary","plot"],default="all")
    args=parser.parse_args()
    validate()
    stages={"ablation":ablation,"width":width,"benchmark":benchmark,"limits":limits,
            "manufactured_limits":manufactured_limits,
            "summary":summarize,"plot":plot}
    for name,function in stages.items():
        if args.stage in (name,"all"):
            function()


if __name__ == "__main__":
    main()
