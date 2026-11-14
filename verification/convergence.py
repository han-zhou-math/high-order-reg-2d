"""Run analytic benchmarks; write CSV tables and a convergence plot."""
import argparse
import csv
from pathlib import Path
import numpy as np
from regbie.regularized_slp import circle, ellipse, starfish, single_layer
from regbie.starfish_reference import starfish_reference
from functools import lru_cache


@lru_cache(maxsize=4)
def star_reference_cached(theta):
    return starfish_reference(np.array(theta))

# A constant plus a nontrivial Fourier mode avoids special constant-density
# superconvergence. Ellipse density is divided by speed so f ds = g(t) dt.
MODE = 3
AMPLITUDE = 0.7


def benchmark(case, theta, delta):
    g = lambda t: 1+AMPLITUDE*np.cos(MODE*t)
    if case == 'starfish_on':
        curve = starfish
        density = lambda t: g(t)/np.linalg.norm(curve(t)[1], axis=1)
        targets = curve(theta)[0]
        exact = star_reference_cached(tuple(theta))
        b = 0.
    elif case == 'ellipse_on':
        curve = ellipse
        density = lambda t: g(t)/np.linalg.norm(curve(t)[1], axis=1)
        targets = curve(theta)[0]
        c, q = 1.1, (1.4-0.8)/(1.4+0.8)
        exact = np.log(c)-AMPLITUDE*(1+q**MODE)*np.cos(MODE*theta)/(2*MODE)
        b = 0.
    else:
        lam = {'circle_on': 0., 'circle_inside': -0.5,
               'circle_outside': 0.5}[case]
        b = lam*delta
        radius = 1+b
        if radius <= 0:
            raise ValueError('delta too large for interior target family')
        curve, density = circle, g
        targets = radius*curve(theta)[0]
        exact = (np.log(max(radius, 1.))-AMPLITUDE*
                 min(radius, 1/radius)**MODE*np.cos(MODE*theta)/(2*MODE))
    return curve, density, targets, b, exact


def run(output, ns, kappa=1., common_q=None, cases=None):
    output.mkdir(parents=True, exist_ok=True)
    # Fixed target points include t=0 (a quadrature node) and off-grid points.
    theta = np.linspace(0, 2*np.pi, 97, endpoint=False)
    if cases is None:
        cases = ['circle_on', 'circle_inside', 'circle_outside', 'ellipse_on']
    reference_difference = None
    if 'starfish_on' in cases:
        reference_difference = np.max(np.abs(starfish_reference(theta, 4096)-
                                             starfish_reference(theta, 8192)))
        if reference_difference > 1e-13:
            raise RuntimeError('Starfish reference has not converged')
    rows = []
    for case in cases:
        for p, default_q in [(1, 2/3), (3, 2/3), (5, 4/5), (7, 5/7)]:
            q = default_q if common_q is None else common_q
            previous = None
            for n in ns:
                h = 2*np.pi/n
                delta = kappa*h**q
                curve, density, y, b, exact = benchmark(case, theta, delta)
                approx = single_layer(curve, density, y, delta, n, p, b)
                fine = single_layer(curve, density, y, delta, 4*n, p, b)
                error = np.max(np.abs(approx-exact))
                reg_error = np.max(np.abs(fine-exact))
                quadrature = np.max(np.abs(approx-fine))
                rate_h = np.nan if previous is None else np.log(previous[1]/error)/np.log(previous[0]/h)
                rows.append(dict(case=case, order=p, q=q, n=n, h=h, delta=delta,
                                 delta_over_h=delta/h, error_inf=error,
                                 resolved_error_inf=reg_error, quadrature_estimate=quadrature,
                                 rate_h=rate_h, rate_delta=rate_h/q,
                                 expected_rate_h=p*q, expected_rate_delta=p))
                previous = h, error
    with (output/'convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrows = (len(cases)+1)//2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4*nrows), squeeze=False, constrained_layout=True)
    summaries = ['# Convergence results', '',
                 'Circle/ellipse references are analytic; the starfish uses logarithmic subtraction. The quadrature estimate compares N and 4N at fixed delta.',
                 'Rates below use the last three resolved errors above 5e-13 (before roundoff).', '',
                 '| Case | p | q | fitted rate in h | fitted rate in delta | expected in h |',
                 '|---|---:|---:|---:|---:|---:|']
    for case, ax in zip(cases, axes.flat):
        for p in [1, 3, 5, 7]:
            subset = [r for r in rows if r['case'] == case and r['order'] == p]
            hs = np.array([r['h'] for r in subset])
            es = np.array([r['error_inf'] for r in subset])
            q = subset[0]['q']
            line, = ax.loglog(hs, es, 'o-', label=f'p={p}, expected h^{p*q:g}')
            valid = [r for r in subset if r['error_inf'] > 5e-13 and
                     r['quadrature_estimate'] < .05*r['error_inf']][-3:]
            if len(valid) >= 3:
                slope = np.polyfit(np.log([r['h'] for r in valid]),
                                   np.log([r['error_inf'] for r in valid]), 1)[0]
                summaries.append(f'| {case} | {p} | {q:.4f} | {slope:.3f} | {slope/q:.3f} | {p*q:.3f} |')
                anchor = valid[-1]
                ax.loglog(hs, anchor['error_inf']*(hs/anchor['h'])**(p*q),
                          '--', color=line.get_color(), alpha=.45)
            else:
                summaries.append(f'| {case} | {p} | {q:.4f} | insufficient resolved levels | — | {p*q:.3f} |')
        ax.set(title=case.replace('_', ' '), xlabel='h = 2π/N', ylabel='maximum absolute error')
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(cases):]:
        if 'starfish_on' in cases:
            t = np.linspace(0, 2*np.pi, 601)
            x, _ = starfish(t)
            ax.plot(x[:, 0], x[:, 1])
            ax.set(aspect='equal', title='Three-lobed starfish: r(t) = 1 + 0.2 cos(3t)', xlabel='x', ylabel='y')
            ax.grid(alpha=.2)
        else:
            ax.set_visible(False)
    if reference_difference is not None:
        summaries += ['', f'Starfish reference check (4096 vs 8192 nodes): {reference_difference:.3e}.']
    fig.suptitle('Regularized 2D single layer: δ = κ h^q (κ = '+str(kappa)+')\nDashed lines: predicted slopes')
    fig.savefig(output/'convergence.png', dpi=180)
    plt.close(fig)
    report = '\n'.join(summaries)+'\n'
    (output/'summary.md').write_text(report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--ns', type=int, nargs='+', default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument('--kappa', type=float, default=1.)
    parser.add_argument('--q', type=float, default=None, help='override q for every order; require 0<q<1')
    parser.add_argument('--cases', nargs='+', choices=['circle_on', 'circle_inside', 'circle_outside', 'ellipse_on', 'starfish_on'])
    args = parser.parse_args()
    if (args.q is not None and not 0 < args.q < 1) or args.kappa <= 0:
        parser.error('require 0<q<1 and kappa>0')
    if len(args.ns) < 2 or min(args.ns) < 4 or any(b <= a for a,b in zip(args.ns,args.ns[1:])):
        parser.error('ns must contain at least two strictly increasing integers >=4')
    run(args.output, args.ns, args.kappa, args.q, args.cases)
