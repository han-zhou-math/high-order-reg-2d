"""Sweep normal distances in the manufactured S+D test on the starfish."""
import argparse
import csv
from pathlib import Path
import numpy as np
from regbie.regularized_slp import starfish
from regbie.manufactured_convergence import target_family, check_closest_points, reconstruct


def run(output, ns, scaled, fixed):
    theta = 2*np.pi*np.arange(97)/97
    rows = []
    geometry = {}
    output.mkdir(parents=True, exist_ok=True)
    for kind, distances in [('scaled', scaled), ('fixed', fixed)]:
        for distance in distances:
            for p, q in [(3, 2/3), (5, 4/5), (7, 5/7)]:
                for n in ns:
                    h = 2*np.pi/n
                    delta = h**q
                    for side, sign in [('inside', -1), ('outside', 1)]:
                        b = sign*distance*(delta if kind == 'scaled' else 1.)
                        y, exact = target_family(starfish, theta, b)
                        if n == ns[0] and b not in geometry:
                            geometry[b] = check_closest_points(starfish, theta, y, b)
                        s, d = reconstruct(starfish, theta, y, b, delta, n, p)
                        sf, df = reconstruct(starfish, theta, y, b, delta, 4*n, p)
                        err = s+d-exact
                        rows.append(dict(kind=kind, distance=distance, side=side,
                                         p=p, q=q, n=n, h=h, delta=delta, b=b,
                                         lambda_abs=abs(b)/delta,
                                         error_inf=np.max(np.abs(err)),
                                         error_rms=np.sqrt(np.mean(err**2)),
                                         resolved_error_inf=np.max(np.abs(sf+df-exact)),
                                         quadrature_s=np.max(np.abs(s-sf)),
                                         quadrature_d=np.max(np.abs(d-df))))
                print(f'Finished {kind} distance={distance:g}, order={p}', flush=True)
    for kind, distances in [('scaled', scaled), ('fixed', fixed)]:
        for distance in distances:
            for side in ['inside', 'outside']:
                for p in [3, 5, 7]:
                    previous = None
                    for row in [r for r in rows if (r['kind'],r['distance'],r['side'],r['p']) == (kind,distance,side,p)]:
                        row['rate_h'] = np.nan if previous is None else np.log(previous['error_inf']/row['error_inf'])/np.log(previous['h']/row['h'])
                        previous = row
    with (output/'convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = ['# Distance sweep: manufactured single + double layer', '',
               'Starfish curve; 97 target parameters on each side. delta=h^q, h=2*pi/N.',
               'q=2/3, 4/5, 5/7 for kernel orders p=3,5,7.',
               'Each plotted error is the larger of the inside and outside maximum errors.',
               'The CSV retains both sides separately, RMS errors, distances, and each-layer N-versus-4N quadrature checks.', '',
               '## Fixed scaled distances: |b|=lambda delta', '',
               'Fits use the last three levels with error above 5e-13 and summed layer quadrature estimates below 5% of error.',
               'Rates use the maximum over both sides; small errors can leave too few asymptotic levels before roundoff.', '',
               '| lambda | p | fitted h rate | expected | fit N | finest error |',
               '|---:|---:|---:|---:|---|---:|']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for i, (kind, distances) in enumerate([('scaled', scaled), ('fixed', fixed)]):
        if kind == 'fixed':
            summary += ['', '## Fixed physical distances', '',
                        'Here lambda=|b|/delta increases during refinement. Gaussian localization can produce faster-than-algebraic decay until roundoff; a fixed h order is not assigned.', '',
                        '| distance | p | error at coarsest N | error at finest N | finest lambda |',
                        '|---:|---:|---:|---:|---:|']
        for ax, p in zip(axes[i], [3, 5, 7]):
            for distance in distances:
                pairs = [[r for r in rows if (r['kind'],r['distance'],r['p'],r['n']) == (kind,distance,p,n)] for n in ns]
                aggregate = [dict(n=n, h=pair[0]['h'], error=max(r['error_inf'] for r in pair),
                                  quadrature=max(r['quadrature_s']+r['quadrature_d'] for r in pair))
                             for n,pair in zip(ns,pairs)]
                hs = np.array([r['h'] for r in aggregate])
                es = np.array([r['error'] for r in aggregate])
                label = f'λ={distance:g}' if kind == 'scaled' else f'|b|={distance:g}'
                ax.loglog(hs, es, 'o-', label=label, markersize=4)
                if kind == 'scaled':
                    valid = [r for r in aggregate if r['error'] > 5e-13 and r['quadrature'] < .05*r['error']][-3:]
                    q = pairs[0][0]['q']
                    if len(valid) == 3:
                        slope = np.polyfit(np.log([r['h'] for r in valid]), np.log([r['error'] for r in valid]), 1)[0]
                        fit = ', '.join(str(r['n']) for r in valid)
                        rate = f'{slope:.3f}'
                    else:
                        fit, rate = 'insufficient resolved levels', '—'
                    summary.append(f'| {distance:g} | {p} | {rate} | {p*q:g} | {fit} | {es[-1]:.3e} |')
                else:
                    summary.append(f"| {distance:g} | {p} | {es[0]:.3e} | {es[-1]:.3e} | {pairs[-1][0]['lambda_abs']:.3f} |")
            ax.set(title=f'p={p}: '+('fixed |b|/δ' if kind == 'scaled' else 'fixed |b|'),
                   xlabel='h = 2π/N', ylabel='max error over both sides')
            ax.grid(which='both', alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Manufactured S + D on the starfish: dependence on normal distance')
    fig.savefig(output/'convergence.png', dpi=180)
    plt.close(fig)
    summary += ['', f'Largest checked closest-distance discrepancy: {max(geometry.values()):.3e}.',
                'The largest offset for each family was checked by sampled global search plus local minimization. This is a numerical geometry check, not a certified reach bound.']
    text = '\n'.join(summary)+'\n'
    (output/'summary.md').write_text(text)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/distance_sweep'))
    parser.add_argument('--ns', type=int, nargs='+', default=[128,256,512,1024,2048,4096,8192])
    parser.add_argument('--scaled', type=float, nargs='+', default=[.1,.25,.5,1.,2.])
    parser.add_argument('--fixed', type=float, nargs='+', default=[.01,.05,.1])
    args = parser.parse_args()
    if len(args.ns)<3 or min(args.ns)<4 or any(b<=a for a,b in zip(args.ns,args.ns[1:])):
        parser.error('ns must contain at least three increasing integers >=4')
    if any(not np.isfinite(d) or d<=0 for d in args.scaled+args.fixed):
        parser.error('distances must be finite and positive')
    run(args.output,args.ns,args.scaled,args.fixed)
