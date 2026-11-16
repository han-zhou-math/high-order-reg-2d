"""Piecewise harmonic manufactured solution reconstructed by S[f]+D[g]."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from .regularized_slp import circle, starfish, single_layer
from .regularized_dlp import normals, double_layer


def field(x, side):
    """Real analytic harmonic fields and exact Cartesian gradients."""
    z = x[..., 0]+1j*x[..., 1]
    if side == 'inside':
        value = z**3+.2*z**2+.3*z+1
        derivative = 3*z**2+.4*z+.3
    elif side == 'outside':
        value = .4/z+.2/z**2
        derivative = -.4/z**2-.4/z**3
    else:
        raise ValueError(side)
    return value.real, np.stack((derivative.real, -derivative.imag), axis=-1)


def densities(curve):
    def f(t):
        x, dx = curve(t)
        _, minus = field(x, 'inside')
        _, plus = field(x, 'outside')
        return np.sum((plus-minus)*normals(dx), axis=-1)
    def g(t):
        x, _ = curve(t)
        return field(x, 'inside')[0]-field(x, 'outside')[0]
    return f, g


def target_family(curve, theta, b):
    x, dx = curve(theta)
    y = x+b*normals(dx)
    if b == 0:
        exact = (field(y, 'inside')[0]+field(y, 'outside')[0])/2
    else:
        exact = field(y, 'inside' if b < 0 else 'outside')[0]
    return y, exact


def check_closest_points(curve, theta, y, b):
    """Check constructed normal offsets by sampled global search + minimization.

    This is a numerical geometry check, not a certified reach computation.
    """
    grid = 2*np.pi*np.arange(4096)/4096
    xgrid, _ = curve(grid)
    step = 2*np.pi/4096
    worst = 0.
    for t0, target in zip(theta, y):
        k = np.argmin(np.sum((xgrid-target)**2, axis=1))
        def objective(t):
            x, _ = curve(np.array([t]))
            return np.sum((x[0]-target)**2)
        result = minimize_scalar(objective, bounds=(grid[k]-step, grid[k]+step),
                                 method='bounded', options={'xatol': 1e-13})
        parameter_error = abs(np.angle(np.exp(1j*(result.x-t0))))
        distance_error = abs(np.sqrt(result.fun)-abs(b))
        if not result.success or parameter_error > 2e-6 or distance_error > 2e-10:
            raise RuntimeError('normal offset failed closest-point check')
        worst = max(worst, distance_error)
    return worst


def reconstruct(curve, theta, y, b, delta, n, p):
    f,g = densities(curve)
    s = single_layer(curve, f, y, delta, n, p, b)
    d = double_layer(curve, g, y, delta, n, p, closest_t=theta, b=b)
    return s, d


def run(output, ns, shape='starfish', offset_scale='delta'):
    curve = {'starfish': starfish, 'circle': circle}[shape]
    theta = 2*np.pi*np.arange(97)/97
    rows = []
    worst_geometry = 0.
    for p,q in [(3, 2/3), (5, 4/5), (7, 5/7)]:
        for n in ns:
            h = 2*np.pi/n
            delta = h**q
            for case, lam in [('inside', -.5), ('on', 0.), ('outside', .5)]:
                b = lam*(delta if offset_scale == 'delta' else h)
                y, exact = target_family(curve, theta, b)
                # The largest offset in each family checks the geometry used
                # for every subsequent, smaller normal offset.
                if n == ns[0] and b != 0:
                    worst_geometry = max(worst_geometry, check_closest_points(curve, theta, y, b))
                s,d = reconstruct(curve, theta, y, b, delta, n, p)
                sf,df = reconstruct(curve, theta, y, b, delta, 4*n, p)
                err = s+d-exact
                # Inspect each layer's quadrature change to avoid hiding
                # cancellation of quadrature errors in the sum.
                qs,qd = np.max(np.abs(s-sf)), np.max(np.abs(d-df))
                rows.append(dict(shape=shape, case=case, p=p, q=q, n=n, h=h,
                                 delta=delta, b=b, error_inf=np.max(np.abs(err)),
                                 error_rms=np.sqrt(np.mean(err**2)),
                                 resolved_error_inf=np.max(np.abs(sf+df-exact)),
                                 quadrature_s=qs, quadrature_d=qd,
                                 quadrature_sum=np.max(np.abs(s+d-sf-df)),
                                 single_norm=np.max(np.abs(s)), double_norm=np.max(np.abs(d)),
                                 exact_norm=np.max(np.abs(exact))))
    output.mkdir(parents=True, exist_ok=True)
    for case in ['inside', 'on', 'outside']:
        for p in [3, 5, 7]:
            previous = None
            for row in [r for r in rows if r['case']==case and r['p']==p]:
                row['rate_h'] = np.nan if previous is None else np.log(previous['error_inf']/row['error_inf'])/np.log(previous['h']/row['h'])
                row['rate_delta'] = row['rate_h']/row['q']
                previous = row
    with (output/'convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    report=['# Manufactured solution: combined single and double layers', '',
            f'Shape: {shape}.',
            'Representation: u = S[q_plus-q_minus] + D[u_minus-u_plus].',
            'On the curve the reference is (u_minus+u_plus)/2.',
            f'Near targets use b=±{offset_scale}/2; delta=h^q; h=2*pi/N.',
            'Fit: last three levels above 5e-13, with each-layer quadrature differences summing to less than 5% of total error.', '',
            '| Targets | p | Expected h order | Measured h order | Measured delta order | Fit N |',
            '|---|---:|---:|---:|---:|---|']
    for ax,case in zip(axes.flat,['inside','on','outside']):
        for p in [3,5,7]:
            group=[r for r in rows if r['case']==case and r['p']==p]
            hs=np.array([r['h'] for r in group])
            es=np.array([r['error_inf'] for r in group])
            q=group[0]['q']
            line,=ax.loglog(hs,es,'o-',label=f'p={p}, expected {p*q:g}')
            valid=[r for r in group if r['error_inf']>5e-13 and
                   r['quadrature_s']+r['quadrature_d']<.05*r['error_inf']][-3:]
            if len(valid)==3:
                slope=np.polyfit(np.log([r['h'] for r in valid]),np.log([r['error_inf'] for r in valid]),1)[0]
                report.append(f"| {case} | {p} | {p*q:.1f} | {slope:.3f} | {slope/q:.3f} | {', '.join(str(r['n']) for r in valid)} |")
                a=valid[-1]
                ax.loglog(hs,a['error_inf']*(hs/a['h'])**(p*q),'--',color=line.get_color(),alpha=.5)
            else:
                report.append(f'| {case} | {p} | {p*q:.1f} | insufficient resolved levels | | |')
        ax.set(title=case, xlabel='h = 2π/N',ylabel='maximum absolute error in S + D')
        ax.grid(which='both',alpha=.2)
        ax.legend(fontsize=9)
    ax=axes.flat[3]
    t=np.linspace(0,2*np.pi,601)
    x,_=curve(t)
    ax.plot(x[:,0],x[:,1],color='black',label='boundary')
    largest_offset = (2*np.pi/ns[0])**(2/3) if offset_scale == 'delta' else 2*np.pi/ns[0]
    for b in [-.5*largest_offset, .5*largest_offset]:
        y,_=target_family(curve,theta,b)
        ax.scatter(y[:,0],y[:,1],s=6,label='inside targets' if b<0 else 'outside targets')
    ax.set(aspect='equal', title=f'{shape}: coarsest p=3 target offsets',xlabel='x',ylabel='y')
    ax.legend(fontsize=8)
    fig.suptitle('Manufactured harmonic fields: combined single + double layer\nDashed lines: predicted rates')
    fig.savefig(output/'convergence.png',dpi=180)
    plt.close(fig)
    report+=['',f'Largest closest-distance check discrepancy: {worst_geometry:.3e}.',
             'Raw maximum/RMS errors, individual layer norms, quadrature estimates, and adjacent rates are in convergence.csv.']
    text='\n'.join(report)+'\n'
    (output/'summary.md').write_text(text)
    print(text)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',choices=['starfish','circle'],default='starfish')
    parser.add_argument('--output',type=Path,default=Path('results/manufactured'))
    parser.add_argument('--ns',type=int,nargs='+',default=[128,256,512,1024,2048,4096,8192])
    parser.add_argument('--offset-scale',choices=['delta','h'],default='delta')
    args=parser.parse_args()
    if min(args.ns)<4 or len(args.ns)<3 or any(b<=a for a,b in zip(args.ns,args.ns[1:])):
        parser.error('ns must contain at least three increasing integers >=4')
    run(args.output,args.ns,args.shape,args.offset_scale)
