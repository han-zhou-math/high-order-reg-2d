"""Compare direct source differentiation with the density-subtracted DLP."""
import csv
from pathlib import Path
import numpy as np
from regbie.regularized_slp import circle, starfish, single_layer, coefficients
from regbie.regularized_dlp import double_layer
from regbie.manufactured_convergence import densities, target_family, check_closest_points
from scipy.integrate import quad


def leading_bias(p, lam):
    """Magnitude coefficient for raw DLP error: sign(b)*g(x0)*B_p(lambda)."""
    cs = coefficients(p, lam)
    def integrand(t):
        u = t*t+lam*lam
        polynomial = 1+sum((-1)**k*c*u**k for k,c in enumerate(cs, 1))
        return np.exp(-u)*polynomial/u
    return lam/np.pi*quad(integrand, 0, np.inf, epsabs=2e-13)[0]


def surface_rates(rows):
    report = ['', '## On-curve fitted h rates', '',
              '| Benchmark | Method | p=3 | p=5 | p=7 |',
              '|---|---|---:|---:|---:|']
    for benchmark in ['circle_D', 'starfish_S_plus_D']:
        for method in ['raw', 'subtracted']:
            rates = []
            for p in [3,5,7]:
                valid = [r for r in rows if r['benchmark']==benchmark and r['case']=='on'
                         and r['method']==method and int(r['p'])==p
                         and float(r['error_inf'])>5e-13
                         and float(r['quadrature_estimate'])<.05*float(r['error_inf'])][-3:]
                if len(valid)==3:
                    slope=np.polyfit(np.log([float(r['h']) for r in valid]),
                                     np.log([float(r['error_inf']) for r in valid]),1)[0]
                    rates.append(f'{slope:.3f}')
                else:
                    rates.append('insufficient levels')
            report.append('| '+benchmark+' | '+method+' | '+' | '.join(rates)+' |')
    return report


def run():
    theta = 2*np.pi*np.arange(97)/97
    ns = [128,256,512,1024,2048,4096]
    cases = [('on', 0., 'delta'), ('inside_scaled', -.5, 'delta'),
             ('outside_scaled', .5, 'delta'), ('inside_fixed', -.05, 'fixed'),
             ('outside_fixed', .05, 'fixed')]
    rows = []
    for benchmark, curve in [('circle_D', circle), ('starfish_S_plus_D', starfish)]:
        f, jump_g = densities(curve)
        g = (lambda t: 1+.7*np.cos(3*t)) if benchmark == 'circle_D' else jump_g
        for p,q in [(3,2/3),(5,4/5),(7,5/7)]:
            for case, offset, scale in cases:
                for n in ns:
                    h = 2*np.pi/n
                    delta = h**q
                    b = offset*(delta if scale=='delta' else 1.)
                    y, exact = target_family(curve, theta, b)
                    if n == ns[0] and b != 0:
                        check_closest_points(curve, theta, y, b)
                    if benchmark == 'circle_D':
                        radius = 1+b
                        if b<0: exact = 1+.35*radius**3*np.cos(3*theta)
                        elif b>0: exact = -.35*radius**(-3)*np.cos(3*theta)
                        else: exact = np.full_like(theta, .5)
                    s = single_layer(curve,f,y,delta,n,p,b) if benchmark!='circle_D' else 0.
                    sf = single_layer(curve,f,y,delta,4*n,p,b) if benchmark!='circle_D' else 0.
                    for subtract in [False, True]:
                        d = double_layer(curve,g,y,delta,n,p,closest_t=theta,b=b,subtract=subtract)
                        df = double_layer(curve,g,y,delta,4*n,p,closest_t=theta,b=b,subtract=subtract)
                        err = s+d-exact
                        row = dict(benchmark=benchmark, case=case, method='subtracted' if subtract else 'raw',
                                   p=p,q=q,n=n,h=h,delta=delta,b=b,
                                   error_inf=np.max(np.abs(err)),
                                   quadrature_estimate=np.max(np.abs(d-df))+np.max(np.abs(s-sf)))
                        if scale=='delta' and offset!=0 and not subtract:
                            predicted = np.sign(b)*g(theta)*leading_bias(p,abs(offset))
                            row['predicted_plateau'] = np.max(np.abs(predicted))
                            row['bias_residual'] = np.max(np.abs(err-predicted))
                        else:
                            row['predicted_plateau'] = np.nan
                            row['bias_residual'] = np.nan
                        rows.append(row)
            print(f'Finished {benchmark}, p={p}', flush=True)
    out=Path('results/differentiated_kernel')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'convergence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report=['# Direct differentiation of the regularized single-layer kernel', '',
            'Raw: integrate dG_delta/dn_x against g. Subtracted: integrate against g-g(x0), then add chi*g(x0).',
            'The kernels are identical; only the density subtraction and exact constant term differ.',
            'delta=h^q; q=2/3,4/5,5/7 for p=3,5,7; N=128,...,4096; 97 target angles.', '',
            '| Benchmark | Target | p | raw error at finest N | subtracted error at finest N | predicted raw plateau |',
            '|---|---|---:|---:|---:|---:|']
    for benchmark in ['circle_D','starfish_S_plus_D']:
        for case,_,_ in cases:
            for p in [3,5,7]:
                pair=[r for r in rows if (r['benchmark'],r['case'],r['p'],r['n'])==(benchmark,case,p,ns[-1])]
                raw,sub=pair
                report.append(f"| {benchmark} | {case} | {p} | {raw['error_inf']:.3e} | {sub['error_inf']:.3e} | {raw['predicted_plateau']:.3e} |")
    report += surface_rates(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for axs,benchmark in zip(axes,['circle_D','starfish_S_plus_D']):
        for ax,case in zip(axs,['on','outside_scaled','outside_fixed']):
            for p,color in zip([3,5,7],['C0','C1','C2']):
                for method,style in [('raw','o-'),('subtracted','s--')]:
                    group=[r for r in rows if (r['benchmark'],r['case'],r['p'],r['method'])==(benchmark,case,p,method)]
                    ax.loglog([r['h'] for r in group],[r['error_inf'] for r in group],style,
                              color=color,markersize=4,label=f'p={p}, {method}')
            ax.set(title=f'{benchmark}: '+{'on':'b=0','outside_scaled':'b=δ/2','outside_fixed':'b=0.05'}[case],
                   xlabel='h = 2π/N',ylabel='maximum absolute error')
            ax.grid(which='both',alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('Differentiated regularized kernel: raw versus density-subtracted integration')
    fig.savefig(out/'convergence.png',dpi=180)
    plt.close(fig)
    report += ['', 'For b=±lambda*delta, the predicted plateau is max|g(x0)|*|B_p(lambda)|, where',
               'B_p(lambda)=lambda/(2*pi) integral_R Q_p(t²+lambda²)/(t²+lambda²) dt.',
               'The CSV includes the maximum difference between the signed numerical error and this leading bias.',
               'At b=0 the full kernels converge; the special one-fewer-term kernels are not used.',
               'At fixed nonzero b, raw and subtracted errors eventually approach roundoff as delta decreases.']
    (out/'summary.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


if __name__=='__main__':
    run()
