"""Isolated fixed-width diagnostics; never modifies production data.

Run with --stage circle/complete/sweep/moments. All errors maximize over
97 distributed closest-point parameters for each separately saved phase/offset.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from regbie.regularized_slp import single_layer
from regbie.regularized_dlp import double_layer
from .bie_evaluators import (PeriodicSamples, curve_adapter,
    eval_laplace_regularized, eval_pressure_regularized, eval_stokes_regularized)
from .geometries import circle_geometry
from .manufactured_laplace import circle_density, exact_circle_single, exact_circle_double
from .swapped_low_order import LAPLACE_KITE, STOKES_STARFISH

ROOT = Path('results/fixed_width')


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate(label, problem, targets, theta, b, delta, n, side):
    if label == 'circle':
        curve = curve_adapter(circle_geometry)
        return {'single': single_layer(curve, circle_density, targets, delta, n, 7, b=b),
                'double': double_layer(curve, circle_density, targets, delta, n, 7,
                                      closest_t=theta, b=b, surface=(side=='average'))}
    t = 2*np.pi*np.arange(n)/n
    if label == 'kite_laplace':
        f, g = problem.boundary_densities(t)
        return {'laplace': eval_laplace_regularized(problem.geometry,
            PeriodicSamples(f), PeriodicSamples(g), targets, theta, b, delta, n, 7)}
    q, f = problem.boundary_data(t)
    return {'velocity': eval_stokes_regularized(problem.geometry, q, f, targets,
                                               theta, b, delta, 7),
            'pressure': eval_pressure_regularized(problem.geometry, q, f, targets,
                                                 theta, b, delta, 7)}


def local(stage):
    labels = ['circle'] if stage == 'circle' else ['kite_laplace', 'starfish_stokes']
    ns = [128*2**j for j in range(8)] if stage=='circle' else [1024*2**j for j in range(5)]
    ks = [1,1.5,2,3,4,6,8] if stage=='circle' else [2,3,4,6]
    if stage in ['sweep','phase']:
        ns, ks = [4096], [.75,1,1.5,2,2.5,3,4,5,6,8]
    # Quarter-step is a supplementary check motivated by circle DLP aliasing.
    if stage=='phase': ks=[1,1.5,2,3,4]
    for label in labels:
        problem = {'circle': None, 'kite_laplace':LAPLACE_KITE,
                   'starfish_stokes': STOKES_STARFISH}[label]
        geom = circle_geometry if problem is None else problem.geometry
        rows = []
        filename = ROOT/label/({'sweep':'sweep.csv','phase':'phase_quarter.csv'}.get(stage,'errors.csv'))
        for n in ns:
            h = 2*np.pi/n
            x = geom(h*np.arange(n))[0]
            hp = np.max(np.linalg.norm(np.roll(x,-1,axis=0)-x,axis=1))
            indices = np.unique(np.floor(n*(np.arange(97)+.37)/97).astype(int))
            for k in ks:
                delta = k*(h if label=='circle' else hp)
                for phase in ([.25] if stage=='phase' else ([0,.25,.5] if label=='circle' else [0,.5])):
                    theta = (indices+phase)*h
                    x0, _, _, normal, *_ = geom(theta)
                    for lam in [-.5,0,.5]:
                        b = np.full(len(theta),lam*delta)
                        targets = x0+b[:,None]*normal
                        side = 'average' if lam==0 else ('inside' if lam<0 else 'outside')
                        val = evaluate(label,problem,targets,theta,b,delta,n,side)
                        # Same delta, targets and closest points: isolates quadrature.
                        check = evaluate(label,problem,targets,theta,b,delta,2*n,side)
                        if label=='circle':
                            exact = {'single':exact_circle_single(1+lam*delta,theta),
                                     'double':exact_circle_double(1+lam*delta,theta,side)}
                        elif label=='kite_laplace':
                            exact = {'laplace':problem.exact(targets,side)}
                        else:
                            ue,pe = problem.exact(targets,side)
                            exact = {'velocity':ue,'pressure':pe}
                        for field,v in val.items():
                            mag = lambda a: np.linalg.norm(a,axis=-1) if field=='velocity' else np.abs(a)
                            e = mag(v-exact[field]); d = mag(v-check[field])
                            rows.append(dict(geometry=label,field=field,n_gamma=n,h_gamma=h,
                                h_phys_max=hp,delta=delta,kappa=k,delta_over_h_phys=delta/hp,
                                p=7,phase=phase,b_over_delta=lam,target_count=len(theta),
                                error_inf=float(e.max()),quadrature_difference_inf=float(d.max()),
                                maximum_target_parameter=float(theta[np.argmax(e)])))
                save(filename,rows)
                print(f'{stage} {label} N={n} k={k}',flush=True)


def moments():
    import mpmath as mp
    mp.mp.dps = 90
    rows = []
    for kf in [.75,1,1.25,1.5,2,2.5,3,3.5,4,5,6,8]:
        k = mp.mpf(str(kf))
        for lf in [0,.5,1,2]:
            lam = mp.mpf(str(lf)); z=lam*lam
            h0=2*(1-mp.sqrt(mp.pi)*lam*mp.exp(z)*mp.erfc(lam))
            h1=(1-z*h0)/mp.mpf('1.5'); h2=(1-z*h1)/mp.mpf('2.5')
            d1=h0-h1; d2=h0-2*h1+h2
            cs=[]
            for m in [1,2,3]:
                c3=d2/2 if m==3 else mp.mpf(0)
                c2=d1+(2*z+5)*c3 if m>=2 else mp.mpf(0)
                c1=h0+(z+mp.mpf('1.5'))*c2-(z*z+3*z+mp.mpf('3.75'))*c3
                cs.append((c1,c2,c3))
            for ph in range(33):
                theta=mp.mpf(ph)/64
                sums={(m,n):mp.mpf(0) for m in [1,2,3] for n in range(0,2*m,2)}
                # Add symmetric shells until every shell contribution < 1e-95
                # beyond |t|=12. Three consecutive small shells are required.
                small=0; j=0
                while True:
                    largest=mp.mpf(0)
                    for jj in ([0] if j==0 else [-j,j]):
                        t=(jj-theta)/k; u=t*t+z
                        if u==0:
                            for m in [1,2,3]: sums[m,0]=mp.inf
                            continue
                        e=mp.exp(-u); e1=mp.e1(u)
                        for m,(c1,c2,c3) in enumerate(cs,1):
                            s=e1+e*(-c1+c2*(1+u)-c3*(2+2*u+u*u))
                            for n in range(0,2*m,2):
                                term=t**n*s/k
                                sums[m,n]+=term
                                largest=max(largest,abs(term))
                    small=small+1 if j/k>12 and largest<mp.mpf('1e-95') else 0
                    if small>=3: break
                    j+=1
                for phase in ([ph] if ph in [0,32] else [ph,64-ph]):
                    for (m,n),v in sums.items():
                        rows.append(dict(kappa=kf,lambda_value=lf,phase=phase/64,
                            p=2*m+1,m=m,n=n,defect=mp.nstr(v,85),absolute_defect=mp.nstr(abs(v),85),
                            shell_limit=j,decimal_precision=90,
                            status='singular' if mp.isinf(v) else 'finite'))
            save(ROOT/'discrete_moments'/'moments.csv',rows)
        print(f'moments k={kf}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['moments','circle','complete','sweep','phase'],required=True)
    args=parser.parse_args()
    for folder in ['discrete_moments','circle','kite_laplace','starfish_stokes','allgrid']:
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    if args.stage=='moments': moments()
    else: local(args.stage)
