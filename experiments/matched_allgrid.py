"""Matched physical grids and instrumented exact-direct/BIE reconstructions."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.fft import next_fast_len
from .allgrid import (grid_coordinates, classify_crossing, closest_points,
    _boundary_points, _unregularized_laplace, _unregularized_stokes,
    LAPLACE_OFFSETS, GRADIENT_OFFSETS, stencil_halo, nine_point_laplacian,
    dst_solve_nine_point, grad4)
from .allgrid import StokesProblem
from .bie_evaluators import (PeriodicSamples, eval_laplace_regularized,
    eval_pressure_regularized, eval_stokes_regularized)
from .boundary_data import pressure_densities
from .problems import LAPLACE_STARFISH, STOKES_KITE

ROOT = Path('results/matched_allgrid')
LEVELS = (64,128,256,512,768,1024)


def spacing(geometry, h, n=None, ratio=1.):
    if h <= 0 or ratio <= 0:
        raise ValueError('grid spacings must be positive')
    automatic = n is None
    if n is None:
        speed = geometry(np.linspace(0,2*np.pi,16384,endpoint=False))[1].max()
        n = next_fast_len(int(np.ceil(2*np.pi*speed/(ratio*h))))
        while n % 2: n = next_fast_len(n+1)
    for attempt in range(8):
        x = geometry(2*np.pi*np.arange(n)/n)[0]
        chord = np.linalg.norm(np.roll(x,-1,axis=0)-x,axis=1)
        if not automatic or .8 <= chord.max()/(ratio*h) <= 1.2:
            break
        n = next_fast_len(max(4, int(np.ceil(n*chord.max()/(ratio*h)))))
        while n % 2:
            n = next_fast_len(n+1)
    else:
        raise ValueError('unable to match discrete chord spacing')
    return dict(n_gamma=int(n), chord_max=float(chord.max()),
                chord_mean=float(chord.mean()), spacing_ratio=float(chord.max()/h))


def norms(v,h,vector=False):
    v = np.linalg.norm(v,axis=-1) if vector else np.abs(v)
    return dict(inf=float(v.max()), l2=float(h*np.sqrt(np.sum(v*v))))


def run(problem,m,mode='bie',rule='matched',kappa=1.,delta_override=None,
        n_override=None):
    start=time.perf_counter(); sto=isinstance(problem, StokesProblem)
    xx,yy,h=grid_coordinates(m,problem.box); pts=np.stack((xx,yy),axis=-1)
    pars=spacing(problem.geometry,h,n=n_override if n_override is not None else (8*m if rule=='old8m' else None),
                 ratio=.5 if rule=='twice' else 1.)
    n=pars['n_gamma']; delta=kappa*h**(5/7) if delta_override is None else float(delta_override)
    if delta <= 0:
        raise ValueError('smoothing width must be positive')
    cache=ROOT/'geometry_cache'/f'{problem.name}_{m}.npz'
    if cache.exists():
        masks=np.load(cache);cd=masks['cd'];cg=masks['cg']
    else:
        cd=classify_crossing(xx,yy,h,LAPLACE_OFFSETS,problem.inside)
        cg=classify_crossing(xx,yy,h,GRADIENT_OFFSETS,problem.inside) if sto else cd.copy()
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,cd=cd,cg=cg)
    cu=cd|cg; halo=stencil_halo(cu); bm,bpts=_boundary_points(xx,yy)
    exact=problem.exact(pts); ue,pe=exact if sto else (None,exact)
    direct_exact=pe[halo]; u_direct_exact=ue[halo] if sto else None
    report=dict(m_grid=m,h_c=h,geometry=problem.name,mode=mode,rule=rule,
        kappa=kappa,delta=delta,delta_over_h=delta/h,
        delta_over_chord=delta/pars['chord_max'],**pars,
        direct_targets=int(halo.sum()),crossing_count=int(cd.sum()),
        gradient_crossing_count=int(cg.sum()),union_count=int(cu.sum()))
    if mode=='bie':
        t=2*np.pi*np.arange(n)/n
        ct,b=closest_points(pts[halo],problem.geometry,problem.inside)
        if sto:
            q,f=problem.boundary_data(t); pd=pressure_densities(problem.geometry,q,f)
            direct=eval_pressure_regularized(problem.geometry,q,f,pts[halo],ct,b,delta,7)
            ud=eval_stokes_regularized(problem.geometry,q,f,pts[halo],ct,b,delta,7)
            pb=_unregularized_laplace(problem.geometry,PeriodicSamples(pd.phi),PeriodicSamples(pd.psi),bpts,n,-1.)
            ub=_unregularized_stokes(problem.geometry,q,f,bpts)
        else:
            f,g=problem.boundary_densities(t); fi,gi=PeriodicSamples(f),PeriodicSamples(g)
            direct=eval_laplace_regularized(problem.geometry,fi,gi,pts[halo],ct,b,delta,n,7)
            pb=_unregularized_laplace(problem.geometry,fi,gi,bpts,n)
    else:
        direct=direct_exact.copy(); pb=pe[bm]
        if sto: ud=u_direct_exact.copy(); ub=ue[bm]
    report['direct_seconds']=time.perf_counter()-start
    def reconstruct(d,exact_field,bc,cross):
        full=np.full(xx.shape,np.nan); full[halo]=d
        ld=nine_point_laplacian(full,h); le=nine_point_laplacian(exact_field,h)
        rhs=np.zeros((m,m));rhs[cross]=ld[cross]
        boundary=np.zeros(xx.shape);boundary[bm]=bc
        solved=dst_solve_nine_point(rhs,boundary,h)
        hybrid=solved.copy();hybrid[halo]=d
        return solved,hybrid,ld-le,full
    ps,pa,source_error,pfull=reconstruct(direct,pe,pb,cd)
    report['pressure' if sto else 'laplace']=norms(pa-pe,h)
    report['direct_pressure' if sto else 'direct_laplace']=norms(direct-direct_exact,h)
    report['crossing_direct_pressure' if sto else 'crossing_direct_laplace']=norms((pfull-pe)[1:-1,1:-1][cd],h)
    report['pressure_source' if sto else 'laplace_source']=norms(source_error[cd],h)
    report['outer_pressure' if sto else 'outer_laplace']=norms(pb-pe[bm],h)
    fields=dict(xx=xx,yy=yy,error=pa-pe,c_delta=cd,c_gradient=cg,halo=halo)
    if sto:
        safe=~cu; ge=problem.exact_gradient(pts[1:-1,1:-1])
        gp=grad4(ps,h)
        if mode=='exact_gradient': gp[safe]=ge[safe]
        report['pressure_reconstructed']=norms(ps-pe,h)
        report['gradient']=norms((gp-ge)[safe],h,True)
        report['direct_velocity']=norms(ud-u_direct_exact,h,True)
        uf=np.full(xx.shape+(2,),np.nan);uf[halo]=ud
        lu=np.stack([nine_point_laplacian(uf[...,k],h) for k in range(2)],axis=-1)
        lue=np.stack([nine_point_laplacian(ue[...,k],h) for k in range(2)],axis=-1)
        report['velocity_source_crossing']=norms((lu-lue)[cu],h,True)
        report['crossing_direct_velocity']=norms((uf-ue)[1:-1,1:-1][cu],h,True)
        rhs=np.empty((m,m,2));rhs[cu]=lu[cu];rhs[safe]=gp[safe]/problem.mu
        assert not np.any(safe & cg)
        ua=[]
        for k in range(2):
            bc=np.zeros(xx.shape);bc[bm]=ub[:,k]
            u=dst_solve_nine_point(rhs[...,k],bc,h);u[halo]=ud[:,k];ua.append(u)
        ua=np.stack(ua,-1)
        report['velocity']=norms(ua-ue,h,True)
        fields.update(pressure_error=pa-pe,velocity_error=ua-ue)
        rows=[]
        for target in (0.,h,-h,2*h,-2*h,.1,-.1):
            j=int(np.argmin(abs(yy[:,0]-target)));e=(pa-pe)[j]
            rows.append(dict(requested_y=target,actual_y=float(yy[j,0]),row=j,
                             max_abs=float(abs(e).max()),zeros=int(np.count_nonzero(e==0)),
                             finite=bool(np.isfinite(e).all())))
        report['stripe_rows']=rows
        fields['profile_rows']=np.array([r['row'] for r in rows])
        fields['pressure_profiles']=(pa-pe)[fields['profile_rows']]
    report['total_seconds']=time.perf_counter()-start
    return report,fields


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='exact_direct',choices=['bie','exact_direct','exact_gradient'])
    ap.add_argument('--problem',choices=['laplace','stokes'],required=True)
    ap.add_argument('--levels',nargs='+',type=int,default=LEVELS)
    ap.add_argument('--rule',default='matched',choices=['matched','twice','old8m'])
    ap.add_argument('--kappa',type=float,default=1.)
    ap.add_argument('--run',action='store_true',help='accepted for compatibility with production drivers')
    a=ap.parse_args(argv);problem=LAPLACE_STARFISH if a.problem=='laplace' else STOKES_KITE
    out=ROOT/f'{a.problem}_{a.mode}_{a.rule}_k{a.kappa:g}';out.mkdir(parents=True,exist_ok=True)
    for m in a.levels:
        dest=out/f'M{m}.json'
        if dest.exists():continue
        r,f=run(problem,m,a.mode,a.rule,a.kappa)
        dest.write_text(json.dumps(r,indent=2)+'\n')
        if m==max(a.levels):np.savez_compressed(out/f'fields_M{m}.npz',**f)
        print(json.dumps(r),flush=True)

if __name__=='__main__':main()
