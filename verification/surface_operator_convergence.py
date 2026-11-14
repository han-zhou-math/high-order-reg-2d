"""On-curve derivative operators: circle eigenfunctions and starfish references."""
import csv
from pathlib import Path
import numpy as np
from regbie.regularized_slp import circle, starfish
from regbie.regularized_dlp import normals
from regbie.manufactured_convergence import densities, field
from regbie.surface_operators import evaluate_surface


def starfish_reference(theta, n=4096):
    """Unregularized K and K' by smooth shifted trapezoidal quadrature.

    Compute normal numerators after removing the exactly tangential linear
    term in the Fourier representation of the curve. W follows from the
    independent manufactured normal-trace identity K'f+Wg=(q-+q+)/2.
    """
    theta=np.asarray(theta)
    step=2*np.pi/n
    s=step*(np.arange(n)+.5)
    s=(s+np.pi)%(2*np.pi)-np.pi
    t=theta[:,None]+s[None,:]
    z0=np.zeros(len(theta),complex)
    dz0=np.zeros(len(theta),complex)
    dz=np.zeros_like(t,dtype=complex)
    r=np.zeros_like(t,dtype=complex)
    remainder_y=np.zeros_like(t,dtype=complex)
    remainder_x=np.zeros_like(t,dtype=complex)
    for k,c in [(1,1.),(4,.1),(-2,.1)]:
        e0=c*np.exp(1j*k*theta)
        et=c*np.exp(1j*k*t)
        z0+=e0
        dz0+=1j*k*e0
        dz+=1j*k*et
        r+=e0[:,None]*np.expm1(1j*k*s)[None,:]
        remainder_y+=e0[:,None]*(np.expm1(1j*k*s)-1j*k*s)[None,:]
        remainder_x+=et*(np.expm1(-1j*k*s)+1j*k*s)[None,:]
    nx=-1j*dz/np.abs(dz)
    ny=-1j*dz0/np.abs(dz0)
    nxr=-np.real(np.conj(nx)*remainder_x)
    nyr=np.real(np.conj(ny[:,None])*remainder_y)
    r2=np.abs(r)**2
    f,g=densities(starfish)
    fv=f(t.ravel()).reshape(t.shape)
    gv=g(t.ravel()).reshape(t.shape)
    kp=np.sum(-nyr/r2*fv*np.abs(dz),axis=1)/n
    kval=np.sum(nxr/r2*gv*np.abs(dz),axis=1)/n
    x,dx=starfish(theta)
    qm=np.sum(field(x,'inside')[1]*normals(dx),axis=1)
    qp=np.sum(field(x,'outside')[1]*normals(dx),axis=1)
    return dict(K=kval,Kprime=kp,W=(qm+qp)/2-kp)


def benchmark(shape, theta):
    if shape=='circle':
        curve=circle
        f=g=lambda t: 1+.7*np.cos(3*t)
        g_dt=lambda t: -2.1*np.sin(3*t)
        exact=dict(K=np.full_like(theta,.5),Kprime=np.full_like(theta,.5),
                   W=1.05*np.cos(3*theta))
        discrepancy=0.
    else:
        curve=starfish
        f,g=densities(curve)
        def g_dt(t):
            x,dx=curve(t)
            return np.sum((field(x,'inside')[1]-field(x,'outside')[1])*dx,axis=1)
        exact=starfish_reference(theta,8192)
        lower=starfish_reference(theta,4096)
        discrepancy=max(np.max(np.abs(exact[k]-lower[k])) for k in exact)
        if discrepancy>2e-12:
            raise RuntimeError(f'reference has not converged: {discrepancy}')
    return curve,f,g,g_dt,exact,discrepancy


def run():
    theta=2*np.pi*np.arange(97)/97
    ns=[128,256,512,1024,2048,4096]
    rows=[]
    reference_errors={}
    for shape in ['circle','starfish']:
        curve,f,g,g_dt,exact,referr=benchmark(shape,theta)
        reference_errors[shape]=referr
        for p,q in [(3,2/3),(5,4/5),(7,5/7)]:
            for n in ns:
                h=2*np.pi/n
                delta=h**q
                value=evaluate_surface(curve,f,g,g_dt,theta,delta,n,p)
                fine=evaluate_surface(curve,f,g,g_dt,theta,delta,4*n,p)
                for op in value:
                    ref=exact['W'] if op.startswith('W') else exact[op]
                    error=np.max(np.abs(value[op]-ref))
                    rows.append(dict(shape=shape,p=p,q=q,n=n,h=h,delta=delta,
                                     operator=op,error_inf=error,
                                     quadrature_estimate=np.max(np.abs(value[op]-fine[op])),
                                     maue_identity_discrepancy=np.max(np.abs(fine['Wcorrected']-fine['Wmaue']))))
            print(f'Finished {shape}, p={p}',flush=True)
    out=Path('results/surface_operators')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'convergence.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report=['# On-surface normal derivatives of the regularized SLP kernel','',
            'K: source-normal derivative; Kprime: target-normal derivative.',
            'Wraw: mixed source/target normal derivative.',
            'Wcorrected: Wraw plus (nx dot ny)*Laplacian(G_delta).',
            'Wmaue: target tangential derivative of S_delta applied to dg/ds.',
            'All use the full on-surface order-p SLP kernel, with lambda=0 frozen.', '',
            '| Shape | Operator | p | fitted h rate | predicted for convergent operators | finest error | fit N |',
            '|---|---|---:|---:|---:|---:|---|']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for axs,shape in zip(axes,['circle','starfish']):
        for j,operators in enumerate([['K','Kprime'],['Wcorrected','Wmaue'],['Wraw']]):
            ax=axs[j]
            for p,color in zip([3,5,7],['C0','C1','C2']):
                for op,style in zip(operators,['o-','s--']):
                    group=[r for r in rows if (r['shape'],r['p'],r['operator'])==(shape,p,op)]
                    ax.loglog([r['h'] for r in group],[r['error_inf'] for r in group],style,
                              color=color,markersize=4,label=f'{op}, p={p}')
                    threshold=max(5e-12,100*reference_errors[shape])
                    valid=[r for r in group if r['error_inf']>threshold and r['quadrature_estimate']<.05*r['error_inf']][-3:]
                    q=group[0]['q']
                    if len(valid)==3:
                        slope=np.polyfit(np.log([r['h'] for r in valid]),np.log([r['error_inf'] for r in valid]),1)[0]
                        rate=f'{slope:.3f}'
                        fit=', '.join(str(r['n']) for r in valid)
                    else: rate,fit='—','insufficient resolved levels'
                    expected='diverges' if op=='Wraw' else f'{p*q:g}'
                    report.append(f"| {shape} | {op} | {p} | {rate} | {expected} | {group[-1]['error_inf']:.3e} | {fit} |")
            ax.set(title=f'{shape}: '+['K and adjoint K','hypersingular: corrected forms','hypersingular: direct derivative'][j],
                   xlabel='h = 2π/N',ylabel='maximum absolute error')
            ax.grid(which='both',alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('On-surface operators derived from the regularized single-layer kernel')
    fig.savefig(out/'convergence.png',dpi=180)
    plt.close(fig)
    max_identity=max(r['maue_identity_discrepancy'] for r in rows)
    report+=['',f'Starfish reference difference (4096 vs 8192): {reference_errors["starfish"]:.3e}.',
             f'Maximum corrected-Hessian / Maue difference at 4N: {max_identity:.3e}.',
             'Fit threshold: max(5e-12,100*reference discrepancy), plus quadrature below 5% of error.',
             'Circle references: K(1+0.7 cos 3t)=Kprime(1+0.7 cos 3t)=1/2; W(1+0.7 cos 3t)=1.05 cos 3t.',
             'Starfish K and Kprime references use unregularized smooth quadrature. W follows from the exact manufactured normal-trace identity.',
             'Add +g/2 or -g/2 to K for interior/exterior D traces; add -f/2 or +f/2 to Kprime for interior/exterior normal S traces. These exact jumps do not change the errors.',
             'The direct mixed derivative has a local O(1/delta) defect; the Laplacian correction removes it.']
    (out/'summary.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


if __name__=='__main__':
    run()
