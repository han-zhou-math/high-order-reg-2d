"""Read-only analysis of saved fixed-width runs (no numerical evaluations)."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .fixed_width import ROOT, save


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def aggregate(rows):
    groups={}
    for r in rows:
        key=(r['field'],float(r['kappa']),int(r['n_gamma']))
        groups.setdefault(key,[]).append(r)
    out=[]
    for (field,k,n),rs in sorted(groups.items()):
        phases=sorted(set(float(r['phase']) for r in rs))
        pe=[max(float(r['error_inf']) for r in rs if float(r['phase'])==p) for p in phases]
        out.append(dict(field=field,kappa=k,n_gamma=n,h_gamma=float(rs[0]['h_gamma']),
            h_phys_max=float(rs[0]['h_phys_max']),delta=float(rs[0]['delta']),
            error_inf=max(pe),quadrature_difference_inf=max(float(r['quadrature_difference_inf']) for r in rs),
            phase_min_error=min(pe),phase_max_error=max(pe),phase_ratio=max(pe)/min(pe),
            successive_rate='',window_rate=''))
    for r in out:
        prev=[s for s in out if s['field']==r['field'] and s['kappa']==r['kappa'] and s['n_gamma']<r['n_gamma']]
        if prev:
            s=prev[-1]
            r['successive_rate']=np.log(s['error_inf']/r['error_inf'])/np.log(r['n_gamma']/s['n_gamma'])
        if len(prev)>=2:
            window=prev[-2:]+[r]
            r['window_rate']=np.polyfit(np.log([s['h_gamma'] for s in window]),np.log([s['error_inf'] for s in window]),1)[0]
    return out


def production(label,field):
    if label=='circle':
        rows=read(Path('results/redesigned/circle_layers/errors.csv'))
        rows=[r for r in rows if int(r['order'])==7 and r['solution']=='circle_'+field]
    else:
        rows=read(Path('results/swapped_low_order/local/errors.csv'))+read(Path('results/swapped_low_order/local/p7_extension.csv'))
        rows=[r for r in rows if int(r['order'])==7 and r['field']==field and abs(float(r['b_over_delta']))<=.5]
    ns=sorted(set(int(r['n_gamma']) for r in rows))
    return ns,[max(float(r['error_inf']) for r in rows if int(r['n_gamma'])==n) for n in ns]


def main():
    summaries={}
    for label in ['circle','kite_laplace','starfish_stokes']:
        for name in ['errors','sweep']:
            path=ROOT/label/(name+'.csv')
            if not path.exists(): continue
            rows=aggregate(read(path)); save(ROOT/label/(name+'_summary.csv'),rows)
            summaries[label+'_'+name]=rows
            fields=sorted(set(r['field'] for r in rows))
            fig,axes=plt.subplots(1,len(fields),figsize=(6*len(fields),4.5),squeeze=False)
            for field,ax in zip(fields,axes[0]):
                rr=[r for r in rows if r['field']==field]
                if name=='errors':
                    for k in sorted(set(r['kappa'] for r in rr)):
                        ss=[r for r in rr if r['kappa']==k]
                        ax.loglog([r['h_gamma'] for r in ss],[r['error_inf'] for r in ss],'.-',label=f'κ={k:g}')
                    ns,es=production(label,field)
                    ax.loglog(2*np.pi/np.array(ns),es,'k--',label='production q=5/7')
                    ax.set_xlabel('parameter spacing hΓ')
                else:
                    rr=sorted(rr,key=lambda r:r['kappa'])
                    ax.semilogy([r['kappa'] for r in rr],[r['error_inf'] for r in rr],'o-',label='total error')
                    ax.semilogy([r['kappa'] for r in rr],[r['quadrature_difference_inf'] for r in rr],'s--',label='N versus 2N, fixed δ')
                    ax.set_xlabel('κ = δ / maximum physical spacing')
                ax.set_title(label.replace('_',' ')+' — '+field)
                ax.set_ylabel('maximum absolute error'); ax.grid(alpha=.25); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(ROOT/label/(name+'.png'),dpi=180); fig.savefig(ROOT/label/(name+'.pdf')); plt.close(fig)
    p=ROOT/'discrete_moments/moments.csv'
    if p.exists():
        rows=read(p); out=[]
        for order in [3,5,7]:
            for n in range(0,order-1,2):
                for k in sorted(set(float(r['kappa']) for r in rows)):
                    for subset in ['all','finite_sampled','off_surface']:
                        rr=[r for r in rows if int(r['p'])==order and int(r['n'])==n and float(r['kappa'])==k and (subset=='all' or r['status']=='finite') and (subset!='off_surface' or float(r['lambda_value'])>0)]
                        r=max(rr,key=lambda r:float(r['absolute_defect']))
                        out.append(dict(p=order,n=n,kappa=k,subset=subset,maximum=r['absolute_defect'],phase=r['phase'],lambda_value=r['lambda_value']))
        save(ROOT/'discrete_moments/maxima.csv',out)
        fig,axes=plt.subplots(1,3,figsize=(15,4.4))
        for order,ax in zip([3,5,7],axes):
            for n in range(0,order-1,2):
                for subset,style in [('finite_sampled','-'),('off_surface','--')]:
                    rr=[r for r in out if r['p']==order and r['n']==n and r['subset']==subset]
                    ax.semilogy([r['kappa'] for r in rr],[float(r['maximum']) for r in rr],style,label=f'n={n}, '+('finite phases' if subset=='finite_sampled' else 'λ>0'))
            ax.set_title(f'p={order}; full max D₀ = ∞'); ax.set_xlabel('κ'); ax.set_ylabel('maximum |Dₙ|'); ax.grid(alpha=.25); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(ROOT/'discrete_moments/defects.png',dpi=180); fig.savefig(ROOT/'discrete_moments/defects.pdf'); plt.close(fig)
    (ROOT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__': main()
