"""Small optional all-grid comparison after the local fixed-width diagnostics."""
import json
import numpy as np
from .fixed_width import ROOT, evaluate
from .swapped_low_order import LAPLACE_KITE, STOKES_STARFISH
from .matched_allgrid import run, spacing
from .allgrid import closest_points


def main():
    for label,problem in [('kite_laplace',LAPLACE_KITE),('starfish_stokes',STOKES_STARFISH)]:
        for k in [2,3]:
            for m in [128,256,512,1024]:
                path=ROOT/'allgrid'/f'{label}_k{k}_M{m}.json'
                if path.exists(): continue
                h=(problem.box[1]-problem.box[0])/(m+1)
                pars=spacing(problem.geometry,h); n=pars['n_gamma']; delta=k*h
                report,fields=run(problem,m,kappa=k,delta_override=delta,n_override=n)
                targets=np.stack((fields['xx'][fields['halo']],fields['yy'][fields['halo']]),axis=-1)
                # Evenly sample halo ordering, and explicitly include error maxima.
                ii=list(np.unique(np.linspace(0,len(targets)-1,97).astype(int)))
                names=['error'] if label=='kite_laplace' else ['pressure_error','velocity_error']
                for name in names:
                    e=fields[name][fields['halo']]
                    mag=np.linalg.norm(e,axis=-1) if e.ndim==2 else abs(e)
                    ii.append(int(np.argmax(mag)))
                targets=targets[np.unique(ii)]
                theta,b=closest_points(targets,problem.geometry,problem.inside)
                v=evaluate(label,problem,targets,theta,b,delta,n,'mixed')
                w=evaluate(label,problem,targets,theta,b,delta,2*n,'mixed')
                report['quadrature_check_target_count']=len(targets)
                report['quadrature_check_includes_direct_maxima']=True
                report['quadrature_difference_inf']={field:float(np.max(np.linalg.norm(a-w[field],axis=-1) if field=='velocity' else np.abs(a-w[field]))) for field,a in v.items()}
                path.write_text(json.dumps(report,indent=2)+'\n')
                # Error fields only; no exact-solution grid plots.
                np.savez_compressed(path.with_suffix('.npz'),**fields)
                print(label,k,m,report['quadrature_difference_inf'],flush=True)

if __name__=='__main__':main()
