"""Validate the saved raw moment sums against an independent Fourier identity."""
import json
import mpmath as mp
from .fixed_width import ROOT
from .analyze_fixed_width import read


def main():
    mp.mp.dps=90
    path=ROOT/'discrete_moments'
    raw=read(path/'moments.csv')
    aliases=read(path/'full_kernel_alias.csv')
    lookup={(r['kappa'],r['lambda_value'],r['phase']):mp.mpf(r['defect'])
            for r in raw if r['p']=='7' and r['n']=='0'}
    errors=[]
    for r in aliases:
        k,lam,theta=map(mp.mpf,[r['kappa'],r['lambda_value'],r['phase']])
        if lam==0 and theta==0: continue
        a=mp.exp(-2*mp.pi*k*lam)
        full=lookup[r['kappa'],r['lambda_value'],r['phase']]+mp.log(1-2*a*mp.cos(2*mp.pi*theta)+a*a)/k
        errors.append(abs(full-mp.mpf(r['full_kernel_alias_defect'])))
    assert max(errors)<mp.mpf('1e-80')
    report=dict(raw_plus_log_vs_fourier_max_absolute_discrepancy=mp.nstr(max(errors),30),
                precision=90,comparison_count=len(errors),
                limitation='Absolute identity check does not resolve aliases smaller than 1e-85; those use the factored Fourier formula.')
    (path/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)

if __name__=='__main__':main()
