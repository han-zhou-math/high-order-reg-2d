"""Build the diagnostic report and overview figures exclusively from saved data."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .fixed_width import ROOT, save
from .analyze_fixed_width import read, production


def table(headers,rows):
    headers=[str(v).replace('|', r'\|') for v in headers]
    rows=[[str(v).replace('|', r'\|') for v in row] for row in rows]
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def fmt(x):return f'{float(x):.3e}'


def main():
    ss=json.loads((ROOT/'summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,label,field,title in zip(axes,['kite_laplace','starfish_stokes','starfish_stokes'],['laplace','velocity','pressure'],['Complete Laplace — kite','Stokes velocity — starfish','Stokes pressure — starfish']):
        for k in [2,3,4,6]:
            rr=[r for r in ss[label+'_errors'] if r['field']==field and r['kappa']==k]
            ax.loglog([r['h_gamma'] for r in rr],[r['error_inf'] for r in rr],'o-',label=f'κ={k}')
        ns,es=production(label,field);ax.loglog(2*np.pi/np.array(ns),es,'k--',label='production q=5/7')
        ax.set(title=title,xlabel='parameter spacing hΓ',ylabel='maximum absolute error');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'complete_comparison.png',dpi=180);fig.savefig(ROOT/'complete_comparison.pdf');plt.close(fig)
    aliases=read(ROOT/'discrete_moments/full_kernel_alias.csv')
    aa=[]
    for k in sorted(set(float(r['kappa']) for r in aliases)):
        r=max([r for r in aliases if float(r['kappa'])==k],key=lambda r:float(r['absolute_defect']))
        aa.append(dict(kappa=k,maximum=r['absolute_defect'],phase=r['phase'],lambda_value=r['lambda_value']))
    save(ROOT/'discrete_moments/full_kernel_alias_maxima.csv',aa)
    fig,ax=plt.subplots(figsize=(6,4.5))
    ax.semilogy([r['kappa'] for r in aa],[float(r['maximum']) for r in aa],'o-')
    ax.set(xlabel='κ',ylabel='maximum full-kernel alias defect',title='Smooth combined kernel, p=7 (90-digit arithmetic)');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/'discrete_moments/full_kernel_alias.png',dpi=180);plt.close(fig)
    text=['# Fixed-width regularization: diagnostic report\n',
'''The tests support **κ between 2 and 4 as a practical range for these problems**, with κ=2–3 particularly attractive for all-grid work. This is a discrete-resolution study, not an additional verification of the continuous p=7 theorem. The production runs, geometries, densities, and kernel implementations have not been changed.

## Scope and reproducibility

All local tests use p=7, signed offsets b/δ = −1/2, 0, +1/2, and maxima over 97 distributed closest-point parameters. For each N, indices are floor(N(j+0.37)/97); closest-point parameters are (index+phase)2π/N. This retains the existing normal-offset construction while imposing actual quadrature-grid phases. Circle phases are 0, 1/4, 1/2; complete-problem convergence phases are 0, 1/2, supplemented by quarter-step checks at N=4096. These are sampled maxima, not certified suprema over the entire curve.

Circle δ=κ(2π/N). Kite and starfish δ=κ max_j |x_{j+1}−x_j|, using the original analytic parameterizations. Every local error has a separate N-versus-2N check at identical δ and targets. Quadrature differences near roundoff measure floating-point/spectral differentiation effects too. Production comparison curves read existing data, with offsets restricted to the same three values; their original 97 target angles differ slightly from the phase-controlled samples here.

All CSVs preserve phase, offset, geometry, spacings, width, order, target count, maximum-error parameter, and quadrature difference. `errors_summary.csv` contains robust maxima, successive rates in hΓ, and trailing three-level log–log fits. With fixed κ, hΓ and maximum physical spacing have the same asymptotic scaling. No global fit is used. The optional p=3,5 circle and N=32768 extensions were omitted because p=7 already resolves the relevant crossover or floating-point floor.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width --stage moments
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width_alias
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width --stage circle
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width --stage complete
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width --stage sweep
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width --stage phase
OPENBLAS_NUM_THREADS=1 python3 -m experiments.fixed_width_allgrid
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/reg-bie-mpl python3 -m experiments.analyze_fixed_width
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/reg-bie-mpl python3 -m experiments.report_fixed_width
```

## A. Discrete moments: an essential correction to the proposed interpretation

The requested raw smoother satisfies s_m(u)=−log(u)+O(1). Therefore D₀(κ,0,0)=+∞ for every κ and every tested order. **The requested maximum over all phases and offsets is infinite**, and its on-surface supremum remains infinite even if the single aligned phase is removed. It cannot be plotted as a finite decaying resolution measure. For n=2,4, the diagonal limit tⁿs_m(t²)=0 is used.

The requested sums were nevertheless computed for p=3,5,7; all 64 phases; λ=0,0.5,1,2; and all 12 requested κ values. Computation uses 90 decimal digits, symmetric shells, and adaptive truncation: three consecutive shells below 10⁻⁹⁵ after |t|>12. Symmetry supplies phases 1−θ. Raw infinities are saved explicitly. Separate finite-sampled and off-surface maxima are labeled as such. See `discrete_moments/moments.csv` and `maxima.csv`; the latter records the maximizing phase and λ for every order/moment/κ/subset.

For large κ, the finite sampled on-surface D₀ maximum behaves like 4.64281529/κ for this particular 64-phase set. D₂ and D₄ decay algebraically, approximately κ⁻³ and κ⁻⁵, respectively. The finite D₀ maximum is at θ=1/64 or 63/64, λ=0. Higher-moment maxima eventually occur at θ=0, λ=0. Off-surface defects decay exponentially with κλ; their largest tested values are generally at λ=0.5. The full CSV captures small-κ changes in maximizing phase. These facts reflect the unsmoothed logarithm, not inadequate evaluation of the smooth implemented kernel.

Selected p=7 raw defects (D₀ excludes the singular aligned sample in its finite column):
''']
    mm=read(ROOT/'discrete_moments/maxima.csv')
    rr=[]
    for k in [1,2,4,8]:
        vals=[]
        for n,sub in [(0,'finite_sampled'),(0,'off_surface'),(2,'all'),(4,'all')]:
            vals.append(next(r['maximum'] for r in mm if int(r['p'])==7 and int(r['n'])==n and float(r['kappa'])==k and r['subset']==sub))
        rr.append([k]+[fmt(v) for v in vals])
    text.append(table(['κ','finite sampled max |D₀|','λ>0 max |D₀|','max |D₂|','max |D₄|'],rr))
    text.append(r'''
![Raw moment diagnostics](discrete_moments/defects.png)

### Supplement: aliasing of the smooth combined kernel

The implemented single-layer kernel contains log(t²+λ²)+s_m(t²+λ²;λ), not s_m alone. For λ>0, the local full-kernel alias defect is

\[
A_0=D_0+\frac1\kappa\log\left(1-2e^{-2\pi\kappa\lambda}\cos(2\pi\theta)+e^{-4\pi\kappa\lambda}\right).
\]

This also holds at λ=0 for nonaligned phases; at the aligned phase the two infinities must cancel analytically. We evaluate the combined defect directly in Fourier space for all phases. With ω≠0, the Fourier transform of log(t²+λ²)+E₁(t²+λ²) is

\[
-\sqrt\pi\int_0^1 a^{-3/2}e^{-\lambda^2a-\omega^2/(4a)}\,da.
\]

The Gaussian correction transforms analytically. Factoring out exp(−ω²/4) before integrating preserves relative accuracy even for defects below 10⁻²⁶⁰. Summing the nonzero alias modes gives A₀. A separate comparison with the raw sum plus the analytic logarithmic defect agrees to 4.89×10⁻⁸⁵ in absolute error over all nonsingular sampled cases. This absolute cross-check cannot resolve the smallest aliases; their evaluation uses the factored Fourier formula. See `validation.json` and `full_kernel_alias.csv`.

Maximum |A₀| over all tested phases and λ, p=7 (all maxima occur at λ=0, θ=0):
'''.replace('\\','\\'))
    text.append(table(['κ','max |A₀|'],[[r['kappa'],fmt(r['maximum'])] for r in aa]))
    text.append('''
![Smooth-kernel alias diagnostic](discrete_moments/full_kernel_alias.png)

For the full single-layer local model, the relevant leading term is δ A₀ ρ/(4π), rather than δ D₀ alone. For κ=1 at N=16384, this estimates 6.34×10⁻⁷ using the maximum circle density, matching the measured error. For κ=1.5 it gives 1.83×10⁻¹¹, also matching. At κ=2 the same local model is below 2.4×10⁻¹⁸ at N=16384. This is an interpretation of the local model, not a theorem for corrected double layers or curved complete representations.

## B. Circle benchmark

Errors below maximize over all three phases and offsets. Rates are with respect to hΓ. The last rate and trailing three-level fit are reported even at saturation, where they have no asymptotic meaning.
''')
    rr=[]
    for field in ['single','double']:
        for k in [1,1.5,2,3,4,6,8]:
            rs=[r for r in ss['circle_errors'] if r['field']==field and r['kappa']==k];r=rs[-1]
            rr.append([field,k,fmt(rs[0]['error_inf']),fmt(r['error_inf']),fmt(min(v['error_inf'] for v in rs)),f"{r['successive_rate']:.2f}",f"{r['window_rate']:.2f}"])
    text.append(table(['layer','κ','E at N=128','E at N=16384','minimum E','last rate','last 3-level fit'],rr))
    text.append('''
κ=1 and 1.5 visibly enter an approximately first-order regime above floating-point accuracy. κ≥2 reaches floating-point accuracy first. The complete per-level rates and moving-window fits are in `circle/errors_summary.csv`; minimum-error levels and crossover indicators are in `circle/diagnostics.json`. The existing q=5/7 reference remains the scale-separated theoretical verification.

Phase is decisive: at N=16384, κ=1 the single-layer errors for phases (0,1/4,1/2) are approximately (6.34×10⁻⁷,9.90×10⁻¹⁰,6.34×10⁻⁷). The double-layer errors are (1.13×10⁻⁹,5.81×10⁻⁷,1.13×10⁻⁹). At κ=1.5, quarter-step double-layer error is 2.39×10⁻¹¹ versus 1.19×10⁻¹³ for aligned and half-step targets. **Aligned and half-step targets alone would substantially understate the underresolved double-layer error.**

![Circle error curves](circle/errors.png)

## C. Complete Laplace on the current scaled kite
''')
    def finest(label,field):
        rows=[]
        for k in [2,3,4,6]:
            r=[r for r in ss[label+'_errors'] if r['field']==field and r['kappa']==k][-1]
            rows.append([k,r['n_gamma'],fmt(r['error_inf']),fmt(r['quadrature_difference_inf']),f"{r['successive_rate']:.2f}",f"{r['window_rate']:.2f}",f"{r['phase_ratio']:.3f}"])
        return table(['κ','N','E','N vs 2N','last rate','last 3-level fit','max/min phase error'],rows)
    text.append(finest('kite_laplace','laplace'))
    text.append('''
The finest kite errors remain smoothing-dominated for κ=2–6. Successive fine-grid rates approach seven, but this finite-range observation does not prove fixed-κ seventh-order asymptotics. Increasing κ increases the smoothing error substantially. The phase sensitivity is small in this resolved range.

## D. Complete Stokes on the current scaled starfish

Velocity:
''')
    text.append(finest('starfish_stokes','velocity'));text.append('\nPressure:\n');text.append(finest('starfish_stokes','pressure'))
    text.append('''
Velocity remains well resolved. Pressure reaches an implementation-dependent numerical floor around 10⁻¹³ on the finest grids: its boundary densities involve spectral derivatives, and N-versus-2N differences also rise to this level. In particular, the κ=2 finest-grid rate deteriorates; this is not evidence of the fixed-width first-order aliasing regime. The κ=3 pressure total error is already comparable to that floor, so its near-seven last rate must not be overinterpreted. Pressure jump conventions and correction formulas were retained; the repository regression suite passed (see `verification.txt`).

![Complete-problem comparisons](complete_comparison.png)

## E. Fixed-N sweep and practical interval

N=4096; each value maximizes over phases 0,1/2 and offsets. Each paired entry gives total error / N-versus-2N difference at fixed δ.
''')
    rr=[]
    for k in [.75,1,1.5,2,2.5,3,4,5,6,8]:
        vals=[]
        for label,field in [('kite_laplace','laplace'),('starfish_stokes','velocity'),('starfish_stokes','pressure')]:
            r=next(r for r in ss[label+'_sweep'] if r['field']==field and r['kappa']==k)
            vals.append(fmt(r['error_inf'])+' / '+fmt(r['quadrature_difference_inf']))
        rr.append([k]+vals)
    text.append(table(['κ','Laplace E / Q','velocity E / Q','pressure E / Q'],rr))
    text.append('''
The sampled curves decrease steeply through κ=1.5 and then rise as the smoothing width grows. Their smallest total errors occur near κ=1.5, **but that is not a robust quadrature-resolved choice**: quadrature differences are a substantial fraction of the total error (and essentially all of the velocity error). κ=0.75 and 1 are clearly underresolved. The interval 2≤κ≤4 provides good practical accuracy with quadrature differences well below total errors at N=4096. κ=2–3 is the more economical part of this interval; κ=4 gives extra resolution margin at a higher smoothing cost. This interval is evidence for these tests, not a geometry-independent guarantee, and the error-versus-quadrature separation naturally fails once a floating-point floor is reached.

### Supplementary quarter-step targets

Because the circle revealed a double-layer phase blind spot, a separate N=4096 check adds phase 1/4 for κ=1,1.5,2,3,4. These rows are kept separately in `phase_quarter.csv`; they do not silently replace the requested two-phase datasets.
''')
    rr=[]
    for label in ['kite_laplace','starfish_stokes']:
        path=ROOT/label/'phase_quarter.csv'
        if not path.exists():continue
        rs=read(path)
        for field in sorted(set(r['field'] for r in rs)):
            for k in [1,1.5,2,3,4]:
                vals=[r for r in rs if r['field']==field and float(r['kappa'])==k]
                if vals:rr.append([field,k,fmt(max(float(r['error_inf']) for r in vals)),fmt(max(float(r['quadrature_difference_inf']) for r in vals))])
    text.append(table(['field','κ','quarter-step E','quarter-step Q'],rr))
    text.append('''
The additional phases preserve the practical κ≥2 recommendation. The underresolved κ=1 and 1.5 results must be judged using the full phase checks, not a favorable aligned sample.

![Kite sweep](kite_laplace/sweep.png)
![Stokes sweep](starfish_stokes/sweep.png)

## F. Optional all-grid comparison

Two widths only: δ=2h_c and δ=3h_c, with the existing physical boundary matching unchanged. The table reports global maximum and discrete-L2 errors and their ratio to the corresponding saved production errors. Ratios below one improve upon production. Direct-halo norms and representative fixed-δ doubling differences are saved in each JSON; the check uses 97 distributed halo points plus each field's actual direct-halo error maximizer. This is a sampled quadrature check, not a full-halo bound. Exact-direct controls were not rerun.
''')
    grid=[]
    for label,prod in [('kite_laplace','laplace'),('starfish_stokes','stokes')]:
        for k in [2,3]:
            for m in [128,256,512,1024]:
                path=ROOT/'allgrid'/f'{label}_k{k}_M{m}.json'
                if not path.exists():continue
                r=json.loads(path.read_text());base=json.loads(Path(f'results/swapped_low_order/allgrid/{prod}/M{m}.json').read_text())
                for field in (['laplace'] if prod=='laplace' else ['velocity','pressure']):
                    grid.append(dict(field=field,kappa=k,m_grid=m,h_c=r['h_c'],n_gamma=r['n_gamma'],
                        error_inf=r[field]['inf'],error_l2=r[field]['l2'],
                        direct_inf=r['direct_'+field]['inf'],direct_l2=r['direct_'+field]['l2'],
                        quadrature_difference_inf=r['quadrature_difference_inf'][field],
                        ratio_inf=r[field]['inf']/base[field]['inf'],ratio_l2=r[field]['l2']/base[field]['l2']))
    if grid:
        for r in grid:
            prev=[v for v in grid if v['field']==r['field'] and v['kappa']==r['kappa'] and v['m_grid']<r['m_grid']]
            r['successive_rate_inf']='';r['successive_rate_l2']=''
            if prev:
                s=prev[-1]
                for norm in ['inf','l2']:r['successive_rate_'+norm]=np.log(s['error_'+norm]/r['error_'+norm])/np.log(s['h_c']/r['h_c'])
        save(ROOT/'allgrid/summary.csv',grid)
        text.append(table(['field','κ','M','max error','L2 error','max / production','L2 / production','halo max','sample Q'],[[r['field'],r['kappa'],r['m_grid'],fmt(r['error_inf']),fmt(r['error_l2']),f"{r['ratio_inf']:.3f}",f"{r['ratio_l2']:.3f}",fmt(r['direct_inf']),fmt(r['quadrature_difference_inf'])] for r in grid]))
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for field,ax in zip(['laplace','velocity','pressure'],axes):
            for k in [2,3]:
                rr=[r for r in grid if r['field']==field and r['kappa']==k]
                ax.loglog([r['h_c'] for r in rr],[r['error_inf'] for r in rr],'o-',label=f'κ={k}, max')
                ax.loglog([r['h_c'] for r in rr],[r['error_l2'] for r in rr],'s--',label=f'κ={k}, L2')
            prod='laplace' if field=='laplace' else 'stokes'
            base=[json.loads(Path(f'results/swapped_low_order/allgrid/{prod}/M{m}.json').read_text()) for m in [128,256,512,1024]]
            ax.loglog([r['h_c'] for r in base],[r[field]['inf'] for r in base],'k:',label='production max')
            ax.set(title=field,xlabel='bulk spacing hc',ylabel='error');ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(ROOT/'allgrid/comparison.png',dpi=180);fig.savefig(ROOT/'allgrid/comparison.pdf');plt.close(fig)
    text.append('''
![All-grid comparison](allgrid/comparison.png)

At M=1024, κ=2 gives maximum errors 1.35×10⁻⁹ (Laplace), 1.02×10⁻⁸ (velocity), and 2.28×10⁻⁹ (pressure): approximately 142, 149, and 168 times smaller than the saved production maxima. The corresponding L2 improvements are approximately 136, 50, and 161 times. The maxima occur in the direct halo. κ=3 also improves the fine-grid maxima by roughly 9–12 times. Representative doubling differences are far below the total errors. This makes the optional all-grid comparison a useful practical result.

The comparison is practical: the fixed-width rule uses a smaller width than production on sufficiently fine grids, reducing smoothing error. On coarse grids κ=3 can be wider than production and less accurate. All-grid behavior includes Cartesian reconstruction and is not an isolated boundary quadrature convergence theorem.

## G. Scientific conclusions and recommendation

1. **How large must δ/h be?** For these smooth manufactured tests, about two *maximum physical* boundary spacings already resolves p=7 very well. A practical interval is 2–4, with 2–3 attractive when smoothing error matters. Parameter spacing alone is not a transferable physical criterion on a nonuniform curve.
2. **Is κ=1 underresolved?** Yes. Local doubling differences are comparable to total errors; circle refinement exposes an approximately first-order regime. Favorable phases can hide it.
3. **Is there a broad robust interval?** Yes, 2–4 over the tested practical resolutions, subject to the pressure numerical floor. κ=1.5 can yield a smaller error at one N but fails the quadrature-separation test and is not the robust recommendation.
4. **Does fixed κ eventually lose nominal high-order behavior?** It demonstrably does for κ=1 and 1.5. Nonzero full-kernel alias coefficients support eventual lower-order behavior in a local single-layer model for larger finite κ. We do not claim a universal asymptotic theorem for all corrected curved operators.
5. **Is crossover inside or below practical double precision?** It is visible above roundoff for κ=1 and 1.5. For κ≥2, the tested circle reaches roundoff first; its leading local alias prediction is below the observed floating-point floor at the fine resolutions. The smallest computed coefficients strongly explain the delayed crossover for larger κ. Stokes pressure reaches a spectral-differentiation/implementation floor earlier, which must not be mistaken for this asymptotic crossover.
''')
    # Markdown math uses single backslashes.
    (ROOT/'REPORT.md').write_text('\n'.join(text).replace('\\\\','\\'))
    diag=[]
    for field in ['single','double']:
        for k in [1,1.5,2,3,4,6,8]:
            rr=[r for r in ss['circle_errors'] if r['field']==field and r['kappa']==k]
            best=min(rr,key=lambda r:r['error_inf'])
            first=[r['n_gamma'] for r in rr if r['successive_rate']!='' and .8<r['successive_rate']<1.2 and r['error_inf']>1e-14]
            floor=[r['n_gamma'] for r in rr if r['error_inf']<2e-15]
            diag.append(dict(field=field,kappa=k,minimum_error=best['error_inf'],minimum_at_n=best['n_gamma'],first_near_first_order_above_floor=min(first) if first else None,first_below_2e_15=min(floor) if floor else None,criterion='Rate in (0.8,1.2) is an indicator; inspect adjacent rates, not a certified crossover.'))
    (ROOT/'circle/diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')

if __name__=='__main__':main()
