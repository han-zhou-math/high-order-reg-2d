"""Checks of the regularized double-layer kernel (not a general DLP solver)."""
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1
from regbie.regularized_slp import coefficients
from regbie.regularized_dlp import smoothing_over_r2


def qpoly(u, p, lam=0., surface=False):
    if surface:
        if p == 3: return np.ones_like(u)
        if p == 5: return 1-2*u/3
        if p == 7: return 1-4*u/3+4*u*u/15
        raise ValueError(p)
    result = np.ones_like(u)
    for k, c in enumerate(coefficients(p, lam), 1):
        result += (-1)**k*c*u**k
    return result


def check_moments():
    residual = 0.
    identity_error = 0.
    for p in [3, 5, 7]:
        for lam in [0., .5, 1., 2.]:
            cs = coefficients(p, lam)
            for j in range((p-1)//2+1):
                def integrand(t):
                    u = t*t+lam*lam
                    return t**(2*j+2)/u*np.exp(-u)*qpoly(np.array(u), p, lam)
                moment = 2*quad(integrand, 0, np.inf, epsabs=1e-12)[0]
                def slp_integrand(t):
                    u = t*t+lam*lam
                    poly = -cs[0]
                    if p >= 5: poly += cs[1]*(1+u)
                    if p >= 7: poly -= cs[2]*(2+2*u+u*u)
                    return t**(2*j)*(exp1(u)+np.exp(-u)*poly)
                related = (2*j+1)*quad(slp_integrand, 0, np.inf, epsabs=1e-12)[0]
                identity_error = max(identity_error, abs(moment-related))
                if j < (p-1)//2:
                    residual = max(residual, abs(moment))
        for j in range(1, (p-1)//2):
            moment = 2*quad(lambda t: t**(2*j)*np.exp(-t*t)*
                            qpoly(np.array(t*t), p, surface=True),
                            0, np.inf, epsabs=1e-12)[0]
            residual = max(residual, abs(moment))
    assert residual < 2e-11, residual
    assert identity_error < 2e-10, identity_error
    return residual, identity_error


def circle_dlp(theta, delta, n, p, offset, surface=False):
    t = 2*np.pi*np.arange(n)/n
    rho = 1+offset*delta
    d = t[None, :]-theta[:, None]
    # Stable distance and normal numerator, including coincident nodes.
    r2 = (rho-1)**2+4*rho*np.sin(d/2)**2
    numerator = 1-rho+2*rho*np.sin(d/2)**2
    u = r2/delta**2
    ratio = smoothing_over_r2(r2, delta, p, offset*delta, surface)
    chi = 1. if offset < 0 else (0. if offset > 0 else .5)
    g0 = 1+.7*np.cos(3*theta)
    density_difference = -1.4*np.sin(1.5*(t[None, :]+theta[:, None]))*np.sin(1.5*d)
    value = np.sum(numerator*ratio*density_difference, axis=1)/n+chi*g0
    if offset < 0: exact = 1+.35*rho**3*np.cos(3*theta)
    elif offset > 0: exact = -.35*rho**(-3)*np.cos(3*theta)
    else: exact = np.full_like(theta, .5)
    return value, exact


def main():
    residual, identity_error = check_moments()
    theta = 2*np.pi*np.arange(97)/97
    rows = []
    summary = []
    for label, offset, surface in [('on', 0., False), ('inside', -.5, False),
                                    ('outside', .5, False), ('on_special', 0., True)]:
        for p,q in [(3, 2/3), (5, 4/5), (7, 5/7)]:
            group = []
            for n in [64, 128, 256, 512, 1024, 2048]:
                h = 2*np.pi/n
                delta = h**q
                v, exact = circle_dlp(theta, delta, n, p, offset, surface)
                fine, _ = circle_dlp(theta, delta, 4*n, p, offset, surface)
                err = np.max(np.abs(v-exact))
                qerr = np.max(np.abs(v-fine))
                row = dict(case=label, p=p, q=q, n=n, h=h, delta=delta,
                           error=err, quadrature_estimate=qerr)
                rows.append(row)
                if err > 5e-13 and qerr < .05*err:
                    group.append(row)
            fit = group[-3:]
            assert len(fit) == 3, (label,p)
            rate = np.polyfit(np.log([r['h'] for r in fit]),
                              np.log([r['error'] for r in fit]), 1)[0]
            assert abs(rate-p*q) < .35, (label,p,rate)
            summary.append(f'{label:12s} p={p} q={q:.6f} rate_h={rate:.4f} expected={p*q:.1f}')
    out = Path('results/double_layer')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'convergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    text = (f'Maximum cancelled moment residual: {residual:.3e}\n'
            f'Maximum moment identity discrepancy: {identity_error:.3e}\n'+
            '\n'.join(summary)+'\n')
    (out/'verification.txt').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
