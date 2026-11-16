"""Saved-data panels and explicitly selected resolved ranges."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/fixed_width')
STYLE = {2: ('#332288', 'o', '-'), 3: ('#CC6677', 's', '-.'),
         4: ('#AA8800', '^', ':')}


def series(geometry, field, kappa):
    with (ROOT / geometry / 'errors.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        if row['field'] == field and float(row['kappa']) == kappa:
            groups.setdefault(int(row['n_gamma']), []).append(row)
    result = []
    for n, rows in sorted(groups.items()):
        result.append(dict(n=n, h_gamma=float(rows[0]['h_gamma']),
            h_phys=float(rows[0]['h_phys_max']),
            error=max(float(r['error_inf']) for r in rows),
            doubling=max(float(r['quadrature_difference_inf']) for r in rows)))
    return result


def resolved(row):
    # Empirical selection for these tests, not a universal quadrature criterion.
    return row['doubling'] < .1 * row['error'] and row['error'] > 2e-15


def panel(ax, geometry, field):
    key = 'h_gamma' if geometry == 'circle' else 'h_phys'
    for k, (color, marker, style) in STYLE.items():
        rows = series(geometry, field, k)
        ax.loglog([r[key] for r in rows], [r['error'] for r in rows],
                  color=color, linestyle=style, linewidth=1.2,
                  label=fr'$\kappa={k}$')
        for ok in [True, False]:
            rr = [r for r in rows if resolved(r) == ok]
            ax.loglog([r[key] for r in rr], [r['error'] for r in rr],
                      linestyle='none', marker=marker, markersize=3.7,
                      color=color, markerfacecolor=color if ok else 'white')
    # A guide anchored to two contiguous resolved points, never the floor.
    rr = [r for r in series(geometry, field, 3) if resolved(r)][-2:]
    x = np.array([r[key] for r in rr])
    y = rr[-1]['error'] * 3 * (x / x[-1])**7
    ax.loglog(x, y, '--', color='.45', lw=.8, label='slope $7$')
    ax.set(xlabel=r'$h$' if key == 'h_gamma' else r'$h_{\rm phys}$',
           ylabel='maximum error')
    ax.grid(which='both', alpha=.25)
    ax.legend(loc='lower right', fontsize=7)


def metadata():
    records = []
    for geometry, fields in [('circle', ['single', 'double']),
                             ('kite_laplace', ['laplace']),
                             ('starfish_stokes', ['velocity', 'pressure'])]:
        for field in fields:
            for k in STYLE:
                rows = series(geometry, field, k)
                # Kite and velocity: 2^12-2^14. Pressure: 2^11-2^13, since its
                # 2^14 level is at roundoff and fails the resolved() test.
                fit_n = ([4096, 8192, 16384] if field in ['laplace', 'velocity']
                         else [2048, 4096, 8192] if field == 'pressure' else [])
                fit = [r for r in rows if r['n'] in fit_n]
                assert all(resolved(r) for r in fit)
                rate = float(np.polyfit(np.log([r['h_phys'] for r in fit]),
                                       np.log([r['error'] for r in fit]), 1)[0]) if fit else None
                records.append(dict(geometry=geometry, field=field, kappa=k,
                    fit_n=fit_n, fit_rate=rate,
                    fit_spacing='h_Gamma_phys' if fit else None,
                    resolved_n=[r['n'] for r in rows if resolved(r)],
                    excluded_n=[r['n'] for r in rows if not resolved(r)],
                    policy='No circle fits; pressure fitted over 2^11-2^13. Hollow markers are excluded from order assessment.',
                    levels=rows))
    (ROOT / 'paper_fit_metadata.json').write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__':
    metadata()
