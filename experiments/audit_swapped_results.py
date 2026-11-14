"""Recompute numerical-section fits and checks from the accepted saved data."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path('results/swapped_low_order')


def fit(levels, spacings, errors):
    h, e = np.asarray(spacings), np.asarray(errors)
    return dict(levels=list(levels), errors=e.tolist(),
                slope=float(np.polyfit(np.log(h), np.log(e), 1)[0]),
                successive=(np.log(e[:-1]/e[1:])/np.log(h[:-1]/h[1:])).tolist())


def main():
    report = dict(allgrid={}, local={}, controls={}, circle={})
    for problem, fields in [('laplace', ['laplace']),
                            ('stokes', ['pressure', 'velocity'])]:
        rows = [json.loads((ROOT/'allgrid'/problem/f'M{m}.json').read_text())
                for m in (64,128,256,512,768,1024)]
        for field in fields:
            report['allgrid'][field] = {}
            for norm in ('inf', 'l2'):
                report['allgrid'][field][norm] = fit(
                    [r['m_grid'] for r in rows], [r['h_c'] for r in rows],
                    [r[field][norm] for r in rows])
                report['allgrid'][field][norm]['late'] = fit(
                    [r['m_grid'] for r in rows[-3:]], [r['h_c'] for r in rows[-3:]],
                    [r[field][norm] for r in rows[-3:]])
        data = np.load(ROOT/'allgrid'/problem/'fields_M1024.npz')
        for field in fields:
            key = 'error' if field == 'laplace' else field+'_error'
            e = data[key]
            if field == 'velocity': e = np.linalg.norm(e, axis=-1)
            assert np.isfinite(e).all()
            assert np.isclose(rows[-1][field]['l2'], rows[-1]['h_c']*np.linalg.norm(e))
            assert data['halo'][np.unravel_index(np.abs(e).argmax(), e.shape)]
            report['allgrid'][field]['zero_rows'] = int(np.all(e == 0, axis=1).sum())
    for path in (ROOT/'allgrid'/'controls').glob('*.json'):
        raw = json.loads(path.read_text())
        report['controls'][path.stem] = {k:raw[k]['inf'] for k in
            ('laplace','pressure','velocity') if k in raw}
    with (ROOT/'local'/'errors.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    ext = ROOT/'local'/'p7_extension.csv'
    if ext.exists():
        with ext.open() as stream: rows += list(csv.DictReader(stream))
    for field in ('laplace','velocity','pressure'):
        for order in (3,5,7):
            selected = [r for r in rows if r['field']==field and int(r['order'])==order
                        and abs(float(r['b_over_delta']))<=.5]
            levels = sorted({int(r['n_gamma']) for r in selected})
            errors = [max(float(r['error_inf']) for r in selected if int(r['n_gamma'])==n)
                      for n in levels]
            item = fit(levels, [2*np.pi/n for n in levels], errors)
            item['late'] = fit(levels[-3:], [2*np.pi/n for n in levels[-3:]], errors[-3:])
            item['finest_doubling'] = max(float(r['quadrature_difference_inf'])
                for r in selected if int(r['n_gamma'])==levels[-1])
            report['local'][f'{field}_p{order}'] = item
    for lam in (0,.5,1,2):
        selected = [r for r in rows if r['field']=='laplace' and int(r['order'])==7
                    and abs(float(r['b_over_delta']))==lam]
        levels = [4096,8192,16384]
        report['local'][f'distance_{lam}'] = fit(levels, [2*np.pi/n for n in levels],
            [max(float(r['error_inf']) for r in selected if int(r['n_gamma'])==n) for n in levels])
    with Path('results/redesigned/circle_layers/errors.csv').open() as stream:
        circle = list(csv.DictReader(stream))
    for field in ('circle_single','circle_double'):
        for order in (3,5,7):
            levels = [512,1024,2048,4096] if order==7 else [2048,4096,8192,16384]
            report['circle'][f'{field}_p{order}'] = fit(levels,[2*np.pi/n for n in levels],
                [max(float(r['error_inf']) for r in circle if r['solution']==field
                     and int(r['order'])==order and int(r['n_gamma'])==n) for n in levels])
    (ROOT/'numerical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
