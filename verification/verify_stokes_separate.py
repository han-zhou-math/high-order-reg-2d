"""Independent off-curve checks of the two corrected Stokes operators.

Compare to oversampled ORIGINAL kernels, without regularization or subtraction.
Run: OPENBLAS_NUM_THREADS=1 python3 verify_stokes_separate.py
"""
import csv
from pathlib import Path
import numpy as np
from regbie.verify_stokes_derivation import geometry, field, traction, evaluate


def reference(n, y):
    x, speed, _, normal, _, _ = geometry(2*np.pi*(np.arange(n)+.173)/n)
    f, q = traction(x, normal), field(x)
    w = 2*np.pi*speed/n
    sl, dl = [], []
    for target in y:
        r = x-target
        r2 = np.sum(r*r, axis=1)
        sf = (-.5*np.log(r2)[:, None]*f
              +r*(np.sum(r*f, axis=1)/r2)[:, None])/(4*np.pi)
        dq = r*(np.sum(r*q, axis=1)*np.sum(r*normal, axis=1)/(np.pi*r2**2))[:, None]
        sl.append(np.einsum('ni,n->i', sf, w))
        dl.append(np.einsum('ni,n->i', dq, w))
    return np.array(sl), np.array(dl)


def main():
    theta = 2*np.pi*(np.arange(37)+.31)/37
    x0, _, _, n0, _, _ = geometry(theta)
    rows = []
    for lam in (-.5, .5):
        for delta in (.04, .02, .01):
            b = lam*delta
            refs = reference(32768, x0+b*n0)
            finer = reference(65536, x0+b*n0)
            ref_error = max(np.max(np.linalg.norm(a-bb, axis=1)) for a, bb in zip(refs, finer))
            assert ref_error < 1e-11, ref_error
            for order in (3, 5, 7):
                approx = evaluate(8192, delta, order, b, theta, return_parts=True)
                for operator, val, ref in zip(('SL', 'DL'), approx, finer):
                    error = np.max(np.linalg.norm(val-ref, axis=1))
                    rows.append(dict(operator=operator, order=order, lam=lam, delta=delta,
                                     error=error, reference_check=ref_error))
    dest = Path('results/stokes_derivation/separate_operators.csv')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    print('Maximum reference refinement discrepancy:', max(r['reference_check'] for r in rows))
    for lam in (-.5, .5):
        for order in (3, 5, 7):
            for op in ('SL', 'DL'):
                rs = [r for r in rows if r['lam'] == lam and r['order'] == order and r['operator'] == op]
                rate = np.log2(rs[-2]['error']/rs[-1]['error'])
                print(f'{op}, p={order}, b/delta={lam:+.1f}: error {rs[-1]["error"]:.3e}, delta rate {rate:.3f}')


if __name__ == '__main__':
    main()
