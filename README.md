# High-order regularized layer potentials in 2D

Python code for evaluating the Laplace single- and double-layer potentials
and the Stokeslet and stresslet integrals on a smooth closed curve in the
plane, for targets on, near, or far from the curve. The singular kernels are
replaced by E1-regularized kernels of order p = 3, 5, 7, with smoothing width
δ; the regularized integrals are then summed with the trapezoidal rule on the
same nodes that are used away from the curve. For targets near the curve,
corrections depending on λ = b/δ (b the signed distance to the curve) cancel
the leading moments of the regularization error, so the error is O(δ^p) plus
a quadrature error that is negligible when δ/h is large enough.

## Layout

| Folder | Contents |
|---|---|
| `regbie/` | Core kernels. `regularized_slp.py`: coefficients a_k(λ), regularized G_δ and the single layer. `regularized_dlp.py`: the double layer. `surface_operators.py`: radial factors for gradients and the Stokes kernels. `verify_stokes_derivation.py`: Stokes moment N₀ and checks. `all_grid_reconstruction.py`: Cartesian-grid stencils and solvers. |
| `experiments/` | Drivers, saved-data analysis and plotting for the numerical tests (circle, kite, starfish, corrections, near-curve limits, choice of δ, timing, Cartesian grid). |
| `examples/` | Two short runs with their saved output in `examples/output/`. |
| `verification/` | Earlier convergence and derivation checks (circle, ellipse, starfish; double layer; Stokes). |
| `tests/` | Unit and regression tests (`unittest`). |
| `results/` | Saved CSV/JSON results of the numerical tests. Grid-field arrays (`.npz`) are not included. |
| `cpp/` | C++17 version of `regbie/` and of the drivers that produce `results/` (FFTW, Eigen, OpenMP); see `cpp/README.md`. |

## Requirements

Python 3.10 or later with NumPy, SciPy and Matplotlib; `mpmath` for the
high-precision tests.

```sh
python3 -m pip install -r requirements.txt
```

Run every command below from the repository root.

## Tests

```sh
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -t .
```

The 42 tests take about a minute, most of it in `tests/test_api.py`.

## Examples

```sh
python3 -m examples.circle_single_layer   # about 1 s
python3 -m examples.kite_and_starfish     # about 15 s
```

`circle_single_layer` uses only `regbie`. It evaluates the single layer for
the unit circle with δ = h^q at 97 targets with λ = −0.5, 0, 0.5, and shows
errors decreasing as h², h⁴, h⁵ for p = 3, 5, 7.

`kite_and_starfish` is a shorter version of the main tests in `experiments/`: S f + D g
for the nonconvex kite, and the Stokes velocity and pressure for the starfish,
each compared with an exact solution. Its errors agree with the saved
values in `results/swapped_low_order/local/errors.csv`.

A minimal call:

```python
import numpy as np
from regbie.regularized_slp import circle, single_layer

theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
targets = circle(theta)[0]                      # targets on the curve, b = 0
values = single_layer(curve=circle, density=lambda t: 1 + 0.7 * np.cos(3 * t),
                      targets=targets, delta=0.05, n=1024, order=7, b=0.0)
```

`curve(t)` returns points and derivatives on [0, 2π); the density is per unit
arclength; `b` is the signed distance from each target to the curve
(negative inside, positive outside). Conventions: G(r) = log(r)/(2π), outward
normals, and for targets on the curve the double layer gives the average of
the two one-sided limits.

### Layer potentials at arbitrary targets

`regbie.api` finds closest points and signed distances itself, uses the
regularized kernels only near the curve, and chooses δ (3 node spacings) by
default:

```python
from regbie.api import Curve, LayerPotentials

curve = Curve.from_points(points)          # (N, 2), counterclockwise; or Curve.from_derivatives(...)
layers = LayerPotentials(curve, n=len(points), order=7)
u = layers.single_layer(f, targets) + layers.double_layer(g, targets)    # Laplace
v = layers.stokeslet(f2, targets) + layers.stresslet(q, targets)         # Stokes velocity
p = layers.stokes_pressure(f2, q, targets)                               # Stokes pressure
```

Densities are given at the nodes `layers.parameters` (t_k = 2πk/n); see the
docstring of `regbie/api.py` for the conventions and `examples/layer_potentials.py`
for a complete run. The C++ library in `cpp/` has the same interface.

## Reproducing the figures

Plotting reads only the saved data and writes PDFs to `figures/`:

```sh
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
python3 -m experiments.plot_revised_convergence    # circle, kite, starfish
python3 -m experiments.plot_revision_figures       # corrections, near-curve limits, choice of δ, grid widths
python3 -m experiments.plot_revised_grid_figures   # Cartesian grid convergence, exact-data control, stencils
```

To recompute the data (these runs take from minutes to hours):

| Data | Command |
|---|---|
| Circle (`results/redesigned/circle_layers`) | `python3 -m experiments.run_circle_layers --run` |
| Kite and starfish, near-curve targets (`results/swapped_low_order/local`) | `python3 -m experiments.run_swapped_production --local` |
| Cartesian grid, δ = h^{5/7} (`results/swapped_low_order/allgrid`) | `python3 -m experiments.run_swapped_production --allgrid` |
| Fixed δ/h curves (`results/fixed_width`) | `python3 -m experiments.fixed_width --stage circle` and `--stage complete`, then `python3 -m experiments.fixed_width_allgrid` |
| Corrections, near-curve limits, choice of δ, timing (`results/paper_revision`) | `python3 -m experiments.paper_revision --stage all` |
| Pressure densities | `python3 -m experiments.pressure_density_diagnostic` |
| Exact-data grid control | `python3 -m experiments.exact_data_reconstruction` |

The Cartesian-grid runs use δ = (2π/N_Γ)^{5/7}, passed explicitly by
`run_swapped_production`; `experiments.matched_allgrid.run()` on its own
defaults to a different δ.
