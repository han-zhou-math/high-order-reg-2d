# C++ version

A C++17 implementation of the computations in this repository: a library for
the regularized Laplace and Stokes layer potentials at targets on, near or far
from a smooth closed curve, the Cartesian-grid reconstruction, and the drivers
that produce the data in `results/`. Plotting and the analysis of saved data
stay in Python; the C++ drivers write the same CSV and JSON files, so the
Python plotting scripts can read either.

## Using the library

Two classes cover most uses; include `regbie/regbie.hpp`.

```cpp
#include "regbie/regbie.hpp"

// A counterclockwise curve from N points x(2 pi k/N) (or from functions, below).
regbie::Curve curve = regbie::Curve::from_points(points);   // std::vector<regbie::Vec2>
regbie::LayerPotentials layers(curve, n);                    // order 7, delta = 3 node spacings

// Densities per unit arclength at the nodes t_k = 2 pi k/n (layers.nodes()[k].x, .normal, ...).
std::vector<double> s = layers.single_layer(f, targets);      // S f, Laplace
std::vector<double> d = layers.double_layer(g, targets);      // D g, Laplace
std::vector<regbie::Vec2> u = layers.stokeslet(f2, targets);  // S f, Stokes
std::vector<regbie::Vec2> v = layers.stresslet(q, targets);   // D q, Stokes
std::vector<double> p = layers.stokes_pressure(f2, q, targets);
```

Targets may lie anywhere. Those within `near * delta` of the curve (default
`near = 6`) are evaluated with the regularized kernels and local corrections;
the routine finds their closest points and signed distances itself. The others
use the trapezoidal rule with the original kernels. The error near the curve is
O(δ^p) plus a negligible quadrature error; with the default δ = 3Δs (Δs the
largest node spacing) it is about 1e-8 at n = 4096 on the kite and starfish
tests (`tests/api_check.cpp`).

| Choice | How |
|---|---|
| Curve from functions | `Curve::from_derivatives(t -> {x, x', x'', x'''})`, or `Curve::from_function(t -> GeomPoint)` |
| Curve from points | `Curve::from_points(points)`: N ≥ 8 points at equally spaced parameters; derivatives by FFT |
| Order, δ, near-zone size, viscosity | `PotentialOptions{order, delta, near, mu}` as the third argument |
| Closest points of your own targets | `curve.project(points)` returns parameters t and signed distances b |

Conventions:

| Quantity | Convention |
|---|---|
| Orientation | counterclockwise; the normal (τ₂, −τ₁) points outward; clockwise input is rejected |
| Laplace kernel | G(r) = log(r)/(2π) (note the sign); S f = ∫ G f ds, D g = ∫ ∂G/∂n_x g ds |
| Densities | per unit arclength, given at the n nodes |
| Signed distance | negative inside, positive outside |
| Representation | u harmonic inside: S(−∂u/∂n) + D(u) = u inside, 0 outside. Stokes flow inside with viscosity μ, q = u and f = σn on the curve: S f + D q = u inside, 0 outside; `stokes_pressure(f, q)` = p inside, 0 outside |
| On the curve | the double layers return the average of the two one-sided limits |

The near-curve corrections need 1 + bκ > 0 at every near target (the target lies within a
curvature radius); otherwise the routines throw and ask for a larger n or a smaller δ.
`examples/layer_potentials.cpp` is a complete program. The Python package has the
same interface in `regbie/api.py` (`Curve`, `LayerPotentials`), with
`examples/layer_potentials.py`.

The lower-level routines remain available: `layers.hpp` evaluates the layer
potentials for given closest parameters and signed distances, and
`kernels.hpp` gives the regularized kernels and their coefficients.

## Layout

| Folder | Contents |
|---|---|
| `include/regbie/`, `src/` | Library `regbie`. `curve`, `potentials`: the interface above. `kernels`: coefficients a_k(λ), G_δ, T/r², radial factors, λN₀. `layers`: layer potentials for given closest points, Laplace, pressure and Stokes evaluators. `geometry`: frames, point-in-curve test, closest points. `periodic`: trigonometric interpolation, spectral derivatives, pressure densities. `grid`: stencils, nine-point Laplacian, fourth-order gradient, fast sine-transform solver. `io`: CSV and JSON in the formats the Python code writes. `numerics`: quadrature, bounded minimization, FFT lengths, fits. |
| `include/regbie/experiments/`, `src/experiments/` | Library `regbie_experiments`, used only by the drivers: the kite and starfish test problems, the Cartesian-grid reconstruction, fixed-width and local-test helpers. |
| `examples/` | `layer_potentials`: the interface on an ellipse given by points. |
| `drivers/` | One program per Python driver (see below). |
| `tests/` | `api_check`: accuracy of the interface against exact solutions. `core_values`: prints core-library values for `tools/check_core.py`. |
| `tools/` | `check_core.py`, `compare_results.py`: comparisons with the Python code. |

## Building and installing

Requirements: a C++17 compiler with OpenMP, CMake 3.20 or later, FFTW 3, and
Eigen 3.4 (downloaded by CMake if not installed). On OSC Cardinal:

```sh
module load gcc/13.2.0 mvapich/4.1 fftw/3.3.10 cmake/3.25.2
cmake -S cpp -B cpp/build
cmake --build cpp/build -j 8
cpp/build/api_check          # accuracy check of the interface
```

Run every program from the repository root. The number of threads is set by
`OMP_NUM_THREADS`.

To use the library from another CMake project, install it and link `regbie::regbie`:

```sh
cmake -S cpp -B cpp/build -DREGBIE_BUILD_EXPERIMENTS=OFF -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build cpp/build -j 8 && cmake --install cpp/build
```

```cmake
find_package(regbie REQUIRED)                 # with CMAKE_PREFIX_PATH=$HOME/.local
target_link_libraries(myprogram PRIVATE regbie::regbie)
```

The installed headers exclude `experiments/`; when Eigen was downloaded, its
headers are installed with the library.

## Output

Each driver writes the file that its Python counterpart writes below
`results/`, at the same relative path below `cpp/results/`; pass
`--output-root DIR` to choose another root. The saved Python results are
never overwritten unless `--output-root results` is given.

## Checking against the Python code

```sh
cpp/build/core_values > cpp/build/values.json
PYTHONPATH=. python3 cpp/tools/check_core.py cpp/build/values.json
```

prints the largest difference, relative to the size of the values, for each
part of the library. The C++ and Python values agree to roundoff (about
1e-15; about 1e-13 for the nine-point Laplacian and spectral derivatives,
whose factors 1/h² and n² amplify roundoff).

## Programs

Paths in the last column are below the output root (`cpp/results` by default).

| Program | Python counterpart | Output |
|---|---|---|
| `circle_single_layer` | `examples/circle_single_layer.py` | printed table |
| `run_circle_layers --run` | `experiments.run_circle_layers --run` | `redesigned/circle_layers/{errors.csv, metadata.json}` |
| `run_swapped_local` | `experiments.run_swapped_production --local` | `swapped_low_order/local/errors.csv` |
| `run_swapped_allgrid --allgrid` | `experiments.run_swapped_production --allgrid` | `swapped_low_order/allgrid/{laplace,stokes}/M*.json` |
| `fixed_width --stage circle\|complete\|sweep\|phase` | `experiments.fixed_width` | `fixed_width/{circle,kite_laplace,starfish_stokes}/errors.csv`, ... |
| `fixed_width_allgrid` | `experiments.fixed_width_allgrid` | `fixed_width/allgrid/*.json` |
| `paper_revision --stage all` | `experiments.paper_revision --stage all` | `paper_revision/*.csv`, `summary.json`, `reference_checks.json`, `metadata.json` |
| `pressure_density_diagnostic` | `experiments.pressure_density_diagnostic` | `paper_revision/pressure_density_diagnostic.csv` |
| `exact_data_reconstruction` | `experiments.exact_data_reconstruction` | `paper_revision/exact_data_reconstruction.json` |
| `run_missing_practical` | `experiments.run_missing_practical` | `paper_revision/allgrid_practical/{laplace,stokes}/M{64,768}.json` |
| `geometric_diagnostic` | `experiments.geometric_diagnostic` | `paper_revision/geometric_diagnostic.{csv,json}` |
| `matched_allgrid --problem laplace\|stokes` | `experiments.matched_allgrid` | `matched_allgrid/...` |

`geometric_diagnostic` reads the Stokes reports of `run_swapped_allgrid`, so
run it last. Plots, fits of saved data (`fixed_width_paper`, the analysis
and report scripts) and the `--stage moments` diagnostic of `fixed_width`,
which needs 90-digit arithmetic, remain in Python. The `.npz` field arrays
are not written.

To compare a set of C++ results with the saved Python results:

```sh
python3 cpp/tools/compare_results.py cpp/results results
```

It prints, for each file and column, the largest difference relative to the
largest Python value of that column. Timing fields are skipped.

## Differences from the Python code

- Sums over the quadrature nodes use compensated summation, so roundoff-level
  quantities (differences between N and 2N nodes, errors near 1e-14) agree
  with the Python values in size but not digit for digit.
- Where the largest error is attained at several points with equal values
  (for example by symmetry on the circle), the reported location
  (`maximum_target_parameter`) may be a different one of these points.
- Timings differ. In `paper_revision` the timed evaluations run on one thread,
  like the Python runs with `OPENBLAS_NUM_THREADS=1`; `metadata.json` records
  the compiler, Eigen and FFTW versions in place of the Python, NumPy and
  SciPy versions.
