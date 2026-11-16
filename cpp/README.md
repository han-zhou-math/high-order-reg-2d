# C++ version

A C++17 implementation of the computations in this repository: the regularized
single- and double-layer potentials, the Stokeslet and stresslet integrals, the
Cartesian-grid reconstruction, and the drivers that produce the data in
`results/`. Plotting and the analysis of saved data stay in Python; the C++
drivers write the same CSV and JSON files, so the Python plotting scripts can
read either.

## Layout

| Folder | Contents |
|---|---|
| `include/regbie/`, `src/` | Library `regbie`. `kernels`: coefficients a_k(λ), G_δ, T/r², radial factors, λN₀. `layers`: single and double layer, Laplace, pressure and Stokes evaluators. `geometry`: curves, point-in-curve test, closest points. `periodic`: trigonometric interpolation, spectral derivatives, pressure densities. `grid`: stencils, nine-point Laplacian, fourth-order gradient, fast sine-transform solver. `problems`: the kite (Laplace) and starfish (Stokes) tests. `io`: CSV and JSON in the formats the Python code writes. `numerics`: quadrature, bounded minimization, FFT lengths, fits. |
| `drivers/` | One program per Python driver (see below). |
| `tests/` | `core_values`: prints core-library values for `tools/check_core.py`. |
| `tools/` | `check_core.py`: compares `core_values` with the Python code. |

## Building

Requirements: a C++17 compiler with OpenMP, CMake 3.20 or later, FFTW 3, and
Eigen 3.4 (downloaded by CMake if not installed). On OSC Cardinal:

```sh
module load gcc/13.2.0 mvapich/4.1 fftw/3.3.10 cmake/3.25.2
cmake -S cpp -B cpp/build
cmake --build cpp/build -j 8
```

Run every program from the repository root. The number of threads is set by
`OMP_NUM_THREADS`.

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
