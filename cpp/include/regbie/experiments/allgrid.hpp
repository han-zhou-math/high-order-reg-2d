// Cartesian-grid reconstruction of the Laplace and Stokes interface problems:
// direct (regularized) values on the stencil halo of the interface, a nine-point
// sine-transform solve elsewhere. C++ counterparts of experiments/allgrid.py
// (solve_laplace, solve_stokes) and experiments/matched_allgrid.py (spacing,
// norms, run).
#pragma once
#include <array>
#include <optional>
#include <string>
#include <vector>

#include "regbie/grid.hpp"
#include "regbie/io.hpp"
#include "regbie/experiments/problems.hpp"

namespace regbie {

// Sum of a contiguous array as np.sum (pairwise summation).
double pairwise_sum(const double* a, long n);

// Outer ring of the full grid and its points in row-major order (allgrid._boundary_points).
struct BoundaryPoints {
  Mask mask;
  std::vector<Vec2> points;
};
BoundaryPoints boundary_points(const GridCoords& grid);

// allgrid.solve_laplace: direct values on the halo of the nine-point crossing set,
// checked against 2x oversampled quadrature.
struct AllgridLaplaceResult {
  Grid values, error, xx, yy;
  Mask c_delta, halo;
  double h = 0;
  double error_inf = 0, error_l2 = 0, oversampled_error_inf = 0;
  int direct_targets = 0;
  double bie_seconds = 0, dst_seconds = 0, total_seconds = 0;
};
AllgridLaplaceResult solve_laplace(const LaplaceProblem& problem, int m, int n_gamma, double delta,
                                   int order = 7, int oversample_factor = 2);

// allgrid.solve_stokes: pressure first, then the velocity with -Lap u = -grad p / mu
// away from the interface.
struct AllgridStokesResult {
  Grid pressure, pressure_error, xx, yy;
  std::array<Grid, 2> velocity, velocity_error;
  double h = 0;
  Mask c_delta, c_gradient, halo;
  double pressure_error_inf = 0, pressure_error_l2 = 0, velocity_error_inf = 0, velocity_error_l2 = 0;
  double gradient_error_inf = 0, gradient_error_l2 = 0;
  double oversampled_error_inf = 0, pressure_oversampled_error_inf = 0, velocity_oversampled_error_inf = 0;
  int direct_targets = 0;
  double bie_seconds = 0, dst_seconds = 0, total_seconds = 0;
};
AllgridStokesResult solve_stokes(const StokesProblem& problem, int m, int n_gamma, double delta,
                                 int order = 7, int oversample_factor = 2);

// matched_allgrid.spacing: the even 11-smooth node count N whose largest chord is
// within [0.8, 1.2] * ratio * h (or the given n, unchanged).
struct Spacing {
  int n_gamma = 0;
  double chord_max = 0, chord_mean = 0, spacing_ratio = 0;
};
Spacing spacing(const Geometry& geometry, double h, std::optional<int> n = std::nullopt, double ratio = 1.0);

// matched_allgrid.norms: {"inf": max |v|, "l2": h sqrt(sum |v|^2)}; vectors by Euclidean length.
Json norms(const std::vector<double>& v, double h);
Json norms(const std::vector<Vec2>& v, double h);

// The fields dict of matched_allgrid.run. The Stokes entries are empty for Laplace.
struct MatchedFields {
  Grid xx, yy, error;
  Mask c_delta, c_gradient, halo;
  Grid pressure_error;
  std::array<Grid, 2> velocity_error;
  std::vector<int> profile_rows;
  Grid pressure_profiles;
};

struct MatchedRun {
  Json report;
  MatchedFields fields;
};

// Arguments of matched_allgrid.run. kappa keeps its Python type (int or float) for the report.
struct MatchedOptions {
  std::string mode = "bie";     // bie, exact_direct, exact_gradient
  std::string rule = "matched";  // matched, twice, old8m
  Json kappa = 1.0;
  std::optional<double> delta_override;
  std::optional<int> n_override;
};

MatchedRun matched_run(const LaplaceProblem& problem, int m, const MatchedOptions& options = {});
MatchedRun matched_run(const StokesProblem& problem, int m, const MatchedOptions& options = {});

// Grid node of the largest |error| (run_swapped_production._maximum_node):
// {"index": [row, col], "point": [x, y], "error", "in_direct_halo"}.
Json maximum_node(const MatchedFields& fields, const Grid& error);
Json maximum_node(const MatchedFields& fields, const std::array<Grid, 2>& error);

// experiments/problems.py: LAPLACE_STARFISH and STOKES_KITE (defaults of matched_allgrid.main).
const LaplaceProblem& laplace_starfish_problem();
const StokesProblem& stokes_kite_problem();

}  // namespace regbie
