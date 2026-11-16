// Print core-library values as JSON for comparison with the Python code
// (cpp/tools/check_core.py). Run from the repository root:
//   cpp/build/core_values > values.json
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/grid.hpp"
#include "regbie/io.hpp"
#include "regbie/kernels.hpp"
#include "regbie/layers.hpp"
#include "regbie/periodic.hpp"
#include "regbie/problems.hpp"
#include "regbie/special.hpp"

using namespace regbie;

int main() {
  Json out = Json::object();
  const std::vector<int> orders = {3, 5, 7};
  const std::vector<double> lambdas = {0.0, 0.3, 1.0, 1.5, 3.0, 10.0};

  Json coef = Json::array(), moment = Json::array();
  for (int p : orders)
    for (double lam : lambdas) {
      const Coefficients cs = coefficients(p, lam);
      coef.push_back(Json::Array{Json(cs.c[0]), Json(cs.c[1]), Json(cs.c[2])});
      moment.push_back(scaled_moment_n0(p, lam));
    }
  out["coefficients"] = coef;
  out["scaled_moment_n0"] = moment;

  const double delta = 0.1;
  Json kernel = Json::array(), smooth = Json::array(), radial = Json::array();
  for (int p : orders)
    for (double lam : {0.0, 0.5, 2.0}) {
      const Coefficients cs = coefficients(p, lam);
      for (double r : {0.0, 1e-3, 0.05, 0.07, 0.2, 1.0, 3.0}) {
        kernel.push_back(regularized_kernel(r, delta, cs));
        smooth.push_back(smoothing_over_r2(r * r, delta, cs));
      }
      for (double u : {0.0, 0.1, 0.49, 0.51, 2.0, 30.0}) {
        const RadialFactors rf = radial_factors(u, cs);
        radial.push_back(Json::Array{Json(rf.a), Json(rf.ap), Json(rf.tp)});
      }
    }
  out["regularized_kernel"] = kernel;
  out["smoothing_over_r2"] = smooth;
  out["radial_factors"] = radial;

  Json special = Json::object();
  Json e1 = Json::array(), ex = Json::array();
  for (double x : {0.01, 0.5, 0.99, 1.0, 2.5, 10.0, 50.0}) e1.push_back(exp1(x));
  for (double x : {0.0, 0.3, 1.0, 3.0, 6.0, 30.0}) ex.push_back(erfcx(x));
  special["exp1"] = e1;
  special["erfcx"] = ex;
  out["special"] = special;

  // Layer potentials on the regbie starfish, density 1 + 0.3 sin 2t, at targets x(s) + b n(s).
  const int n = 256;
  std::vector<double> density(n);
  const auto t = periodic_nodes(n);
  for (int k = 0; k < n; ++k) density[k] = 1 + 0.3 * std::sin(2 * t[k]);
  std::vector<Vec2> targets;
  std::vector<double> b, s, g0;
  for (int i = 0; i < 12; ++i) {
    const double si = 2 * PI * (i + 0.3) / 12, bi = 0.02 * (i % 5 - 2);
    const GeomPoint g = regbie_starfish(si);
    targets.push_back(g.x + bi * g.normal);
    b.push_back(bi);
    s.push_back(si);
    g0.push_back(1 + 0.3 * std::sin(2 * si));
  }
  Json layers = Json::object();
  for (int p : orders) {
    const double d = 0.05;
    layers["single_" + std::to_string(p)] = Json(single_layer(regbie_starfish, density, targets, b, d, p));
    layers["double_" + std::to_string(p)] = Json(double_layer(regbie_starfish, density, g0, targets, b, d, p));
  }
  out["layers"] = layers;

  // Stokes on the scaled starfish: closest points, pressure and velocity at grid-like targets.
  const StokesProblem& sp = stokes_starfish();
  std::vector<Vec2> st;
  for (int i = 0; i < 10; ++i) {
    const double si = 2 * PI * (i + 0.17) / 10;
    const GeomPoint g = sp.geometry(si);
    st.push_back(g.x + (0.015 * (i % 3) - 0.01) * g.normal);
  }
  const ClosestPoints cp = closest_points(st, sp.geometry, sp.inside);
  std::vector<Vec2> q, f;
  sp.boundary_data(periodic_nodes(n), sp.mu, q, f);
  const auto pressure = eval_pressure_regularized(sp.geometry, q, f, st, cp.t, cp.b, 0.04, 7, sp.mu);
  const auto velocity = eval_stokes_regularized(sp.geometry, q, f, st, cp.t, cp.b, 0.04, 7, sp.mu);
  Json stokes = Json::object();
  stokes["closest_t"] = Json(cp.t);
  stokes["b"] = Json(cp.b);
  stokes["pressure"] = Json(pressure);
  std::vector<double> vx, vy;
  for (const Vec2& v : velocity) { vx.push_back(v.x); vy.push_back(v.y); }
  stokes["velocity_x"] = Json(vx);
  stokes["velocity_y"] = Json(vy);
  stokes["starfish_scale"] = starfish_scale();
  stokes["kite_scale"] = kite_scale();
  out["stokes"] = stokes;

  // Grid tools on a 30 x 30 interior grid of the box [-1.5, 1.5]^2.
  const GridCoords grid = grid_coordinates(30, sp.box);
  Grid rhs(30, 30), boundary = Grid::Zero(32, 32), full(32, 32);
  for (int j = 0; j < 32; ++j)
    for (int i = 0; i < 32; ++i) {
      const double x = grid.xx(j, i), y = grid.yy(j, i);
      full(j, i) = std::sin(1.3 * x + 0.4) * std::cos(0.7 * y) + x * y * y;
      if (j == 0 || i == 0 || j == 31 || i == 31) boundary(j, i) = full(j, i);
      if (j > 0 && i > 0 && j < 31 && i < 31) rhs(j - 1, i - 1) = std::cos(x - 2 * y);
    }
  const Grid solved = dst_solve_nine_point(rhs, boundary, grid.h);
  const Gradient gradient = grad4(full, grid.h);
  const Grid lap = nine_point_laplacian(full, grid.h);
  const Mask crossing = classify_crossing(grid, LAPLACE_OFFSETS, sp.inside);
  Json grids = Json::object();
  grids["solved"] = Json(std::vector<double>(solved.data(), solved.data() + solved.size()));
  grids["gx"] = Json(std::vector<double>(gradient.gx.data(), gradient.gx.data() + gradient.gx.size()));
  grids["gy"] = Json(std::vector<double>(gradient.gy.data(), gradient.gy.data() + gradient.gy.size()));
  grids["laplacian"] = Json(std::vector<double>(lap.data(), lap.data() + lap.size()));
  std::vector<int> cross;
  for (int j = 0; j < 30; ++j)
    for (int i = 0; i < 30; ++i) cross.push_back(crossing(j, i) ? 1 : 0);
  grids["crossing"] = Json(cross);
  out["grid"] = grids;

  // Periodic tools.
  std::vector<double> samples(64);
  for (int k = 0; k < 64; ++k) samples[k] = std::exp(std::sin(2 * PI * k / 64));
  const PeriodicSamples ps(samples);
  Json periodic = Json::object();
  periodic["derivative"] = Json(spectral_derivative(samples, 1));
  periodic["second_derivative"] = Json(spectral_derivative(samples, 2));
  periodic["interpolant"] = Json(std::vector<double>{ps(0.1), ps(1.7), ps(4.0)});
  out["periodic"] = periodic;

  // Python float repr.
  Json reprs = Json::array();
  for (double v : {0.1, 1e-5, 1.5e-16, 100.0, 1e16, 123456789012345680.0, 2.0 / 3, -0.0, 0.00012, 1e22})
    reprs.push_back(py_repr(v));
  out["repr"] = reprs;

  std::cout << out.dump(-1) << "\n";
  return 0;
}
