// Test problems: interface geometry, densities and exact solutions
// (experiments/allgrid.LaplaceProblem/StokesProblem, experiments/swapped_low_order.py).
#pragma once
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/grid.hpp"

namespace regbie {

// Which branch of a piecewise exact solution to evaluate (side=None, "inside", "outside", "average").
enum class Side { Piecewise, Inside, Outside, Average };

struct LaplaceProblem {
  std::string name;
  Geometry geometry;
  InsideFn inside;
  // f = [du/dn] and g = [u] (inside minus outside) at parameters t.
  std::function<void(const std::vector<double>& t, std::vector<double>& f, std::vector<double>& g)>
      boundary_densities;
  std::function<double(Vec2, Side)> exact;
  Box box;
};

struct StokesSolution {
  Vec2 u;
  double p;
};

struct StokesProblem {
  std::string name;
  Geometry geometry;
  InsideFn inside;
  // Velocity q and traction f = sigma n on the curve at parameters t.
  std::function<void(const std::vector<double>& t, double mu, std::vector<Vec2>& q, std::vector<Vec2>& f)>
      boundary_data;
  std::function<StokesSolution(Vec2, Side, double mu)> exact;
  std::function<Vec2(Vec2)> exact_gradient;  // piecewise grad p (zero outside)
  Box box;
  double mu = 1.0;
};

// Scaled kite (Laplace) and scaled five-lobed starfish (Stokes), box [-1.5, 1.5]^2.
const LaplaceProblem& laplace_kite();
const StokesProblem& stokes_starfish();

// The scaled geometries and the coordinate scales (swapped_low_order).
double kite_scale();
double starfish_scale();
GeomPoint laplace_kite_geometry(double t);
GeomPoint stokes_starfish_geometry(double t);

// Exact fields of the Stokes test, inside the starfish (swapped_low_order.stokes_*).
Vec2 stokes_velocity(Vec2 p);
double stokes_pressure(Vec2 p, double mu = 1.0);
Vec2 stokes_grad_pressure(Vec2 p, double mu = 1.0);
// grad u as [[du1/dx, du1/dy], [du2/dx, du2/dy]].
std::array<std::array<double, 2>, 2> stokes_grad_velocity(Vec2 p);

}  // namespace regbie
