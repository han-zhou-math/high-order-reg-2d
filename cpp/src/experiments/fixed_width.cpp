#include "regbie/experiments/fixed_width.hpp"

#include <stdexcept>

#include "regbie/layers.hpp"
#include "regbie/experiments/local_experiments.hpp"
#include "regbie/periodic.hpp"
#include "regbie/experiments/problems.hpp"

namespace regbie {

FieldValues fixed_width_evaluate(const std::string& label, const std::vector<Vec2>& targets,
                                 const std::vector<double>& theta, const std::vector<double>& b, double delta, int n,
                                 const std::string& /*side*/) {
  const auto t = periodic_nodes(n);
  FieldValues out;
  if (label == "kite_laplace") {
    const LaplaceProblem& problem = laplace_kite();
    std::vector<double> f, g;
    problem.boundary_densities(t, f, g);
    out.laplace = eval_laplace_regularized(problem.geometry, PeriodicSamples(f), PeriodicSamples(g), targets, theta,
                                           b, delta, n, 7);
  } else if (label == "starfish_stokes") {
    const StokesProblem& problem = stokes_starfish();
    std::vector<Vec2> q, f;
    problem.boundary_data(t, 1.0, q, f);
    out.velocity = eval_stokes_regularized(problem.geometry, q, f, targets, theta, b, delta, 7);
    out.pressure = eval_pressure_regularized(problem.geometry, q, f, targets, theta, b, delta, 7);
  } else {
    throw std::invalid_argument("fixed_width_evaluate: unknown label " + label);
  }
  return out;
}

CircleLayerValues fixed_width_evaluate_circle(const std::vector<Vec2>& targets, const std::vector<double>& theta,
                                              const std::vector<double>& b, double delta, int n) {
  // single_layer samples the density at the nodes; double_layer subtracts g(theta).
  const auto t = periodic_nodes(n);
  std::vector<double> density(n), g0(theta.size());
  for (int k = 0; k < n; ++k) density[k] = circle_density(t[k]);
  for (size_t i = 0; i < theta.size(); ++i) g0[i] = circle_density(theta[i]);
  CircleLayerValues out;
  out.single = single_layer(circle_geometry, density, targets, b, delta, 7);
  out.double_layer = double_layer(circle_geometry, density, g0, targets, b, delta, 7);
  return out;
}

}  // namespace regbie
