// Layer potentials by the trapezoidal rule with regularized kernels, and the
// Laplace, pressure and Stokes evaluators used by the experiments.
// C++ counterparts of regbie/regularized_slp.single_layer,
// regbie/regularized_dlp.double_layer, experiments/bie_evaluators.py and the
// unregularized sums in experiments/allgrid.py.
#pragma once
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/periodic.hpp"
#include "regbie/types.hpp"

namespace regbie {

// S_delta f at the targets: sum_k G_delta(|y - x_k|) f_k |x'(t_k)| 2 pi/n.
// density holds f(t_k) per unit arclength at the n nodes; b is the signed
// distance of each target (b = 0 on the curve). The kernel coefficients use |b|/delta.
std::vector<double> single_layer(const Geometry& curve, const std::vector<double>& density,
                                 const std::vector<Vec2>& targets, const std::vector<double>& b,
                                 double delta, int order = 7);

// D_delta g with density subtraction: chi g(t0) + sum_k (r.n) T/r^2 (g_k - g(t0)) |x'| / n,
// r = x_k - y. g0 holds g at the closest parameter of each target. On the curve
// this is the average of the one-sided limits. subtract = false integrates g directly.
std::vector<double> double_layer(const Geometry& curve, const std::vector<double>& density,
                                 const std::vector<double>& g0, const std::vector<Vec2>& targets,
                                 const std::vector<double>& b, double delta, int order = 7,
                                 bool subtract = true);

// Laplace S f + D g (bie_evaluators.eval_laplace_regularized); f, g are
// evaluated at the n_gamma nodes and g at closest_t.
std::vector<double> eval_laplace_regularized(const Geometry& geometry, const PeriodicSamples& f,
                                             const PeriodicSamples& g, const std::vector<Vec2>& targets,
                                             const std::vector<double>& closest_t,
                                             const std::vector<double>& b, double delta, int n_gamma,
                                             int order);

// Pressure S phi_p - D psi_p from velocity and traction samples q, f (eval_pressure_regularized).
std::vector<double> eval_pressure_regularized(const Geometry& geometry, const std::vector<Vec2>& q,
                                              const std::vector<Vec2>& f, const std::vector<Vec2>& targets,
                                              const std::vector<double>& closest_t,
                                              const std::vector<double>& b, double delta, int order,
                                              double mu = 1.0);

// Corrected Stokes velocity S f + D q (eval_stokes_regularized).
std::vector<Vec2> eval_stokes_regularized(const Geometry& geometry, const std::vector<Vec2>& q,
                                          const std::vector<Vec2>& f, const std::vector<Vec2>& targets,
                                          const std::vector<double>& closest_t,
                                          const std::vector<double>& b, double delta, int order,
                                          double mu = 1.0);

// Unregularized trapezoidal sums for targets far from the curve (allgrid._unregularized_*).
// Laplace: sum log r f |x'| / n + d_sign * sum (r.n)/r^2 g |x'| / n.
std::vector<double> unregularized_laplace(const Geometry& geometry, const PeriodicSamples& f,
                                          const PeriodicSamples& g, const std::vector<Vec2>& targets,
                                          int n, double d_sign = 1.0);
std::vector<Vec2> unregularized_stokes(const Geometry& geometry, const std::vector<Vec2>& q,
                                       const std::vector<Vec2>& f, const std::vector<Vec2>& targets,
                                       double mu = 1.0);

}  // namespace regbie
