// Laplace and Stokes layer potentials at arbitrary targets: the user interface.
//
// LayerPotentials discretizes a Curve with n equally spaced parameters
// t_k = 2*pi*k/n and evaluates layer potentials of densities given at those
// nodes. Targets may lie anywhere: on the curve, near it, or far from it.
// Targets within `near * delta` of the curve use the regularized kernels of
// order p with the local corrections (the error is O(delta^p) plus a
// negligible quadrature error); the others use the trapezoidal rule with the
// original kernels. Closest points and signed distances are computed here.
//
// Conventions. G(r) = log(r)/(2*pi); r = x - y with x on the curve and y the
// target; outward normal n; densities per unit arclength. For u harmonic
// inside the curve, S(-du/dn) + D(u) equals u inside and 0 outside. For a
// Stokes flow (u, p) inside with viscosity mu, velocity q = u and traction
// f = sigma n on the curve, S f + D q equals u inside and 0 outside, and
// stokes_pressure(f, q) equals p inside and 0 outside. On the curve the double
// layers return the average of the two one-sided limits.
#pragma once
#include <vector>

#include "regbie/curve.hpp"
#include "regbie/types.hpp"

namespace regbie {

struct PotentialOptions {
  int order = 7;      // 3, 5 or 7: the regularization error is O(delta^order)
  double delta = 0;   // smoothing width; 0 selects 3 times the largest node spacing
  double near = 6;    // targets closer than near*delta to the curve are treated as near
  double mu = 1;      // viscosity, for the Stokes routines
};

class LayerPotentials {
 public:
  LayerPotentials(Curve curve, int n, PotentialOptions options = {});

  int n() const { return n_; }
  double delta() const { return delta_; }
  const PotentialOptions& options() const { return options_; }
  const Curve& curve() const { return curve_; }
  // The parameters t_k and the curve data at them (x, |x'|, tangent, normal, curvature).
  const std::vector<double>& parameters() const { return t_; }
  const std::vector<GeomPoint>& nodes() const { return nodes_; }
  // Largest distance between neighbouring nodes.
  double spacing() const { return spacing_; }

  // Laplace single layer S f = int G f ds and double layer D g = int dG/dn_x g ds.
  std::vector<double> single_layer(const std::vector<double>& f, const std::vector<Vec2>& targets) const;
  std::vector<double> double_layer(const std::vector<double>& g, const std::vector<Vec2>& targets) const;

  // Stokes single layer (Stokeslet) S f and double layer (stresslet) D q, and the
  // pressure of S f + D q (pass an empty vector for a density that is zero).
  std::vector<Vec2> stokeslet(const std::vector<Vec2>& f, const std::vector<Vec2>& targets) const;
  std::vector<Vec2> stresslet(const std::vector<Vec2>& q, const std::vector<Vec2>& targets) const;
  std::vector<double> stokes_pressure(const std::vector<Vec2>& f, const std::vector<Vec2>& q,
                                      const std::vector<Vec2>& targets) const;

  // The near targets (indices into `targets`) with their closest parameters and signed distances.
  struct NearTargets {
    std::vector<size_t> index;
    std::vector<Vec2> points;
    std::vector<double> t, b;
  };
  NearTargets near_targets(const std::vector<Vec2>& targets) const;

 private:
  Geometry node_geometry() const;
  void check_size(size_t size, const char* name) const;

  Curve curve_;
  int n_;
  PotentialOptions options_;
  double delta_ = 0, spacing_ = 0;
  std::vector<double> t_;
  std::vector<GeomPoint> nodes_;
};

}  // namespace regbie
