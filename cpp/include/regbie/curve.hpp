// A smooth closed curve for the layer-potential routines.
//
// The curve is parameterized by t in [0, 2*pi) and must be counterclockwise,
// so that normal = (tau_y, -tau_x) points outward. Build it from
//   - an analytic frame (from_function),
//   - x(t) and its first three derivatives (from_derivatives), or
//   - N points x(2*pi*k/N) (from_points); the parameterization is then the
//     trigonometric interpolant of the points, differentiated by FFT.
#pragma once
#include <array>
#include <functional>
#include <memory>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/types.hpp"

namespace regbie {

class Curve {
 public:
  // geometry(t) returns x, |x'|, unit tangent, outward normal, curvature and d(kappa)/ds.
  static Curve from_function(Geometry geometry);
  // derivatives(t) returns {x, x', x'', x'''}.
  static Curve from_derivatives(std::function<std::array<Vec2, 4>(double)> derivatives);
  // At least 8 points of a smooth closed curve at equally spaced parameters, counterclockwise.
  static Curve from_points(const std::vector<Vec2>& points);

  GeomPoint operator()(double t) const { return geometry_(t); }
  const Geometry& geometry() const { return geometry_; }
  Vec2 second_derivative(double t) const { return second_(t); }

  // Point-in-curve test (polygon through dense samples of the curve).
  bool inside(Vec2 p) const;

  // Closest parameter t and signed distance b (negative inside, positive outside)
  // of each point. Intended for points within a curvature radius of the curve,
  // where the closest point is unique.
  struct Projection {
    std::vector<double> t, b;
  };
  Projection project(const std::vector<Vec2>& points) const;

  // Distance from each point to the nearest of the dense samples (a cheap upper
  // bound on the distance to the curve, used to separate near and far targets).
  std::vector<double> sample_distance(const std::vector<Vec2>& points) const;

 private:
  Curve() = default;
  void build_samples();

  Geometry geometry_;
  std::function<Vec2(double)> second_;
  std::shared_ptr<const CurveInside> inside_;
  std::vector<double> sample_t_;
  std::vector<Vec2> sample_x_;
};

}  // namespace regbie
