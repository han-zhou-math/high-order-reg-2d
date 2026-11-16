// Curves and their Frenet data, point-in-curve tests and closest points.
// C++ counterparts of experiments/geometries.py, the curves in
// regbie/regularized_slp.py, and allgrid.closest_points.
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "regbie/types.hpp"

namespace regbie {

// Values returned by the Python geometry(t): x, speed, tau, normal, kappa, kappa_s.
// Normals are outward for counterclockwise curves: normal = (tau_y, -tau_x).
struct GeomPoint {
  Vec2 x;
  double speed = 0;
  Vec2 tau, normal;
  double kappa = 0, kappa_s = 0;
};
using Geometry = std::function<GeomPoint(double)>;

// Frame from the derivatives of a parameterization (geometries._frame).
GeomPoint frame(Vec2 x, Vec2 dx, Vec2 ddx, Vec2 dddx);

// Geometries of experiments/geometries.py.
GeomPoint circle_geometry(double t);
GeomPoint laplace_starfish_geometry(double t);  // r = 1 + 0.3 cos 5t
GeomPoint stokes_kite_geometry(double t);       // (cos t + 0.65 cos 2t - 0.65, 1.5 sin t)
double starfish_level(Vec2 p);                  // |p| - (1 + 0.3 cos 5 theta)

// Curves of regbie/regularized_slp.py (circle and ellipse a=1.4, b=0.8; starfish r = 1 + 0.2 cos 3t).
GeomPoint regbie_ellipse(double t, double a = 1.4, double b = 0.8);
GeomPoint regbie_circle(double t);
GeomPoint regbie_starfish(double t);

// geometry(t) evaluated at the n periodic nodes.
std::vector<GeomPoint> sample(const Geometry& geometry, int n);
std::vector<GeomPoint> sample(const Geometry& geometry, const std::vector<double>& t);

// Uniformly scaled geometry: x, speed scale by s; kappa by 1/s, kappa_s by 1/s^2.
Geometry scaled_geometry(Geometry geometry, double scale);

// radius / max_k max(|x_1(t_k)|, |x_2(t_k)|) over 65536 nodes (swapped_low_order._coordinate_scale).
double coordinate_scale(const Geometry& geometry, double radius = 0.895);

// Point-in-polygon test for the curve sampled at `samples` nodes, matching
// matplotlib Path(polygon, closed=True).contains_points: the last sample is
// replaced by the closing vertex, so the polygon has samples-1 vertices.
// Edges are binned in y so that each query touches only nearby edges.
class CurveInside {
 public:
  CurveInside(const Geometry& geometry, int samples = 4096);
  bool operator()(Vec2 p) const;

 private:
  std::vector<Vec2> v_;
  double ymin_ = 0, ymax_ = 0, bin_height_ = 1;
  std::vector<std::vector<int>> bins_;  // edge k joins v_[k] and v_[(k+1) % size]
};
using InsideFn = std::function<bool(Vec2)>;

// Minimum distance from a complex pole (px, py) to the curve (geometries.minimum_pole_distance).
double minimum_pole_distance(Vec2 pole, const Geometry& geometry, int samples = 4096);

// Closest parameters and signed distances (positive outside) for targets near the
// curve: nearest of `samples` nodes, then Newton steps (allgrid.closest_points).
// Throws if the sign disagrees with `inside` away from the curve.
struct ClosestPoints {
  std::vector<double> t, b;
};
ClosestPoints closest_points(const std::vector<Vec2>& targets, const Geometry& geometry,
                             const InsideFn& inside, int samples = 4096);

}  // namespace regbie
