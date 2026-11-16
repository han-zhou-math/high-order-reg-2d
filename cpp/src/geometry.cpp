#include "regbie/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "regbie/numerics.hpp"

namespace regbie {

GeomPoint frame(Vec2 x, Vec2 dx, Vec2 ddx, Vec2 dddx) {
  GeomPoint g;
  g.x = x;
  g.speed = norm(dx);
  g.tau = dx / g.speed;
  g.normal = {g.tau.y, -g.tau.x};
  const double s = g.speed;
  g.kappa = cross(dx, ddx) / (s * s * s);
  g.kappa_s = cross(dx, dddx) / (s * s * s * s) - 3 * cross(dx, ddx) * dot(dx, ddx) / std::pow(s, 6);
  return g;
}

GeomPoint circle_geometry(double t) {
  const Vec2 x{std::cos(t), std::sin(t)};
  const Vec2 dx{-std::sin(t), std::cos(t)};
  return frame(x, dx, -x, -dx);
}

namespace {
// Polar curve r(t)(cos t, sin t) with the first three derivatives of r.
GeomPoint polar(double t, double r, double rp, double rpp, double rppp) {
  const Vec2 e{std::cos(t), std::sin(t)}, ep{-std::sin(t), std::cos(t)};
  const Vec2 x = r * e;
  const Vec2 dx = rp * e + r * ep;
  const Vec2 ddx = (rpp - r) * e + (2 * rp) * ep;
  const Vec2 dddx = (rppp - 3 * rp) * e + (3 * rpp - r) * ep;
  return frame(x, dx, ddx, dddx);
}
}  // namespace

GeomPoint laplace_starfish_geometry(double t) {
  const double amplitude = 0.3;
  const int lobes = 5;
  const double r = 1.0 + amplitude * std::cos(lobes * t);
  const double rp = -amplitude * lobes * std::sin(lobes * t);
  const double rpp = -amplitude * lobes * lobes * std::cos(lobes * t);
  const double rppp = amplitude * lobes * lobes * lobes * std::sin(lobes * t);
  return polar(t, r, rp, rpp, rppp);
}

double starfish_level(Vec2 p) {
  const double theta = std::atan2(p.y, p.x);
  return norm(p) - (1.0 + 0.3 * std::cos(5 * theta));
}

GeomPoint stokes_kite_geometry(double t) {
  const double a = 0.65, sy = 1.5;
  const Vec2 x{std::cos(t) + a * std::cos(2 * t) - a, sy * std::sin(t)};
  const Vec2 dx{-std::sin(t) - 2 * a * std::sin(2 * t), sy * std::cos(t)};
  const Vec2 ddx{-std::cos(t) - 4 * a * std::cos(2 * t), -sy * std::sin(t)};
  const Vec2 dddx{std::sin(t) + 8 * a * std::sin(2 * t), -sy * std::cos(t)};
  return frame(x, dx, ddx, dddx);
}

GeomPoint regbie_ellipse(double t, double a, double b) {
  const Vec2 x{a * std::cos(t), b * std::sin(t)};
  const Vec2 dx{-a * std::sin(t), b * std::cos(t)};
  return frame(x, dx, -x, -dx);
}

GeomPoint regbie_circle(double t) { return regbie_ellipse(t, 1.0, 1.0); }

GeomPoint regbie_starfish(double t) {
  const double r = 1 + 0.2 * std::cos(3 * t);
  return polar(t, r, -0.6 * std::sin(3 * t), -1.8 * std::cos(3 * t), 5.4 * std::sin(3 * t));
}

std::vector<GeomPoint> sample(const Geometry& geometry, int n) {
  return sample(geometry, periodic_nodes(n));
}

std::vector<GeomPoint> sample(const Geometry& geometry, const std::vector<double>& t) {
  std::vector<GeomPoint> out(t.size());
  for (size_t k = 0; k < t.size(); ++k) out[k] = geometry(t[k]);
  return out;
}

Geometry scaled_geometry(Geometry geometry, double scale) {
  return [geometry, scale](double t) {
    GeomPoint g = geometry(t);
    g.x = scale * g.x;
    g.speed = scale * g.speed;
    g.kappa = g.kappa / scale;
    g.kappa_s = g.kappa_s / (scale * scale);
    return g;
  };
}

double coordinate_scale(const Geometry& geometry, double radius) {
  const int n = 65536;
  double largest = 0;
  for (int k = 0; k < n; ++k) {
    const GeomPoint g = geometry(2.0 * PI * k / n);
    largest = std::max({largest, std::fabs(g.x.x), std::fabs(g.x.y)});
  }
  return radius / largest;
}

CurveInside::CurveInside(const Geometry& geometry, int samples) {
  if (samples < 4) throw std::invalid_argument("need at least four samples");
  // matplotlib replaces the CLOSEPOLY vertex (the last sample) by the first one.
  v_.resize(samples - 1);
  for (int k = 0; k < samples - 1; ++k) v_[k] = geometry(2.0 * PI * k / samples).x;
  ymin_ = ymax_ = v_[0].y;
  for (const Vec2& p : v_) {
    ymin_ = std::min(ymin_, p.y);
    ymax_ = std::max(ymax_, p.y);
  }
  const int nbins = std::max(1, static_cast<int>(v_.size()) / 4);
  bin_height_ = (ymax_ - ymin_) / nbins;
  if (!(bin_height_ > 0)) bin_height_ = 1;
  bins_.assign(nbins, {});
  const int size = static_cast<int>(v_.size());
  for (int k = 0; k < size; ++k) {
    const Vec2 a = v_[k], b = v_[(k + 1) % size];
    const double lo = std::min(a.y, b.y), hi = std::max(a.y, b.y);
    int i0 = static_cast<int>(std::floor((lo - ymin_) / bin_height_)) - 1;
    int i1 = static_cast<int>(std::floor((hi - ymin_) / bin_height_)) + 1;
    i0 = std::max(i0, 0);
    i1 = std::min(i1, nbins - 1);
    for (int i = i0; i <= i1; ++i) bins_[i].push_back(k);
  }
}

bool CurveInside::operator()(Vec2 p) const {
  const double tx = p.x, ty = p.y;
  if (!(ty >= ymin_ - 1e-12 && ty <= ymax_ + 1e-12)) return false;
  int bin = static_cast<int>(std::floor((ty - ymin_) / bin_height_));
  bin = std::min(std::max(bin, 0), static_cast<int>(bins_.size()) - 1);
  const int size = static_cast<int>(v_.size());
  bool inside = false;
  // Crossing rule of matplotlib's point_in_path; the parity does not depend on edge order.
  for (int k : bins_[bin]) {
    const Vec2 v0 = v_[k], v1 = v_[(k + 1) % size];
    const bool yflag0 = (v0.y >= ty), yflag1 = (v1.y >= ty);
    if (yflag0 != yflag1) {
      if (((v1.y - ty) * (v0.x - v1.x) >= (v1.x - tx) * (v0.y - v1.y)) == yflag1) inside = !inside;
    }
  }
  return inside;
}

double minimum_pole_distance(Vec2 pole, const Geometry& geometry, int samples) {
  double best = std::numeric_limits<double>::infinity();
  double seed = 0;
  for (int k = 0; k < samples; ++k) {
    const double t = 2.0 * PI * k / samples;
    const double d = norm(geometry(t).x - pole);
    if (d < best) { best = d; seed = t; }
  }
  const double width = 2.0 * PI / samples;
  auto objective = [&](double t) {
    double tm = std::fmod(t, 2.0 * PI);
    if (tm < 0) tm += 2.0 * PI;
    const double d = norm(geometry(tm).x - pole);
    return d * d;
  };
  return std::sqrt(minimize_scalar_bounded(objective, seed - width, seed + width).fun);
}

ClosestPoints closest_points(const std::vector<Vec2>& targets, const Geometry& geometry,
                             const InsideFn& inside, int samples) {
  const size_t m = targets.size();
  std::vector<double> ts = periodic_nodes(samples);
  std::vector<Vec2> xs(samples);
  for (int k = 0; k < samples; ++k) xs[k] = geometry(ts[k]).x;
  ClosestPoints out;
  out.t.resize(m);
  out.b.resize(m);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(m); ++i) {
    double best = std::numeric_limits<double>::infinity();
    int index = 0;
    for (int k = 0; k < samples; ++k) {
      const Vec2 d = xs[k] - targets[i];
      const double d2 = dot(d, d);
      if (d2 < best) { best = d2; index = k; }
    }
    out.t[i] = ts[index];
  }
  // Newton iterations on all targets together; stop when every step is tiny.
  const double eps = 2e-6;
  for (int iteration = 0; iteration < 12; ++iteration) {
    double largest = 0;
#pragma omp parallel for schedule(static) reduction(max : largest)
    for (long i = 0; i < static_cast<long>(m); ++i) {
      const double t = out.t[i];
      const GeomPoint g = geometry(t);
      const Vec2 dx = g.speed * g.tau;
      const GeomPoint gp = geometry(t + eps), gm = geometry(t - eps);
      const Vec2 ddx = (gp.speed * gp.tau - gm.speed * gm.tau) / (2 * eps);
      const Vec2 r = g.x - targets[i];
      const double step = dot(r, dx) / (dot(dx, dx) + dot(r, ddx));
      double tn = std::fmod(t - step, 2.0 * PI);
      if (tn < 0) tn += 2.0 * PI;
      out.t[i] = tn;
      largest = std::max(largest, std::fabs(step));
    }
    if (largest < 2e-13) break;
  }
  bool disagree = false;
#pragma omp parallel for schedule(static) reduction(|| : disagree)
  for (long i = 0; i < static_cast<long>(m); ++i) {
    const GeomPoint g = geometry(out.t[i]);
    double b = dot(targets[i] - g.x, g.normal);
    // Cartesian nodes can lie on the curve to roundoff; their offset is exactly zero.
    const bool on_surface = std::fabs(b) <= 5e-13 * std::max(1.0, norm(targets[i]));
    if (on_surface) b = 0.0;
    out.b[i] = b;
    if (!on_surface && ((b < 0) != inside(targets[i]))) disagree = true;
  }
  if (disagree) throw std::runtime_error("closest-point sign disagrees with phase classifier");
  return out;
}

}  // namespace regbie
