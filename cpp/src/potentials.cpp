#include "regbie/potentials.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "regbie/kernels.hpp"
#include "regbie/layers.hpp"
#include "regbie/periodic.hpp"

namespace regbie {

namespace {

double side_indicator(double b) { return b < 0 ? 1.0 : (b > 0 ? 0.0 : 0.5); }

std::vector<double> zeros_if_empty(const std::vector<double>& v, int n) {
  return v.empty() ? std::vector<double>(n, 0.0) : v;
}
std::vector<Vec2> zeros_if_empty(const std::vector<Vec2>& v, int n) {
  return v.empty() ? std::vector<Vec2>(n, Vec2{0.0, 0.0}) : v;
}

// alpha = sqrt(1 + b kappa) of the local corrections; a target must be within a
// curvature radius of the curve on the concave side.
double alpha(double b, double kappa) {
  const double a2 = 1 + b * kappa;
  if (!(a2 > 0))
    throw std::domain_error(
        "a near target lies beyond the curvature radius of the curve (1 + b*kappa <= 0); "
        "increase n or decrease delta");
  return std::sqrt(a2);
}

}  // namespace

LayerPotentials::LayerPotentials(Curve curve, int n, PotentialOptions options)
    : curve_(std::move(curve)), n_(n), options_(options) {
  if (n < 8) throw std::invalid_argument("n must be at least 8");
  if (options_.order != 3 && options_.order != 5 && options_.order != 7)
    throw std::invalid_argument("order must be 3, 5 or 7");
  if (!(options_.near > 0) || !(options_.mu > 0)) throw std::invalid_argument("near and mu must be positive");
  t_ = periodic_nodes(n);
  nodes_ = sample(curve_.geometry(), t_);
  for (int k = 0; k < n; ++k) spacing_ = std::max(spacing_, norm(nodes_[(k + 1) % n].x - nodes_[k].x));
  delta_ = options_.delta > 0 ? options_.delta : 3 * spacing_;
}

// The curve geometry, answering from the stored nodes at t_k.
Geometry LayerPotentials::node_geometry() const {
  const Geometry g = curve_.geometry();
  const std::vector<GeomPoint>* nodes = &nodes_;
  const int n = n_;
  return [g, nodes, n](double t) {
    const double s = t * n / (2 * PI);
    const double k = std::round(s);
    if (std::fabs(s - k) < 1e-11 && k >= 0 && k < n) return (*nodes)[static_cast<size_t>(k)];
    return g(t);
  };
}

void LayerPotentials::check_size(size_t size, const char* name) const {
  if (static_cast<int>(size) != n_)
    throw std::invalid_argument(std::string(name) + " must have one value per node (n = " + std::to_string(n_) + ")");
}

LayerPotentials::NearTargets LayerPotentials::near_targets(const std::vector<Vec2>& targets) const {
  const auto distance = curve_.sample_distance(targets);
  NearTargets near;
  for (size_t i = 0; i < targets.size(); ++i)
    if (distance[i] < options_.near * delta_) {
      near.index.push_back(i);
      near.points.push_back(targets[i]);
    }
  const Curve::Projection p = curve_.project(near.points);
  near.t = p.t;
  near.b = p.b;
  return near;
}

std::vector<double> LayerPotentials::single_layer(const std::vector<double>& f,
                                                  const std::vector<Vec2>& targets) const {
  check_size(f.size(), "f");
  const NearTargets near = near_targets(targets);
  std::vector<double> out(targets.size());
  // Far targets: sum log(r) f |x'| / n, i.e. the trapezoidal rule for int G f ds.
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum sum;
    for (int k = 0; k < n_; ++k) {
      const Vec2 r = nodes_[k].x - targets[i];
      sum.add(0.5 * std::log(dot(r, r)) * f[k] * nodes_[k].speed);
    }
    out[i] = sum.value() / n_;
  }
  if (!near.index.empty()) {
    const auto value = regbie::single_layer(node_geometry(), f, near.points, near.b, delta_, options_.order);
    for (size_t j = 0; j < near.index.size(); ++j) out[near.index[j]] = value[j];
  }
  return out;
}

std::vector<double> LayerPotentials::double_layer(const std::vector<double>& g,
                                                  const std::vector<Vec2>& targets) const {
  check_size(g.size(), "g");
  const NearTargets near = near_targets(targets);
  std::vector<double> out(targets.size());
  // Far targets: sum (r.n)/r^2 g |x'| / n.
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum sum;
    for (int k = 0; k < n_; ++k) {
      const Vec2 r = nodes_[k].x - targets[i];
      sum.add(dot(r, nodes_[k].normal) / dot(r, r) * g[k] * nodes_[k].speed);
    }
    out[i] = sum.value() / n_;
  }
  if (!near.index.empty()) {
    const PeriodicSamples gi(g);
    const auto value =
        regbie::double_layer(node_geometry(), g, gi.at(near.t), near.points, near.b, delta_, options_.order);
    for (size_t j = 0; j < near.index.size(); ++j) out[near.index[j]] = value[j];
  }
  return out;
}

std::vector<Vec2> LayerPotentials::stokeslet(const std::vector<Vec2>& f, const std::vector<Vec2>& targets) const {
  check_size(f.size(), "f");
  const double mu = options_.mu, d2 = delta_ * delta_;
  const NearTargets near = near_targets(targets);
  std::vector<Vec2> out(targets.size());
  // Far targets: sum (-log(r) f + r (r.f)/r^2)/(4 pi mu) |x'| 2 pi/n.
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum2 sum;
    for (int k = 0; k < n_; ++k) {
      const Vec2 r = nodes_[k].x - targets[i];
      const double r2 = dot(r, r);
      sum.add((-0.5 * std::log(r2) * f[k] + r * (dot(r, f[k]) / r2)) * nodes_[k].speed);
    }
    out[i] = sum.value() / (2.0 * mu * n_);
  }
  if (near.index.empty()) return out;
  // Near targets: regularized Stokeslet plus the correction |b| N0 n0 (n0.f0)/(4 pi mu alpha).
  const PeriodicSamples fi(f);
  const auto f0 = fi.vec_at(near.t);
  const auto close = sample(node_geometry(), near.t);
#pragma omp parallel for schedule(dynamic, 16)
  for (long j = 0; j < static_cast<long>(near.index.size()); ++j) {
    const double b = near.b[j], lam = std::fabs(b) / delta_;
    const Coefficients cs = coefficients(options_.order, lam);
    Sum2 sum;
    for (int k = 0; k < n_; ++k) {
      const Vec2 r = nodes_[k].x - near.points[j];
      const double r2 = dot(r, r);
      const RadialFactors rf = radial_factors(r2 / d2, cs);
      const double gf = regularized_kernel(std::sqrt(r2), delta_, cs);
      const Vec2 term = r * dot(r, f[k]) * rf.a / (4 * PI * d2) - gf * f[k] / 2;
      sum.add(term * (2 * PI * nodes_[k].speed / n_));
    }
    const GeomPoint& c = close[j];
    const double scale = std::fabs(b) * scaled_moment_n0(options_.order, lam);
    const Vec2 b0 = c.normal * dot(c.normal, f0[j]) / alpha(b, c.kappa);
    out[near.index[j]] = sum.value() / mu + scale * b0 / (4 * PI * mu);
  }
  return out;
}

std::vector<Vec2> LayerPotentials::stresslet(const std::vector<Vec2>& q, const std::vector<Vec2>& targets) const {
  check_size(q.size(), "q");
  const double d4 = std::pow(delta_, 4);
  const NearTargets near = near_targets(targets);
  std::vector<Vec2> out(targets.size());
  // Far targets: sum r (r.q)(r.n)/(pi r^4) |x'| 2 pi/n.
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum2 sum;
    for (int k = 0; k < n_; ++k) {
      const Vec2 r = nodes_[k].x - targets[i];
      const double r2 = dot(r, r);
      sum.add(r * (dot(r, q[k]) * dot(r, nodes_[k].normal) / (r2 * r2)) * nodes_[k].speed);
    }
    out[i] = sum.value() * (2.0 / n_);
  }
  if (near.index.empty()) return out;
  // Near targets: subtracted regularized stresslet, chi q0, and |b| N0 c2/(2 pi).
  std::vector<Vec2> q_s = spectral_derivative(q);
  for (int k = 0; k < n_; ++k) q_s[k] = q_s[k] / nodes_[k].speed;
  std::vector<Vec2> q_ss = spectral_derivative(q_s);
  for (int k = 0; k < n_; ++k) q_ss[k] = q_ss[k] / nodes_[k].speed;
  const auto q0 = PeriodicSamples(q).vec_at(near.t);
  const auto qs0 = PeriodicSamples(q_s).vec_at(near.t);
  const auto qss0 = PeriodicSamples(q_ss).vec_at(near.t);
  const auto close = sample(node_geometry(), near.t);
#pragma omp parallel for schedule(dynamic, 16)
  for (long j = 0; j < static_cast<long>(near.index.size()); ++j) {
    const double b = near.b[j], lam = std::fabs(b) / delta_;
    const Coefficients cs = coefficients(options_.order, lam);
    Sum2 sum;
    for (int k = 0; k < n_; ++k) {
      const Vec2 r = nodes_[k].x - near.points[j];
      const RadialFactors rf = radial_factors(dot(r, r) / (delta_ * delta_), cs);
      const Vec2 term = r * (dot(r, q[k] - q0[j]) * dot(r, nodes_[k].normal) * (-rf.ap) / (PI * d4));
      sum.add(term * (2 * PI * nodes_[k].speed / n_));
    }
    const GeomPoint& c = close[j];
    const double a = alpha(b, c.kappa), a3 = a * a * a, a5 = a3 * a * a;
    const double scale = std::fabs(b) * scaled_moment_n0(options_.order, lam);
    const Vec2 sym = c.normal * dot(c.tau, qs0[j]) + c.tau * dot(c.normal, qs0[j]);
    Vec2 c2 = sym / a3;
    c2 -= (b / (2 * a3)) * c.normal * dot(c.normal, qss0[j]);
    c2 += (b * b * c.kappa_s / (2 * a5)) * c.normal * dot(c.normal, qs0[j]);
    out[near.index[j]] = sum.value() + side_indicator(b) * q0[j] + scale * c2 / (2 * PI);
  }
  return out;
}

std::vector<double> LayerPotentials::stokes_pressure(const std::vector<Vec2>& f_in, const std::vector<Vec2>& q_in,
                                                     const std::vector<Vec2>& targets) const {
  const auto f = zeros_if_empty(f_in, n_), q = zeros_if_empty(q_in, n_);
  check_size(f.size(), "f");
  check_size(q.size(), "q");
  // p = S phi - D psi with the pressure densities phi = d/ds(f.tau) - 2 mu d/ds(q_s.n),
  // psi = f.n + 2 mu q_s.tau (Laplace layers).
  const PressureBoundaryData data = pressure_densities(node_geometry(), q, f, options_.mu);
  const auto sl = single_layer(data.phi, targets);
  const auto dl = double_layer(data.psi, targets);
  std::vector<double> out(targets.size());
  for (size_t i = 0; i < out.size(); ++i) out[i] = sl[i] - dl[i];
  return out;
}

}  // namespace regbie
