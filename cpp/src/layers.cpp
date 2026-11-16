#include "regbie/layers.hpp"

#include <cmath>
#include <stdexcept>

#include "regbie/kernels.hpp"

namespace regbie {

namespace {

// Nodes of the curve adapter: x and dx = speed * tau, as bie_evaluators.curve_adapter.
struct Nodes {
  std::vector<Vec2> x, dx;
};

Nodes adapter_nodes(const Geometry& curve, int n) {
  Nodes nodes;
  nodes.x.resize(n);
  nodes.dx.resize(n);
  const auto t = periodic_nodes(n);
  for (int k = 0; k < n; ++k) {
    const GeomPoint g = curve(t[k]);
    nodes.x[k] = g.x;
    nodes.dx[k] = g.speed * g.tau;
  }
  return nodes;
}

double side_indicator(double b) { return b < 0 ? 1.0 : (b > 0 ? 0.0 : 0.5); }

void check_targets(size_t targets, size_t b) {
  if (targets != b) throw std::invalid_argument("one signed distance per target is required");
}

}  // namespace

std::vector<double> single_layer(const Geometry& curve, const std::vector<double>& density,
                                 const std::vector<Vec2>& targets, const std::vector<double>& b,
                                 double delta, int order) {
  const int n = static_cast<int>(density.size());
  if (n < 4) throw std::invalid_argument("n must be an integer >= 4");
  check_targets(targets.size(), b.size());
  const Nodes nodes = adapter_nodes(curve, n);
  std::vector<double> weights(n);
  for (int k = 0; k < n; ++k) weights[k] = density[k] * norm(nodes.dx[k]) * 2 * PI / n;
  std::vector<double> result(targets.size());
#pragma omp parallel for schedule(dynamic, 16)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    const Coefficients cs = coefficients(order, std::fabs(b[i]) / delta);
    Sum sum;
    for (int k = 0; k < n; ++k) sum.add(regularized_kernel(norm(targets[i] - nodes.x[k]), delta, cs) * weights[k]);
    result[i] = sum.value();
  }
  return result;
}

std::vector<double> double_layer(const Geometry& curve, const std::vector<double>& density,
                                 const std::vector<double>& g0_in, const std::vector<Vec2>& targets,
                                 const std::vector<double>& b, double delta, int order, bool subtract) {
  const int n = static_cast<int>(density.size());
  if (n < 4) throw std::invalid_argument("n must be an integer >= 4");
  check_targets(targets.size(), b.size());
  if (subtract && g0_in.size() != targets.size()) throw std::invalid_argument("g0 needs one value per target");
  const Nodes nodes = adapter_nodes(curve, n);
  std::vector<Vec2> normal(n);
  std::vector<double> weights(n);
  for (int k = 0; k < n; ++k) {
    const double speed = norm(nodes.dx[k]);
    if (!(speed > 0)) throw std::invalid_argument("curve must be regular");
    normal[k] = Vec2{nodes.dx[k].y, -nodes.dx[k].x} / speed;
    weights[k] = speed / n;  // includes dt/(2 pi)
  }
  std::vector<double> answer(targets.size());
#pragma omp parallel for schedule(dynamic, 16)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    const double g0 = subtract ? g0_in[i] : 0.0;
    const Coefficients cs = coefficients(order, std::fabs(b[i]) / delta);
    Sum sum;
    for (int k = 0; k < n; ++k) {
      const Vec2 r = nodes.x[k] - targets[i];
      const double factor = smoothing_over_r2(dot(r, r), delta, cs);
      sum.add(dot(r, normal[k]) * factor * (density[k] - g0) * weights[k]);
    }
    answer[i] = side_indicator(b[i]) * g0 + sum.value();
  }
  return answer;
}

std::vector<double> eval_laplace_regularized(const Geometry& geometry, const PeriodicSamples& f,
                                             const PeriodicSamples& g, const std::vector<Vec2>& targets,
                                             const std::vector<double>& closest_t,
                                             const std::vector<double>& b, double delta, int n_gamma,
                                             int order) {
  const auto sl = single_layer(geometry, f.at_grid(n_gamma), targets, b, delta, order);
  const auto dl = double_layer(geometry, g.at_grid(n_gamma), g.at(closest_t), targets, b, delta, order);
  std::vector<double> out(sl.size());
  for (size_t i = 0; i < sl.size(); ++i) out[i] = sl[i] + dl[i];
  return out;
}

std::vector<double> eval_pressure_regularized(const Geometry& geometry, const std::vector<Vec2>& q,
                                              const std::vector<Vec2>& f, const std::vector<Vec2>& targets,
                                              const std::vector<double>& closest_t,
                                              const std::vector<double>& b, double delta, int order,
                                              double mu) {
  const PressureBoundaryData data = pressure_densities(geometry, q, f, mu);
  const PeriodicSamples phi(data.phi), psi(data.psi);
  const int n = static_cast<int>(q.size());
  const auto sl = single_layer(geometry, phi.at_grid(n), targets, b, delta, order);
  const auto dl = double_layer(geometry, psi.at_grid(n), psi.at(closest_t), targets, b, delta, order);
  std::vector<double> out(sl.size());
  for (size_t i = 0; i < sl.size(); ++i) out[i] = sl[i] - dl[i];
  return out;
}

std::vector<Vec2> eval_stokes_regularized(const Geometry& geometry, const std::vector<Vec2>& q,
                                          const std::vector<Vec2>& f, const std::vector<Vec2>& targets,
                                          const std::vector<double>& closest_t,
                                          const std::vector<double>& b, double delta, int order,
                                          double mu) {
  check_targets(targets.size(), b.size());
  const int n = static_cast<int>(q.size());
  const auto nodes = sample(geometry, n);
  const PressureBoundaryData pdata = pressure_densities(geometry, q, f, mu);
  const PeriodicSamples q_interp(q), f_interp(f), qs_interp(pdata.q_s), qss_interp(pdata.q_ss);
  const auto close = sample(geometry, closest_t);
  const auto q0 = q_interp.vec_at(closest_t), f0 = f_interp.vec_at(closest_t);
  const auto q_s = qs_interp.vec_at(closest_t), q_ss = qss_interp.vec_at(closest_t);
  std::vector<double> weights(n);
  for (int k = 0; k < n; ++k) weights[k] = 2 * PI * nodes[k].speed / n;
  const double d2 = delta * delta, d4 = d2 * d2;
  std::vector<Vec2> answer(targets.size());
#pragma omp parallel for schedule(dynamic, 16)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    const double lam = std::fabs(b[i]) / delta;
    const Coefficients cs = coefficients(order, lam);
    Sum2 sum;
    for (int k = 0; k < n; ++k) {
      const Vec2 r = nodes[k].x - targets[i];
      const double r2 = dot(r, r);
      const RadialFactors rf = radial_factors(r2 / d2, cs);
      const double gf = regularized_kernel(std::sqrt(r2), delta, cs);
      const double r_f = dot(r, f[k]);
      const Vec2 sl = (r * r_f * rf.a / (4 * PI * d2) - gf * f[k] / 2) / mu;
      const double rq = dot(r, q[k] - q0[i]);
      const double rn = dot(r, nodes[k].normal);
      const Vec2 dl = r * (rq * rn * (-rf.ap) / (PI * d4));
      sum.add((sl + dl) * weights[k]);
    }
    Vec2 value = sum.value() + side_indicator(b[i]) * q0[i];
    const GeomPoint& c = close[i];
    const double alpha = std::sqrt(1 + b[i] * c.kappa);
    const double scale = std::fabs(b[i]) * scaled_moment_n0(order, lam);
    const Vec2 b0 = c.normal * dot(c.normal, f0[i]) / alpha;
    const Vec2 sym = c.normal * dot(c.tau, q_s[i]) + c.tau * dot(c.normal, q_s[i]);
    Vec2 c2 = sym / (alpha * alpha * alpha);
    c2 -= (b[i] / (2 * alpha * alpha * alpha)) * c.normal * dot(c.normal, q_ss[i]);
    c2 += (b[i] * b[i] * c.kappa_s / (2 * std::pow(alpha, 5))) * c.normal * dot(c.normal, q_s[i]);
    answer[i] = value + scale * b0 / (4 * PI * mu) + scale * c2 / (2 * PI);
  }
  return answer;
}

std::vector<double> unregularized_laplace(const Geometry& geometry, const PeriodicSamples& f,
                                          const PeriodicSamples& g, const std::vector<Vec2>& targets,
                                          int n, double d_sign) {
  const auto nodes = sample(geometry, n);
  const auto fv = f.at_grid(n), gv = g.at_grid(n);
  std::vector<double> out(targets.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum sl, dl;
    for (int k = 0; k < n; ++k) {
      const Vec2 r = nodes[k].x - targets[i];
      const double r2 = dot(r, r);
      sl.add(std::log(std::sqrt(r2)) * (fv[k] * nodes[k].speed));
      dl.add(dot(r, nodes[k].normal) / r2 * (gv[k] * nodes[k].speed));
    }
    out[i] = sl.value() / n + d_sign * (dl.value() / n);
  }
  return out;
}

std::vector<Vec2> unregularized_stokes(const Geometry& geometry, const std::vector<Vec2>& q,
                                       const std::vector<Vec2>& f, const std::vector<Vec2>& targets,
                                       double mu) {
  const int n = static_cast<int>(q.size());
  const auto nodes = sample(geometry, n);
  std::vector<Vec2> out(targets.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum2 sum;
    for (int k = 0; k < n; ++k) {
      const Vec2 r = nodes[k].x - targets[i];
      const double r2 = dot(r, r);
      const double rf = dot(r, f[k]);
      const Vec2 sl = (-std::log(std::sqrt(r2)) * f[k] + r * (rf / r2)) / (4 * PI * mu);
      const double rq = dot(r, q[k]), rn = dot(r, nodes[k].normal);
      const Vec2 dl = r * (rq * rn / (PI * r2 * r2));
      sum.add((sl + dl) * (2 * PI * nodes[k].speed / n));
    }
    out[i] = sum.value();
  }
  return out;
}

}  // namespace regbie
