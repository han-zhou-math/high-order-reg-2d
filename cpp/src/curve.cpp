#include "regbie/curve.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>

#include "regbie/periodic.hpp"

namespace regbie {

namespace {

double wrap(double t) {
  double s = std::fmod(t, 2 * PI);
  return s < 0 ? s + 2 * PI : s;
}

// Trigonometric interpolant of N real samples, with its derivatives at any t.
// Modes follow np.fft.fftfreq; for even N the Nyquist term is c cos(N t/2),
// whose derivatives the real part of sum c_k (ik)^j e^{ikt} reproduces.
class FourierSeries {
 public:
  explicit FourierSeries(const std::vector<double>& samples)
      : n_(static_cast<int>(samples.size())), values_(samples), coefficients_(fourier_coefficients(samples)) {
    for (int j = 1; j <= 3; ++j) node_derivatives_[j - 1] = spectral_derivative(samples, j);
    modes_.resize(n_);
    for (int k = 0; k < n_; ++k) modes_[k] = (k < (n_ + 1) / 2) ? k : k - n_;
  }

  // d^order x / dt^order at t; exact samples and FFT derivatives at the nodes.
  double operator()(double t, int order) const {
    const int node = node_index(t);
    if (node >= 0) return order == 0 ? values_[node] : node_derivatives_[order - 1][node];
    double sum = 0.0;
    for (int k = 0; k < n_; ++k) {
      std::complex<double> factor(1.0, 0.0);
      for (int j = 0; j < order; ++j) factor *= std::complex<double>(0.0, modes_[k]);
      sum += (coefficients_[k] * factor * std::polar(1.0, modes_[k] * t)).real();
    }
    return sum;
  }

  int node_index(double t) const {
    const double s = wrap(t) * n_ / (2 * PI);
    const double k = std::round(s);
    if (std::fabs(s - k) > 1e-11) return -1;
    return static_cast<int>(k) % n_;
  }

 private:
  int n_ = 0;
  std::vector<double> values_;
  std::array<std::vector<double>, 3> node_derivatives_;
  std::vector<std::complex<double>> coefficients_;
  std::vector<double> modes_;
};

}  // namespace

Curve Curve::from_function(Geometry geometry) {
  Curve c;
  c.geometry_ = std::move(geometry);
  const Geometry g = c.geometry_;
  // x'' by a centered difference of x' = |x'| tau; accurate to about 1e-11, ample for Newton.
  c.second_ = [g](double t) {
    const double eps = 2e-6;
    const GeomPoint p = g(t + eps), m = g(t - eps);
    return (p.speed * p.tau - m.speed * m.tau) / (2 * eps);
  };
  c.build_samples();
  return c;
}

Curve Curve::from_derivatives(std::function<std::array<Vec2, 4>(double)> derivatives) {
  Curve c;
  c.geometry_ = [derivatives](double t) {
    const auto d = derivatives(t);
    return frame(d[0], d[1], d[2], d[3]);
  };
  c.second_ = [derivatives](double t) { return derivatives(t)[2]; };
  c.build_samples();
  return c;
}

Curve Curve::from_points(const std::vector<Vec2>& points) {
  const int n = static_cast<int>(points.size());
  if (n < 8) throw std::invalid_argument("from_points needs at least 8 points");
  double area = 0.0;
  for (int k = 0; k < n; ++k) area += cross(points[k], points[(k + 1) % n]);
  if (!(area > 0)) throw std::invalid_argument("points must go around the curve counterclockwise");
  std::vector<double> xs(n), ys(n);
  for (int k = 0; k < n; ++k) {
    xs[k] = points[k].x;
    ys[k] = points[k].y;
  }
  auto fx = std::make_shared<const FourierSeries>(xs);
  auto fy = std::make_shared<const FourierSeries>(ys);
  Curve c;
  c.geometry_ = [fx, fy](double t) {
    Vec2 d[4];
    for (int j = 0; j < 4; ++j) d[j] = {(*fx)(t, j), (*fy)(t, j)};
    return frame(d[0], d[1], d[2], d[3]);
  };
  c.second_ = [fx, fy](double t) { return Vec2{(*fx)(t, 2), (*fy)(t, 2)}; };
  c.build_samples();
  return c;
}

void Curve::build_samples() {
  // Dense samples for the inside test, the near/far split and Newton seeds.
  const int samples = 4096;
  inside_ = std::make_shared<const CurveInside>(geometry_, samples);
  sample_t_ = periodic_nodes(samples);
  sample_x_.resize(samples);
  for (int k = 0; k < samples; ++k) sample_x_[k] = geometry_(sample_t_[k]).x;
  double area = 0.0;
  for (int k = 0; k < samples; ++k) area += cross(sample_x_[k], sample_x_[(k + 1) % samples]);
  if (!(area > 0)) throw std::invalid_argument("the curve must be counterclockwise");
}

bool Curve::inside(Vec2 p) const { return (*inside_)(p); }

std::vector<double> Curve::sample_distance(const std::vector<Vec2>& points) const {
  std::vector<double> d(points.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(points.size()); ++i) {
    double best = std::numeric_limits<double>::infinity();
    for (const Vec2& x : sample_x_) {
      const Vec2 r = x - points[i];
      best = std::min(best, dot(r, r));
    }
    d[i] = std::sqrt(best);
  }
  return d;
}

Curve::Projection Curve::project(const std::vector<Vec2>& points) const {
  Projection out;
  out.t.resize(points.size());
  out.b.resize(points.size());
#pragma omp parallel for schedule(dynamic, 16)
  for (long i = 0; i < static_cast<long>(points.size()); ++i) {
    const Vec2 y = points[i];
    double best = std::numeric_limits<double>::infinity();
    double t = 0.0;
    for (size_t k = 0; k < sample_x_.size(); ++k) {
      const Vec2 r = sample_x_[k] - y;
      if (dot(r, r) < best) {
        best = dot(r, r);
        t = sample_t_[k];
      }
    }
    // Newton's method for r(t).x'(t) = 0, r = x(t) - y.
    for (int iteration = 0; iteration < 30; ++iteration) {
      const GeomPoint g = geometry_(t);
      const Vec2 dx = g.speed * g.tau;
      const Vec2 r = g.x - y;
      const double step = dot(r, dx) / (dot(dx, dx) + dot(r, second_(t)));
      t = wrap(t - step);
      if (std::fabs(step) < 1e-14) break;
    }
    const GeomPoint g = geometry_(t);
    double b = dot(y - g.x, g.normal);
    if (std::fabs(b) <= 5e-13 * std::max(1.0, norm(y))) b = 0.0;  // on the curve
    out.t[i] = t;
    out.b[i] = b;
  }
  return out;
}

}  // namespace regbie
