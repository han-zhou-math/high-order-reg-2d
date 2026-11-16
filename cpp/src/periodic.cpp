#include "regbie/periodic.hpp"

#include <fftw3.h>

#include <cmath>
#include <mutex>
#include <stdexcept>

namespace regbie {

std::mutex& fftw_planner_mutex() {
  static std::mutex m;
  return m;
}

namespace {

std::mutex& planner_mutex() { return fftw_planner_mutex(); }

// Full complex DFT of real data (numpy.fft.fft), forward or inverse (unnormalized).
std::vector<std::complex<double>> dft(const std::vector<std::complex<double>>& in, int sign) {
  const int n = static_cast<int>(in.size());
  std::vector<std::complex<double>> out(n);
  std::vector<std::complex<double>> work(in);
  fftw_plan plan;
  {
    std::lock_guard<std::mutex> lock(planner_mutex());
    plan = fftw_plan_dft_1d(n, reinterpret_cast<fftw_complex*>(work.data()),
                            reinterpret_cast<fftw_complex*>(out.data()), sign, FFTW_ESTIMATE);
  }
  fftw_execute(plan);
  {
    std::lock_guard<std::mutex> lock(planner_mutex());
    fftw_destroy_plan(plan);
  }
  return out;
}

std::vector<double> fftfreq_modes(int n) {
  std::vector<double> modes(n);
  for (int k = 0; k < n; ++k) modes[k] = (k < (n + 1) / 2) ? k : k - n;
  return modes;
}

}  // namespace

PeriodicSamples::PeriodicSamples(const std::vector<double>& values) : n_(static_cast<int>(values.size())) {
  columns_ = {values};
  transform();
}

PeriodicSamples::PeriodicSamples(const std::vector<Vec2>& values) : n_(static_cast<int>(values.size())) {
  columns_.assign(2, std::vector<double>(n_));
  for (int k = 0; k < n_; ++k) {
    columns_[0][k] = values[k].x;
    columns_[1][k] = values[k].y;
  }
  transform();
}

PeriodicSamples::PeriodicSamples(const std::vector<std::vector<double>>& columns)
    : n_(columns.empty() ? 0 : static_cast<int>(columns[0].size())), columns_(columns) {
  transform();
}

void PeriodicSamples::transform() {
  modes_ = fftfreq_modes(n_);
  coefficients_.clear();
  for (const auto& column : columns_) {
    if (static_cast<int>(column.size()) != n_) throw std::invalid_argument("ragged periodic samples");
    std::vector<std::complex<double>> data(column.begin(), column.end());
    auto c = dft(data, FFTW_FORWARD);
    for (auto& v : c) v /= static_cast<double>(n_);
    coefficients_.push_back(std::move(c));
  }
}

double PeriodicSamples::operator()(double t, int c) const {
  const auto& coef = coefficients_[c];
  double sum = 0.0;
  for (int k = 0; k < n_; ++k) {
    const double phase = t * modes_[k];
    sum += coef[k].real() * std::cos(phase) - coef[k].imag() * std::sin(phase);
  }
  return sum;
}

Vec2 PeriodicSamples::vec(double t) const { return {(*this)(t, 0), (*this)(t, 1)}; }

std::vector<double> PeriodicSamples::at(const std::vector<double>& t, int c) const {
  if (static_cast<int>(t.size()) == n_) {
    bool grid = true;
    for (int k = 0; k < n_ && grid; ++k) grid = std::fabs(t[k] - 2.0 * PI * k / n_) <= 2e-14;
    if (grid) return columns_[c];
  }
  std::vector<double> out(t.size());
#pragma omp parallel for schedule(static)
  for (long i = 0; i < static_cast<long>(t.size()); ++i) out[i] = (*this)(t[i], c);
  return out;
}

std::vector<Vec2> PeriodicSamples::vec_at(const std::vector<double>& t) const {
  const auto x = at(t, 0), y = at(t, 1);
  std::vector<Vec2> out(t.size());
  for (size_t i = 0; i < t.size(); ++i) out[i] = {x[i], y[i]};
  return out;
}

std::vector<double> PeriodicSamples::at_grid(int n, int c) const {
  if (n == n_) return columns_[c];
  return at(periodic_nodes(n), c);
}

std::vector<double> spectral_derivative(const std::vector<double>& values, int order) {
  const int n = static_cast<int>(values.size());
  const auto modes = fftfreq_modes(n);
  std::vector<std::complex<double>> data(values.begin(), values.end());
  auto c = dft(data, FFTW_FORWARD);
  for (int k = 0; k < n; ++k) {
    std::complex<double> factor(1.0, 0.0);  // (i k)^order by repeated products, exact at k = 0
    for (int j = 0; j < order; ++j) factor *= std::complex<double>(0.0, modes[k]);
    c[k] *= factor;
  }
  const auto back = dft(c, FFTW_BACKWARD);
  std::vector<double> out(n);
  for (int k = 0; k < n; ++k) out[k] = back[k].real() / n;
  return out;
}

std::vector<Vec2> spectral_derivative(const std::vector<Vec2>& values, int order) {
  std::vector<double> x(values.size()), y(values.size());
  for (size_t k = 0; k < values.size(); ++k) { x[k] = values[k].x; y[k] = values[k].y; }
  const auto dx = spectral_derivative(x, order), dy = spectral_derivative(y, order);
  std::vector<Vec2> out(values.size());
  for (size_t k = 0; k < values.size(); ++k) out[k] = {dx[k], dy[k]};
  return out;
}

PressureBoundaryData pressure_densities(const Geometry& geometry, const std::vector<Vec2>& q,
                                        const std::vector<Vec2>& f, double mu) {
  const int n = static_cast<int>(q.size());
  const auto g = sample(geometry, n);
  PressureBoundaryData out;
  out.q_s = spectral_derivative(q);
  for (int k = 0; k < n; ++k) out.q_s[k] = out.q_s[k] / g[k].speed;
  out.q_ss = spectral_derivative(out.q_s);
  for (int k = 0; k < n; ++k) out.q_ss[k] = out.q_ss[k] / g[k].speed;
  std::vector<double> f_tau(n), qs_normal(n);
  for (int k = 0; k < n; ++k) {
    f_tau[k] = dot(f[k], g[k].tau);
    qs_normal[k] = dot(out.q_s[k], g[k].normal);
  }
  const auto f_tau_s = spectral_derivative(f_tau), qs_normal_s = spectral_derivative(qs_normal);
  out.phi.resize(n);
  out.psi.resize(n);
  for (int k = 0; k < n; ++k) {
    out.phi[k] = f_tau_s[k] / g[k].speed - 2 * mu * (qs_normal_s[k] / g[k].speed);
    out.psi[k] = dot(f[k], g[k].normal) + 2 * mu * dot(out.q_s[k], g[k].tau);
  }
  return out;
}

}  // namespace regbie
