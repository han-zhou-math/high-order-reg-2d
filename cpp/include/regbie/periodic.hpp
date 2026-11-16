// Periodic samples: trigonometric interpolation, spectral derivatives and the
// pressure densities (bie_evaluators.PeriodicSamples, boundary_data.py).
#pragma once
#include <complex>
#include <mutex>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/types.hpp"

namespace regbie {

// Samples of a 2*pi-periodic function at t_k = 2*pi*k/n, with `dim` components.
// Evaluation at the sample nodes returns the samples; elsewhere it uses the
// trigonometric interpolant with the modes of np.fft.fftfreq(n, 1/n).
class PeriodicSamples {
 public:
  PeriodicSamples() = default;
  explicit PeriodicSamples(const std::vector<double>& values);            // dim = 1
  explicit PeriodicSamples(const std::vector<Vec2>& values);              // dim = 2
  PeriodicSamples(const std::vector<std::vector<double>>& columns);       // dim = columns.size()

  int n() const { return n_; }
  int dim() const { return static_cast<int>(columns_.size()); }
  const std::vector<double>& column(int c = 0) const { return columns_[c]; }

  // Component c at parameter t.
  double operator()(double t, int c = 0) const;
  // Both components (dim = 2) at parameter t.
  Vec2 vec(double t) const;
  // Values of component c at the parameters t (the samples when t is the node grid).
  std::vector<double> at(const std::vector<double>& t, int c = 0) const;
  std::vector<Vec2> vec_at(const std::vector<double>& t) const;
  // Values at the n-point grid, copying the samples when n equals the sample count.
  std::vector<double> at_grid(int n, int c = 0) const;

 private:
  void transform();
  int n_ = 0;
  std::vector<std::vector<double>> columns_;
  std::vector<std::vector<std::complex<double>>> coefficients_;  // fft / n
  std::vector<double> modes_;
};

// FFTW planning is not thread safe; every planner call in the library holds this lock.
std::mutex& fftw_planner_mutex();

// Fourier coefficients fft(values)/n, in the order of np.fft.fftfreq.
std::vector<std::complex<double>> fourier_coefficients(const std::vector<double>& values);

// d^order/dt^order of periodic samples by FFT (boundary_data.spectral_derivative).
std::vector<double> spectral_derivative(const std::vector<double>& values, int order = 1);
std::vector<Vec2> spectral_derivative(const std::vector<Vec2>& values, int order = 1);

// Pressure densities phi_p, psi_p and the arclength derivatives q_s, q_ss of the
// velocity data (boundary_data.pressure_densities). q, f are sampled at the n nodes.
struct PressureBoundaryData {
  std::vector<double> phi, psi;
  std::vector<Vec2> q_s, q_ss;
};
PressureBoundaryData pressure_densities(const Geometry& geometry, const std::vector<Vec2>& q,
                                        const std::vector<Vec2>& f, double mu = 1.0);

}  // namespace regbie
