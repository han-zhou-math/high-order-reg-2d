// Generic numerical tools replacing SciPy: quadrature, bounded scalar
// minimization, FFT-friendly lengths, least-squares slopes, timing.
#pragma once
#include <chrono>
#include <functional>
#include <vector>

namespace regbie {

// Adaptive Gauss-Kronrod (7/15) quadrature on [a, b] (scipy.integrate.quad on a finite interval).
double integrate(const std::function<double(double)>& f, double a, double b,
                 double epsabs = 1.49e-8, double epsrel = 1.49e-8, int max_intervals = 2000);

// Integral over [0, inf) of an integrand bounded by C s^k e^{-s}: [0, 1] U [1, inf)
// with the substitution s = 1/x on the tail (as QUADPACK qagi).
double integrate_semi_infinite(const std::function<double(double)>& f,
                               double epsabs = 1.49e-8, double epsrel = 1.49e-8);

// scipy.optimize.minimize_scalar(method="bounded") (fminbound), same algorithm and defaults.
struct BoundedMinimum {
  double x, fun;
  int nfev;
};
BoundedMinimum minimize_scalar_bounded(const std::function<double(double)>& f, double lo, double hi,
                                       double xatol = 1e-5, int maxiter = 500);

// Smallest n >= target whose prime factors are at most 11 (scipy.fft.next_fast_len).
long next_fast_len(long target);

// Slope of the least-squares line through (x_i, y_i) (np.polyfit(x, y, 1)[0]).
double fitted_slope(const std::vector<double>& x, const std::vector<double>& y);

// Wall-clock seconds (time.perf_counter).
inline double perf_counter() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace regbie
