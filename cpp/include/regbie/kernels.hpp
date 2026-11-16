// Regularized kernels: coefficients a_k(lambda), G_delta, T/r^2, radial factors, N0.
// C++ counterparts of regbie/regularized_slp.py, regularized_dlp.py,
// surface_operators.py and verify_stokes_derivation.scaled_moment_n0.
#pragma once
#include <array>

namespace regbie {

// The coefficients c_1..c_m of the order-p kernel at lambda = |b|/delta
// (m = 0, 1, 2, 3 for p = 1, 3, 5, 7). Computed once per target.
struct Coefficients {
  int order = 7;
  int m = 0;
  std::array<double, 3> c{0.0, 0.0, 0.0};
};

Coefficients coefficients(int order, double lam);

// G_delta(r) including r = 0, with the 1/(4 pi) factor: regularized_kernel(r, delta, order, b).
double regularized_kernel(double r, double delta, const Coefficients& cs);

// T(r^2/delta^2)/r^2, finite at r = 0: smoothing_over_r2(r2, delta, order, b).
double smoothing_over_r2(double r2, double delta, const Coefficients& cs);

// A = T/u, A' and T' at fixed lambda, finite at u = 0: radial_factors(u, order, lam).
struct RadialFactors {
  double a, ap, tp;
};
RadialFactors radial_factors(double u, const Coefficients& cs);

// lambda * N0(lambda), with the finite endpoint pi at lambda = 0.
double scaled_moment_n0(int order, double lam);

}  // namespace regbie
