#include "regbie/kernels.hpp"

#include <cmath>
#include <stdexcept>

#include "regbie/numerics.hpp"
#include "regbie/special.hpp"
#include "regbie/types.hpp"

namespace regbie {

Coefficients coefficients(int order, double lam) {
  if (order != 1 && order != 3 && order != 5 && order != 7)
    throw std::invalid_argument("order must be one of (1, 3, 5, 7)");
  if (!std::isfinite(lam) || lam < 0) throw std::invalid_argument("lam must be finite and nonnegative");
  Coefficients cs;
  cs.order = order;
  cs.m = (order - 1) / 2;
  if (order == 1) return cs;
  // Recurrence near the curve; the positive integral avoids cancellation for large lambda.
  const double z = lam * lam;
  double h0 = 2 * (1 - std::sqrt(PI) * lam * erfcx(lam));
  const double h1 = (1 - z * h0) / 1.5;
  const double h2 = (1 - z * h1) / 2.5;
  double d1 = h0 - h1, d2 = h0 - 2 * h1 + h2;
  if (lam > 1) {
    double* dest[3] = {&h0, &d1, &d2};
    for (int j = 0; j < 3; ++j) {
      // Positive integral, scaled to stay O(1) during quadrature.
      auto integrand = [j, z](double s) {
        return std::exp(-s) * std::pow(s, j) * std::pow(1 + s / z, -j - 1.5);
      };
      const double val = integrate_semi_infinite(integrand, 2e-14, 2e-14);
      *dest[j] = val / std::pow(z, j + 1);
    }
  }
  if (order == 3) {
    cs.c[0] = h0;
  } else if (order == 5) {
    const double c2 = d1;
    cs.c[0] = h0 + (z + 1.5) * c2;
    cs.c[1] = c2;
  } else {
    const double c3 = d2 / 2;
    const double c2 = d1 + (2 * z + 5) * c3;
    cs.c[0] = h0 + (z + 1.5) * c2 - (z * z + 3 * z + 3.75) * c3;
    cs.c[1] = c2;
    cs.c[2] = c3;
  }
  return cs;
}

double regularized_kernel(double r, double delta, const Coefficients& cs) {
  if (!std::isfinite(delta) || delta <= 0) throw std::invalid_argument("delta must be positive and finite");
  if (!std::isfinite(r) || r < 0) throw std::invalid_argument("r must be finite and nonnegative");
  const double q = r / delta;
  const double u = q * q;
  double base;
  if (u < 0.5) {
    // log(u)+E1(u) = -gamma + sum_{k>=1} (-1)^(k+1) u^k/(k k!); never inf-inf at r = 0.
    double series = 0.0, term = 1.0;
    for (int k = 1; k < 19; ++k) {
      term *= -u / k;
      series -= term / k;
    }
    base = 2 * std::log(delta) - EULER_GAMMA + series;
  } else {
    base = 2 * std::log(r) + exp1(u);
  }
  double correction = 0.0;
  if (cs.order >= 3) correction = correction - cs.c[0];
  if (cs.order >= 5) correction = correction + cs.c[1] * (1 + u);
  if (cs.order >= 7) correction = correction - cs.c[2] * (2 + 2 * u + u * u);
  return (base + std::exp(-u) * correction) / (4 * PI);
}

double smoothing_over_r2(double r2, double delta, const Coefficients& cs) {
  if (cs.order != 3 && cs.order != 5 && cs.order != 7) throw std::invalid_argument("order must be 3, 5, or 7");
  const double u = r2 / (delta * delta);
  double pm1_over_u = 0.0;
  for (int k = 1; k <= cs.m; ++k) pm1_over_u += ((k % 2) ? -1.0 : 1.0) * cs.c[k - 1] * std::pow(u, k - 1);
  const double base = (u != 0) ? -std::expm1(-u) / u : 1.0;
  return (base - std::exp(-u) * pm1_over_u) / (delta * delta);
}

RadialFactors radial_factors(double u, const Coefficients& cs) {
  if (cs.order != 3 && cs.order != 5 && cs.order != 7) throw std::invalid_argument("order must be 3, 5, or 7");
  // P = 1 + u V, T = 1 - exp(-u) P.
  double v = 0.0, vp = 0.0;
  for (int k = 1; k <= cs.m; ++k) {
    const double sgn = (k % 2) ? -1.0 : 1.0;
    v += sgn * cs.c[k - 1] * std::pow(u, k - 1);
    if (k > 1) vp += sgn * cs.c[k - 1] * (k - 1) * std::pow(u, k - 2);
  }
  const double base = (u != 0) ? -std::expm1(-u) / u : 1.0;
  double bp;
  if (u < 0.5) {
    // Derivative of sum_j (-u)^j/(j+1)!; Horner form as in the Python code.
    double acc = 0.0;
    for (int j = 20; j >= 1; --j) {
      double fact = 1.0;  // (j+1)!, exact in double for j <= 20
      for (int i = 2; i <= j + 1; ++i) fact *= i;
      acc = acc * u + ((j % 2) ? -1.0 : 1.0) * j / fact;
    }
    bp = acc;
  } else {
    bp = (u * std::exp(-u) + std::expm1(-u)) / (u * u);
  }
  const double e = std::exp(-u);
  RadialFactors out;
  out.a = base - e * v;
  out.ap = bp + e * (v - vp);
  out.tp = e * (1 + u * v - v - u * vp);
  return out;
}

double scaled_moment_n0(int order, double lam) {
  if (!std::isfinite(lam) || lam < 0) throw std::invalid_argument("lam must be finite and nonnegative");
  const Coefficients cs = coefficients(order, lam);
  double ans = PI * std::erfc(lam);
  for (int k = 1; k <= cs.m; ++k) {
    double moment = 0.0;
    for (int j = 0; j < k; ++j) {
      double comb = 1.0;  // binomial(k-1, j)
      for (int i = 1; i <= j; ++i) comb = comb * (k - 1 - j + i) / i;
      moment += comb * std::pow(lam, 2 * (k - 1 - j)) * std::tgamma(j + 0.5);
    }
    ans += lam * ((k % 2) ? -1.0 : 1.0) * cs.c[k - 1] * std::exp(-lam * lam) * moment;
  }
  return ans;
}

}  // namespace regbie
