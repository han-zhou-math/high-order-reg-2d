#include "regbie/special.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "regbie/types.hpp"

namespace regbie {

double exp1(double x) {
  if (!(x > 0.0)) throw std::domain_error("exp1 requires x > 0");
  if (x < 1.0) {
    // E1(x) = -gamma - log x + sum_{k>=1} (-1)^(k+1) x^k / (k k!).
    double sum = 0.0, term = 1.0;
    for (int k = 1; k < 60; ++k) {
      term *= -x / k;
      const double add = -term / k;
      sum += add;
      if (std::fabs(add) < 1e-18 * std::fabs(sum)) break;
    }
    return -EULER_GAMMA - std::log(x) + sum;
  }
  // Continued fraction (modified Lentz), convergent for x >= 1.
  const double tiny = 1e-300, eps = 1e-16;
  double b = x + 1.0, c = 1.0 / tiny, d = 1.0 / b, h = d;
  for (int i = 1; i < 10000; ++i) {
    const double an = -static_cast<double>(i) * i;
    b += 2.0;
    d = 1.0 / (an * d + b);
    c = b + an / c;
    const double del = c * d;
    h *= del;
    if (std::fabs(del - 1.0) < eps) break;
  }
  return h * std::exp(-x);
}

double erfcx(double x) {
  if (x < 0.0) throw std::domain_error("erfcx is only needed for x >= 0");
  if (x < 5.0) return std::exp(x * x) * std::erfc(x);
  // Continued fraction erfcx(x) = 1/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))).
  const double tiny = 1e-300;
  double f = x, c = x, d = 0.0;
  for (int k = 1; k < 500; ++k) {
    const double a = 0.5 * k;
    d = x + a * d;
    if (d == 0.0) d = tiny;
    c = x + a / c;
    if (c == 0.0) c = tiny;
    d = 1.0 / d;
    const double del = c * d;
    f *= del;
    if (std::fabs(del - 1.0) < 1e-16) break;
  }
  return 1.0 / (std::sqrt(PI) * f);
}

}  // namespace regbie
