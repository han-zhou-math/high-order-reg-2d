#include "regbie/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace regbie {

namespace {

// Gauss-Kronrod 7/15 nodes and weights (QUADPACK qk15).
const double XGK[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                       0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                       0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                       0.207784955007898467600689403773245, 0.000000000000000000000000000000000};
const double WGK[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                       0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                       0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                       0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
const double WG[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                      0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

struct Piece {
  double a, b, value, error;
  bool operator<(const Piece& o) const { return error < o.error; }
};

Piece gk15(const std::function<double(double)>& f, double a, double b) {
  const double center = 0.5 * (a + b), half = 0.5 * (b - a);
  const double fc = f(center);
  double resg = fc * WG[3], resk = fc * WGK[7];
  for (int j = 0; j < 3; ++j) {  // Gauss nodes are the odd Kronrod nodes
    const int jj = 2 * j + 1;
    const double dx = half * XGK[jj];
    const double f1 = f(center - dx), f2 = f(center + dx);
    resg += WG[j] * (f1 + f2);
    resk += WGK[jj] * (f1 + f2);
  }
  for (int j = 0; j < 4; ++j) {
    const int jj = 2 * j;
    const double dx = half * XGK[jj];
    resk += WGK[jj] * (f(center - dx) + f(center + dx));
  }
  return {a, b, resk * half, std::fabs((resk - resg) * half)};
}

}  // namespace

double integrate(const std::function<double(double)>& f, double a, double b, double epsabs,
                 double epsrel, int max_intervals) {
  std::priority_queue<Piece> pieces;
  Piece first = gk15(f, a, b);
  double total = first.value, error = first.error;
  pieces.push(first);
  int count = 1;
  while (error > std::max(epsabs, epsrel * std::fabs(total)) && count < max_intervals) {
    const Piece worst = pieces.top();
    pieces.pop();
    const double mid = 0.5 * (worst.a + worst.b);
    const Piece left = gk15(f, worst.a, mid), right = gk15(f, mid, worst.b);
    total += left.value + right.value - worst.value;
    error += left.error + right.error - worst.error;
    pieces.push(left);
    pieces.push(right);
    ++count;
    if (worst.error <= 4 * std::numeric_limits<double>::epsilon() * std::fabs(worst.value)) break;
  }
  // Re-sum to remove the drift of the running totals.
  total = 0.0;
  while (!pieces.empty()) {
    total += pieces.top().value;
    pieces.pop();
  }
  return total;
}

double integrate_semi_infinite(const std::function<double(double)>& f, double epsabs, double epsrel) {
  // Integrands here decay like e^{-s}; beyond s = 80 they are below 1e-30 of their size.
  return integrate(f, 0.0, 80.0, epsabs, epsrel, 4000);
}

BoundedMinimum minimize_scalar_bounded(const std::function<double(double)>& func, double x1, double x2,
                                       double xatol, int maxiter) {
  if (!(x1 <= x2)) throw std::invalid_argument("lower bound must not exceed upper bound");
  const double sqrt_eps = std::sqrt(2.2e-16);
  const double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
  double a = x1, b = x2;
  double fulc = a + golden_mean * (b - a);
  double nfc = fulc, xf = fulc;
  double rat = 0.0, e = 0.0;
  double x = xf;
  double fx = func(x);
  int num = 1;
  double fu = std::numeric_limits<double>::infinity();
  double ffulc = fx, fnfc = fx;
  double xm = 0.5 * (a + b);
  double tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
  double tol2 = 2.0 * tol1;
  auto sign = [](double v) { return (v > 0) - (v < 0); };
  while (std::fabs(xf - xm) > (tol2 - 0.5 * (b - a))) {
    bool golden = true;
    if (std::fabs(e) > tol1) {
      golden = false;
      double r = (xf - nfc) * (fx - ffulc);
      double q = (xf - fulc) * (fx - fnfc);
      double p = (xf - fulc) * q - (xf - nfc) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::fabs(q);
      r = e;
      e = rat;
      if ((std::fabs(p) < std::fabs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))) {
        rat = (p + 0.0) / q;
        x = xf + rat;
        if (((x - a) < tol2) || ((b - x) < tol2)) {
          const double si = sign(xm - xf) + ((xm - xf) == 0);
          rat = tol1 * si;
        }
      } else {
        golden = true;
      }
    }
    if (golden) {
      e = (xf >= xm) ? a - xf : b - xf;
      rat = golden_mean * e;
    }
    const double si = sign(rat) + (rat == 0);
    x = xf + si * std::max(std::fabs(rat), tol1);
    fu = func(x);
    ++num;
    if (fu <= fx) {
      if (x >= xf) a = xf; else b = xf;
      fulc = nfc; ffulc = fnfc;
      nfc = xf; fnfc = fx;
      xf = x; fx = fu;
    } else {
      if (x < xf) a = x; else b = x;
      if ((fu <= fnfc) || (nfc == xf)) {
        fulc = nfc; ffulc = fnfc;
        nfc = x; fnfc = fu;
      } else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
        fulc = x; ffulc = fu;
      }
    }
    xm = 0.5 * (a + b);
    tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
    tol2 = 2.0 * tol1;
    if (num >= maxiter) break;
  }
  return {xf, fx, num};
}

long next_fast_len(long target) {
  if (target <= 6) return std::max(target, 1L);
  for (long n = target;; ++n) {
    long r = n;
    for (long p : {2L, 3L, 5L, 7L, 11L})
      while (r % p == 0) r /= p;
    if (r == 1) return n;
  }
}

double fitted_slope(const std::vector<double>& x, const std::vector<double>& y) {
  if (x.size() != y.size() || x.size() < 2) throw std::invalid_argument("need two or more points");
  const double n = static_cast<double>(x.size());
  double mx = 0, my = 0;
  for (size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  double sxy = 0, sxx = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
  }
  return sxy / sxx;
}

}  // namespace regbie
