// Regularized single-layer potential for the unit circle (examples/circle_single_layer.py).
//
// Density f(t) = 1 + 0.7 cos 3t; targets at signed distance b = lambda*delta from the
// circle. With G(r) = log(r)/(2 pi) the exact potential at rho (cos theta, sin theta) is
//   S f = log(max(1, rho)) - (0.7/6) min(rho, 1/rho)^3 cos 3 theta.
// delta = h^q with h = 2 pi / N, so the error decreases like h^2, h^4, h^5 for p = 3, 5, 7.
//
// Run from the repository root:  cpp/build/circle_single_layer
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/layers.hpp"

using namespace regbie;

int main() {
  const int orders[3] = {3, 5, 7};
  const double qs[3] = {2.0 / 3, 4.0 / 5, 5.0 / 7};
  const int levels[5] = {128, 256, 512, 1024, 2048};
  const double lambdas[3] = {-0.5, 0.0, 0.5};
  const int count = 97;
  std::vector<double> theta(count);
  for (int i = 0; i < count; ++i) theta[i] = 2 * PI * (i + 0.37) / count;

  std::printf("max error over 97 targets and lambda = b/delta in {-0.5, 0, 0.5}\n");
  std::printf("%2s %6s %10s %10s %10s\n", "p", "N", "delta", "error", "rate in h");
  for (int o = 0; o < 3; ++o) {
    double previous = -1;
    for (int n : levels) {
      const double h = 2 * PI / n, delta = std::pow(h, qs[o]);
      std::vector<double> density(n);
      const auto t = periodic_nodes(n);
      for (int k = 0; k < n; ++k) density[k] = 1 + 0.7 * std::cos(3 * t[k]);
      double error = 0;
      for (double lam : lambdas) {
        std::vector<double> b(count, lam * delta);
        std::vector<Vec2> targets(count);
        for (int i = 0; i < count; ++i) targets[i] = (1 + b[i]) * Vec2{std::cos(theta[i]), std::sin(theta[i])};
        const auto value = single_layer(regbie_circle, density, targets, b, delta, orders[o]);
        for (int i = 0; i < count; ++i) {
          const double rho = norm(targets[i]), th = std::atan2(targets[i].y, targets[i].x);
          const double exact = std::log(std::max(1.0, rho)) - 0.7 / 6 * std::pow(std::min(rho, 1 / rho), 3) * std::cos(3 * th);
          error = std::max(error, std::fabs(value[i] - exact));
        }
      }
      char rate[32] = "";
      if (previous > 0) std::snprintf(rate, sizeof(rate), "%10.2f", std::log2(previous / error));
      std::printf("%2d %6d %10.3e %10.2e %s\n", orders[o], n, delta, error, rate);
      previous = error;
    }
  }
  return 0;
}
