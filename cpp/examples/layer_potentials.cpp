// Using the regbie interface: Laplace and Stokes layer potentials on an ellipse
// given by boundary points, evaluated at targets on, near and far from it.
//
// Laplace: u = Re(z^3) is harmonic, so S(-du/dn) + D(u) = u inside, 0 outside.
// Stokes: the rigid rotation u = (-y, x), p = 0 has zero traction on any curve,
// so D q with q = u equals u inside and 0 outside (S f vanishes with f = 0).
//
// Build:  cmake -S cpp -B cpp/build && cmake --build cpp/build --target layer_potentials
// Run:    cpp/build/layer_potentials
#include <cmath>
#include <cstdio>
#include <vector>

#include "regbie/regbie.hpp"

using regbie::Vec2;

int main() {
  // 512 points on the ellipse x = (1.4 cos t, 0.8 sin t), counterclockwise.
  const int n = 512;
  std::vector<Vec2> points(n);
  for (int k = 0; k < n; ++k) {
    const double t = 2 * regbie::PI * k / n;
    points[k] = {1.4 * std::cos(t), 0.8 * std::sin(t)};
  }
  const regbie::Curve curve = regbie::Curve::from_points(points);
  const regbie::LayerPotentials layers(curve, n);  // order 7, delta = 3 * node spacing

  // Densities at the nodes: f = -du/dn and g = u for u = Re(z^3) = x^3 - 3 x y^2.
  std::vector<double> f(n), g(n);
  for (int k = 0; k < n; ++k) {
    const regbie::GeomPoint& p = layers.nodes()[k];
    const double x = p.x.x, y = p.x.y;
    const Vec2 grad{3 * x * x - 3 * y * y, -6 * x * y};
    f[k] = -regbie::dot(grad, p.normal);
    g[k] = x * x * x - 3 * x * y * y;
  }

  // Targets on the normal through x(1): inside, near, on, and outside the curve.
  const regbie::GeomPoint p = curve(1.0);
  std::vector<Vec2> targets;
  for (double b : {-0.5, -0.02, -1e-3, 0.0, 1e-3, 0.02, 0.5}) targets.push_back(p.x + b * p.normal);

  const auto s = layers.single_layer(f, targets);
  const auto d = layers.double_layer(g, targets);
  std::printf("delta = %.3e\n", layers.delta());
  std::printf("%9s %16s %16s\n", "b", "S f + D g", "exact");
  for (size_t i = 0; i < targets.size(); ++i) {
    const double b = regbie::dot(targets[i] - p.x, p.normal);
    const double x = targets[i].x, y = targets[i].y, u = x * x * x - 3 * x * y * y;
    // On the curve the double layer gives the average of the inside value u and the outside value 0.
    const double exact = b < 0 ? u : (b > 0 ? 0.0 : 0.5 * u);
    std::printf("%9.3g %16.12f %16.12f\n", b, s[i] + d[i], exact);
  }

  // Stokes: rigid rotation, q = (-y, x) on the curve, zero traction.
  std::vector<Vec2> q(n);
  for (int k = 0; k < n; ++k) q[k] = {-layers.nodes()[k].x.y, layers.nodes()[k].x.x};
  const auto u = layers.stresslet(q, targets);
  const auto pressure = layers.stokes_pressure({}, q, targets);
  std::printf("\n%9s %30s %12s\n", "b", "D q (exact (-y, x) inside)", "pressure");
  for (size_t i = 0; i < targets.size(); ++i)
    std::printf("%9.3g %14.10f %14.10f %12.2e\n", regbie::dot(targets[i] - p.x, p.normal), u[i].x, u[i].y,
                pressure[i]);
  return 0;
}
