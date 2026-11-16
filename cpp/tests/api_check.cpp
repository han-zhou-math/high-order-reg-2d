// Accuracy of the user interface (Curve, LayerPotentials) on the kite (Laplace)
// and starfish (Stokes) tests with exact solutions, for curves given by
// functions and by points, at targets on, near and far from the curve.
// Exits with status 1 if an error exceeds its tolerance.
#include <cmath>
#include <cstdio>
#include <vector>

#include "regbie/experiments/problems.hpp"
#include "regbie/regbie.hpp"

using namespace regbie;

namespace {

// Targets: a 41 x 41 grid of the box, and points at distances b from the curve
// along the normal at 64 parameters, b in {0, +-delta/4, +-delta, +-3 delta}.
std::vector<Vec2> targets_for(const Curve& curve, const Box& box, double delta) {
  std::vector<Vec2> y;
  for (int j = 0; j <= 40; ++j)
    for (int i = 0; i <= 40; ++i)
      y.push_back({box[0] + (box[1] - box[0]) * i / 40.0, box[2] + (box[3] - box[2]) * j / 40.0});
  for (int k = 0; k < 64; ++k) {
    const GeomPoint p = curve(2 * PI * (k + 0.3) / 64);
    for (double c : {0.0, 0.25, -0.25, 1.0, -1.0, 3.0, -3.0}) y.push_back(p.x + c * delta * p.normal);
  }
  return y;
}

bool report(const char* what, double error, double tolerance) {
  std::printf("%-44s %10.2e  (tolerance %.0e)\n", what, error, tolerance);
  return error <= tolerance;
}

}  // namespace

int main() {
  bool ok = true;
  const int n = 4096;  // delta = 3 spacing: errors about 1e-8 (Python fixed-width runs)

  // Laplace, kite: S f + D g = u- inside, u+ outside, their average on the curve.
  const LaplaceProblem& lp = laplace_kite();
  for (int from_points = 0; from_points < 2; ++from_points) {
    std::vector<Vec2> pts(n);
    for (int k = 0; k < n; ++k) pts[k] = lp.geometry(2 * PI * k / n).x;
    const Curve curve = from_points ? Curve::from_points(pts) : Curve::from_function(lp.geometry);
    const LayerPotentials layers(curve, n);
    std::vector<double> f, g;
    lp.boundary_densities(layers.parameters(), f, g);
    const auto y = targets_for(curve, lp.box, layers.delta());
    const auto s = layers.single_layer(f, y);
    const auto d = layers.double_layer(g, y);
    const auto near = layers.near_targets(y);
    std::vector<bool> is_near(y.size(), false);
    std::vector<double> b(y.size(), 1.0);
    for (size_t j = 0; j < near.index.size(); ++j) {
      is_near[near.index[j]] = true;
      b[near.index[j]] = near.b[j];
    }
    double near_error = 0, far_error = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      const Side side = !is_near[i] ? Side::Piecewise : (b[i] < 0 ? Side::Inside : (b[i] > 0 ? Side::Outside : Side::Average));
      const double e = std::fabs(s[i] + d[i] - lp.exact(y[i], side));
      (is_near[i] ? near_error : far_error) = std::max(is_near[i] ? near_error : far_error, e);
    }
    std::printf("Laplace kite, curve from %s, n = %d, delta = %.3e, %zu near targets\n",
                from_points ? "points" : "function", n, layers.delta(), near.index.size());
    ok &= report("  max error, near targets", near_error, 5e-8);
    ok &= report("  max error, far targets", far_error, 5e-8);
  }

  // Stokes, starfish: S f + D q = u inside, 0 outside; pressure likewise.
  const StokesProblem& sp = stokes_starfish();
  for (int from_points = 0; from_points < 2; ++from_points) {
    std::vector<Vec2> pts(n);
    for (int k = 0; k < n; ++k) pts[k] = sp.geometry(2 * PI * k / n).x;
    const Curve curve = from_points ? Curve::from_points(pts) : Curve::from_function(sp.geometry);
    PotentialOptions options;
    options.mu = sp.mu;
    const LayerPotentials layers(curve, n, options);
    std::vector<Vec2> q, f;
    sp.boundary_data(layers.parameters(), sp.mu, q, f);
    const auto y = targets_for(curve, sp.box, layers.delta());
    const auto s = layers.stokeslet(f, y);
    const auto d = layers.stresslet(q, y);
    const auto p = layers.stokes_pressure(f, q, y);
    const auto near = layers.near_targets(y);
    std::vector<double> b(y.size(), 1.0);
    std::vector<bool> is_near(y.size(), false);
    for (size_t j = 0; j < near.index.size(); ++j) {
      is_near[near.index[j]] = true;
      b[near.index[j]] = near.b[j];
    }
    double u_near = 0, u_far = 0, p_near = 0, p_far = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      const Side side = !is_near[i] ? Side::Piecewise : (b[i] < 0 ? Side::Inside : (b[i] > 0 ? Side::Outside : Side::Average));
      const StokesSolution exact = sp.exact(y[i], side, sp.mu);
      const double eu = norm(s[i] + d[i] - exact.u), ep = std::fabs(p[i] - exact.p);
      if (is_near[i]) {
        u_near = std::max(u_near, eu);
        p_near = std::max(p_near, ep);
      } else {
        u_far = std::max(u_far, eu);
        p_far = std::max(p_far, ep);
      }
    }
    std::printf("Stokes starfish, curve from %s, n = %d, delta = %.3e, %zu near targets\n",
                from_points ? "points" : "function", n, layers.delta(), near.index.size());
    ok &= report("  max velocity error, near targets", u_near, 5e-8);
    ok &= report("  max velocity error, far targets", u_far, 5e-8);
    ok &= report("  max pressure error, near targets", p_near, 5e-8);
    ok &= report("  max pressure error, far targets", p_far, 5e-8);
  }
  std::printf("%s\n", ok ? "all errors within tolerance" : "SOME ERRORS EXCEED TOLERANCE");
  return ok ? 0 : 1;
}
