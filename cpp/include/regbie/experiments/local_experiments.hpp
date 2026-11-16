// Shared pieces of the near-curve (local) experiments: configuration constants
// (experiments/config.py), the target sets of local_drivers._targets, the exact
// circle layers of manufactured_laplace.py and the checks of validation.validate_all.
#pragma once
#include <string>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/experiments/problems.hpp"
#include "regbie/types.hpp"

namespace regbie {

// config.py
inline const std::vector<int>& local_orders() {
  static const std::vector<int> v{3, 5, 7};
  return v;
}
inline const std::vector<int>& boundary_levels() {
  static const std::vector<int> v{128, 256, 512, 1024, 2048, 4096, 8192, 16384};
  return v;
}
inline int order_to_m(int order) { return (order - 1) / 2; }
// ORDER_TO_Q = {3: 2/3, 5: 4/5, 7: 5/7}
inline double order_to_q(int order) { return order == 3 ? 2.0 / 3 : (order == 5 ? 4.0 / 5 : 5.0 / 7); }
constexpr double LOCAL_KAPPA = 1.0;
constexpr int LOCAL_TARGET_COUNT = 97;
constexpr int OVERSAMPLE_FACTOR = 2;

// One target set of local_drivers._targets: theta_i = 2 pi (i + 0.37)/count,
// targets x(theta_i) + b n(theta_i) with b = lam * delta.
struct TargetSet {
  std::vector<double> theta;
  std::vector<Vec2> targets;
  std::vector<double> b;
  double lam = 0;
};
std::vector<TargetSet> local_targets(const Geometry& geometry, double delta, const std::vector<double>& signed_lambdas,
                                     int count = LOCAL_TARGET_COUNT);

// "average" if lam == 0, "inside" if lam < 0, else "outside".
std::string side_name(double lam);
Side side_from_name(const std::string& side);

// Circle benchmark of manufactured_laplace.py: f(t) = 1 + 0.4 cos 3t + 0.3 sin 5t.
double circle_density(double t);
double exact_circle_single(double r, double theta);
// side "average" gives 1/2; otherwise interior (r < 1), exterior (r > 1) or 1/2.
double exact_circle_double(double r, double theta, const std::string& side = "");

// validation.validate_all: the checks that can fail (finite manufactured boundary
// data, Laplace poles on the correct sides of the starfish, Goursat poles outside
// the kite). Throws std::runtime_error on failure.
void validate_all();

}  // namespace regbie
