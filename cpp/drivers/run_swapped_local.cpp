// Near-curve runs of the swapped low-order tests: Laplace on the scaled kite and
// Stokes on the scaled starfish (experiments/run_swapped_production.py --local, run_local).
//
// For N in 256..16384 and p = 3, 5, 7: delta = h^q, 97 targets at b = lambda delta
// (lambda in {-1/2, 0, 1/2}, and -2..2 for Laplace at p = 7). Errors against the exact
// one-sided fields and the change under 2N nodes at fixed delta and targets.
//
// Run from the repository root:  cpp/build/run_swapped_local [--output-root DIR]
// Writes <root>/swapped_low_order/local/errors.csv.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "regbie/io.hpp"
#include "regbie/layers.hpp"
#include "regbie/experiments/local_experiments.hpp"
#include "regbie/experiments/output.hpp"
#include "regbie/periodic.hpp"
#include "regbie/experiments/problems.hpp"

using namespace regbie;

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);
  const std::vector<int> levels{256, 512, 1024, 2048, 4096, 8192, 16384};
  const std::vector<int> orders{3, 5, 7};
  const LaplaceProblem& kite = laplace_kite();
  const StokesProblem& starfish = stokes_starfish();
  std::vector<CsvRow> rows;
  for (const std::string label : {"laplace_kite", "stokes_starfish"}) {
    const bool laplace = label == "laplace_kite";
    const Geometry& geometry = laplace ? kite.geometry : starfish.geometry;
    for (int n : levels) {
      const auto t = periodic_nodes(n), t2 = periodic_nodes(2 * n);
      std::vector<double> f, g, f2, g2;  // Laplace densities
      std::vector<Vec2> q, fs, q2, fs2;  // Stokes velocity and traction
      if (laplace) {
        kite.boundary_densities(t, f, g);
        kite.boundary_densities(t2, f2, g2);
      } else {
        starfish.boundary_data(t, 1.0, q, fs);
        starfish.boundary_data(t2, 1.0, q2, fs2);
      }
      for (int order : orders) {
        const double h_gamma = 2 * PI / n;
        const double delta = std::pow(h_gamma, order_to_q(order));
        std::vector<double> lambdas{-0.5, 0.0, 0.5};
        if (laplace && order == 7) lambdas = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
        for (const TargetSet& s : local_targets(geometry, delta, lambdas)) {
          const std::string side = side_name(s.lam);
          const Side which = side_from_name(side);
          const size_t m = s.targets.size();
          // (field, max error, max quadrature difference)
          std::vector<std::pair<std::string, std::pair<double, double>>> errors;
          if (laplace) {
            const auto value = eval_laplace_regularized(geometry, PeriodicSamples(f), PeriodicSamples(g), s.targets,
                                                        s.theta, s.b, delta, n, order);
            const auto check = eval_laplace_regularized(geometry, PeriodicSamples(f2), PeriodicSamples(g2), s.targets,
                                                        s.theta, s.b, delta, 2 * n, order);
            double e = 0, d = 0;
            for (size_t i = 0; i < m; ++i) {
              e = std::max(e, std::fabs(value[i] - kite.exact(s.targets[i], which)));
              d = std::max(d, std::fabs(check[i] - value[i]));
            }
            errors.push_back({"laplace", {e, d}});
          } else {
            const auto un = eval_stokes_regularized(geometry, q, fs, s.targets, s.theta, s.b, delta, order);
            const auto pn = eval_pressure_regularized(geometry, q, fs, s.targets, s.theta, s.b, delta, order);
            const auto un2 = eval_stokes_regularized(geometry, q2, fs2, s.targets, s.theta, s.b, delta, order);
            const auto pn2 = eval_pressure_regularized(geometry, q2, fs2, s.targets, s.theta, s.b, delta, order);
            double eu = 0, du = 0, ep = 0, dp = 0;
            for (size_t i = 0; i < m; ++i) {
              const StokesSolution exact = starfish.exact(s.targets[i], which, 1.0);
              eu = std::max(eu, norm(un[i] - exact.u));
              du = std::max(du, norm(un2[i] - un[i]));
              ep = std::max(ep, std::fabs(pn[i] - exact.p));
              dp = std::max(dp, std::fabs(pn2[i] - pn[i]));
            }
            errors.push_back({"velocity", {eu, du}});
            errors.push_back({"pressure", {ep, dp}});
          }
          for (const auto& [field, err] : errors)
            rows.push_back({{"problem", label},
                            {"field", field},
                            {"n_gamma", std::int64_t(n)},
                            {"h_gamma", h_gamma},
                            {"order", std::int64_t(order)},
                            {"m", std::int64_t(order_to_m(order))},
                            {"q", order_to_q(order)},
                            {"delta", delta},
                            {"b_over_delta", s.lam},
                            {"side", side},
                            {"error_inf", err.first},
                            {"quadrature_difference_inf", err.second}});
        }
      }
      std::printf("local %s N=%d\n", label.c_str(), n);
      std::fflush(stdout);
    }
  }
  write_csv(output_path(root, "results/swapped_low_order/local/errors.csv"),
            {"problem", "field", "n_gamma", "h_gamma", "order", "m", "q", "delta", "b_over_delta", "side",
             "error_inf", "quadrature_difference_inf"},
            rows);
  return 0;
}
