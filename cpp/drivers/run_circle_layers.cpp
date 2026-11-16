// Circle single- and double-layer convergence at near-curve targets
// (experiments/run_circle_layers.py -> local_drivers.run_circle_layers).
//
// For N in 128..16384 and p = 3, 5, 7: delta = kappa h^q, 97 targets at b = lambda delta,
// lambda in {-1/2, 0, 1/2}; errors against the exact circle layers and the change
// under 2N nodes at fixed delta and targets.
//
// Run from the repository root:  cpp/build/run_circle_layers --run [--output-root DIR]
// Writes <root>/redesigned/circle_layers/{errors.csv, metadata.json}.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <cstdio>
#include <cstring>
#include <vector>

#include "regbie/io.hpp"
#include "regbie/layers.hpp"
#include "regbie/local_experiments.hpp"
#include "regbie/output.hpp"

using namespace regbie;

int main(int argc, char** argv) {
  bool run = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--run") == 0) run = true;
  if (!run) {
    std::fprintf(stderr, "run_circle_layers: error: pass --run to execute\n");
    return 2;
  }
  const auto output = output_path(output_root(argc, argv), "results/redesigned/circle_layers");

  validate_all();
  const std::vector<double> lambdas{-0.5, 0.0, 0.5};
  std::vector<CsvRow> rows;
  for (int n : boundary_levels()) {
    const double h = 2 * PI / n;
    const int n_hi = OVERSAMPLE_FACTOR * n;
    std::vector<double> density(n), density_hi(n_hi);
    const auto t = periodic_nodes(n), t_hi = periodic_nodes(n_hi);
    for (int k = 0; k < n; ++k) density[k] = circle_density(t[k]);
    for (int k = 0; k < n_hi; ++k) density_hi[k] = circle_density(t_hi[k]);
    for (int order : local_orders()) {
      const double q = order_to_q(order), delta = LOCAL_KAPPA * std::pow(h, q);
      for (const TargetSet& s : local_targets(circle_geometry, delta, lambdas)) {
        std::vector<double> g0(s.theta.size());
        for (size_t i = 0; i < s.theta.size(); ++i) g0[i] = circle_density(s.theta[i]);
        const auto sn = single_layer(circle_geometry, density, s.targets, s.b, delta, order);
        const auto dn = double_layer(circle_geometry, density, g0, s.targets, s.b, delta, order);
        const auto sn_hi = single_layer(circle_geometry, density_hi, s.targets, s.b, delta, order);
        const auto dn_hi = double_layer(circle_geometry, density_hi, g0, s.targets, s.b, delta, order);
        const std::string side = side_name(s.lam);
        double se = 0, de = 0, sc = 0, dc = 0;
        for (size_t i = 0; i < s.targets.size(); ++i) {
          const double radius = norm(s.targets[i]), angle = std::atan2(s.targets[i].y, s.targets[i].x);
          se = std::max(se, std::fabs(sn[i] - exact_circle_single(radius, angle)));
          de = std::max(de, std::fabs(dn[i] - exact_circle_double(radius, angle, side)));
          sc = std::max(sc, std::fabs(sn_hi[i] - sn[i]));
          dc = std::max(dc, std::fabs(dn_hi[i] - dn[i]));
        }
        const std::pair<const char*, std::pair<double, double>> layers[2] = {{"circle_single", {se, sc}},
                                                                              {"circle_double", {de, dc}}};
        for (const auto& layer : layers)
          rows.push_back({{"geometry", std::string("unit_circle")},
                          {"solution", std::string(layer.first)},
                          {"n_gamma", std::int64_t(n)},
                          {"h_gamma", h},
                          {"order", std::int64_t(order)},
                          {"m", std::int64_t(order_to_m(order))},
                          {"q", q},
                          {"delta", delta},
                          {"kappa", LOCAL_KAPPA},
                          {"lambda", s.lam},
                          {"side", side},
                          {"error_inf", layer.second.first},
                          {"error_l2", std::string("")},
                          {"oversampled_error_inf", layer.second.second}});
      }
    }
  }
  // result_io.LOCAL_FIELDS
  write_csv(output / "errors.csv",
            {"geometry", "solution", "n_gamma", "h_gamma", "order", "m", "q", "delta", "kappa", "lambda", "side",
             "error_inf", "error_l2", "oversampled_error_inf"},
            rows);

  // result_io.write_metadata (json.dumps(indent=2, sort_keys=True)).
  Json configuration = Json::object();
  configuration["levels"] = Json(boundary_levels());
  configuration["orders"] = Json(local_orders());
  Json q_by_order = Json::object();
  for (int order : local_orders()) q_by_order[std::to_string(order)] = order_to_q(order);
  configuration["q_by_order"] = q_by_order;
  configuration["kappa"] = LOCAL_KAPPA;
  configuration["target_count"] = LOCAL_TARGET_COUNT;
  configuration["signed_lambdas"] = Json(lambdas);
  configuration["oversample_factor"] = OVERSAMPLE_FACTOR;
  Json payload = Json::object();
  payload["experiment"] = "circle_layers";
  payload["configuration"] = configuration;
  payload["timestamp_utc"] = utc_timestamp();
  payload["code_commit"] = nullptr;
  payload["plot_error_floor"] = nullptr;
  write_json(output / "metadata.json", payload, true);
  return 0;
}
