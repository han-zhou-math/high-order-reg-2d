// results/paper_revision/geometric_diagnostic.py: closest-point geometry of the
// direct halo of the Stokes-starfish all-grid runs (alpha^2 = 1 + b kappa), the
// location of the velocity error maximum, and a fresh direct evaluation at M = 128
// with the leading near-curve correction terms.
//
// The Python script loads the saved .npz fields of the separated run
// (run_swapped_allgrid) and of the 2 h_c run (fixed_width_allgrid, else
// run_missing_practical). The C++ drivers write no .npz files, so the same runs are
// recomputed here; a 2 h_c field counts as saved when the JSON report of either
// driver exists below the output root. The swapped JSON reports must exist.
//
// Run from the repository root:  cpp/build/geometric_diagnostic [--output-root DIR]
// Output: <root>/paper_revision/geometric_diagnostic.{json,csv}
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "regbie/allgrid.hpp"
#include "regbie/io.hpp"
#include "regbie/kernels.hpp"
#include "regbie/layers.hpp"
#include "regbie/numerics.hpp"
#include "regbie/output.hpp"
#include "regbie/periodic.hpp"
#include "regbie/problems.hpp"

using namespace regbie;

namespace {

// [x, y] as a JSON list (ndarray.tolist()).
Json json_xy(Vec2 v) { return Json(std::vector<double>{v.x, v.y}); }

// np.argmin / np.argmax: first extremal index.
long argmin(const std::vector<double>& v) {
  long best = 0;
  for (long k = 1; k < static_cast<long>(v.size()); ++k)
    if (v[k] < v[best]) best = k;
  return best;
}
long argmax(const std::vector<double>& v) {
  long best = 0;
  for (long k = 1; k < static_cast<long>(v.size()); ++k)
    if (v[k] > v[best]) best = k;
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);
  const StokesProblem& problem = stokes_starfish();
  const Geometry& geometry = problem.geometry;
  const int levels[] = {64, 128, 256, 512, 768, 1024};
  const int dense = 65536;
  const auto t_dense = periodic_nodes(dense);
  const auto g_dense = sample(geometry, t_dense);
  double kappa_inf = std::fabs(g_dense[0].kappa);
  for (const auto& g : g_dense) kappa_inf = std::max(kappa_inf, std::fabs(g.kappa));

  Json rows = Json::array();
  std::vector<CsvRow> csv_rows;
  Json fresh = Json::object();
  for (int m : levels) {
    const std::string mm = std::to_string(m);
    const Json base_report = read_json(output_path(root, "results/swapped_low_order/allgrid/stokes/M" + mm + ".json"));
    const int n_gamma = static_cast<int>(base_report.at("n_gamma").number());
    const double h_c = base_report.at("h_c").number();
    const double delta_separated = std::pow(2 * PI / n_gamma, 5.0 / 7);

    // Fields of the separated run (run_swapped_production.run_allgrid).
    MatchedOptions separated_options;
    separated_options.delta_override = delta_separated;
    separated_options.n_override = n_gamma;
    const MatchedRun separated = matched_run(problem, m, separated_options);
    const MatchedFields& data = separated.fields;
    const auto halo = mask_indices(data.halo);
    std::vector<Vec2> targets;
    for (const auto& ji : halo) targets.push_back({data.xx(ji[0], ji[1]), data.yy(ji[0], ji[1])});
    const ClosestPoints cp = closest_points(targets, geometry, problem.inside);
    const std::vector<double>& theta = cp.t;
    const std::vector<double>& b = cp.b;
    std::vector<double> alpha2(targets.size()), abs_b(targets.size());
    for (size_t k = 0; k < targets.size(); ++k) {
      alpha2[k] = 1 + b[k] * geometry(theta[k]).kappa;
      abs_b[k] = std::fabs(b[k]);
    }
    const long ia = argmin(alpha2);

    for (const std::string rule : {"separated", "2hc"}) {
      // Saved full-grid fields of this width rule, if any.
      std::optional<MatchedRun> other;
      const MatchedFields* fields = nullptr;
      if (rule == "separated") {
        fields = &data;
      } else if (std::filesystem::exists(
                     output_path(root, "results/fixed_width/allgrid/starfish_stokes_k2_M" + mm + ".json")) ||
                 std::filesystem::exists(
                     output_path(root, "results/paper_revision/allgrid_practical/stokes/M" + mm + ".json"))) {
        const double h = (problem.box[1] - problem.box[0]) / (m + 1);
        MatchedOptions options;
        options.kappa = 2;
        options.delta_override = 2 * h;
        options.n_override = spacing(geometry, h).n_gamma;
        other = matched_run(problem, m, options);
        fields = &other->fields;
      }
      const double delta = rule == "separated" ? delta_separated : 2 * h_c;
      double max_abs_b = abs_b[0];
      for (double x : abs_b) max_abs_b = std::max(max_abs_b, x);

      Json row = Json::object();
      CsvRow csv;
      row["m_grid"] = m;
      row["width_rule"] = rule;
      row["h_c"] = h_c;
      row["n_gamma"] = n_gamma;
      row["delta"] = delta;
      row["curvature_inf"] = kappa_inf;
      row["delta_curvature_inf"] = delta * kappa_inf;
      row["min_alpha_squared"] = alpha2[ia];
      row["min_alpha_point"] = json_xy(targets[ia]);
      row["min_alpha_b"] = b[ia];
      row["direct_targets"] = static_cast<int>(targets.size());
      row["max_abs_b_over_delta"] = max_abs_b / delta;
      csv["m_grid"] = static_cast<std::int64_t>(m);
      csv["width_rule"] = rule;
      csv["h_c"] = h_c;
      csv["n_gamma"] = static_cast<std::int64_t>(n_gamma);
      csv["delta"] = delta;
      csv["delta_curvature_inf"] = delta * kappa_inf;
      csv["min_alpha_squared"] = alpha2[ia];

      if (fields) {
        // Largest velocity error on the full grid and the geometry at that node.
        const Grid magnitude =
            (fields->velocity_error[0] * fields->velocity_error[0] + fields->velocity_error[1] * fields->velocity_error[1])
                .sqrt();
        const long flat = argmax(std::vector<double>(magnitude.data(), magnitude.data() + magnitude.size()));
        const int j = static_cast<int>(flat / magnitude.cols()), i = static_cast<int>(flat % magnitude.cols());
        const Vec2 y{fields->xx(j, i), fields->yy(j, i)};
        const ClosestPoints cy = closest_points({y}, geometry, problem.inside);
        const double a2 = 1 + cy.b[0] * geometry(cy.t[0]).kappa;
        row["velocity_error_inf"] = magnitude(j, i);
        row["velocity_max_point"] = json_xy(y);
        row["velocity_max_index"] = std::vector<int>{j, i};
        row["velocity_max_is_direct"] = static_cast<bool>(fields->halo(j, i));
        row["velocity_max_alpha_squared"] = a2;
        row["velocity_max_b"] = cy.b[0];
        row["velocity_max_alpha_minus5"] = std::pow(a2, -2.5);
        csv["velocity_error_inf"] = magnitude(j, i);
        csv["velocity_max_alpha_squared"] = a2;
        csv["velocity_max_alpha_minus5"] = std::pow(a2, -2.5);
      }

      if (m == 128) {
        // Fresh direct evaluation on the halo and its largest error.
        const int n = n_gamma;
        std::vector<Vec2> q, f;
        problem.boundary_data(periodic_nodes(n), 1.0, q, f);
        const auto computed = eval_stokes_regularized(geometry, q, f, targets, theta, b, delta, 7);
        std::vector<Vec2> exact(targets.size());
        std::vector<double> errors(targets.size());
        for (size_t k = 0; k < targets.size(); ++k) {
          exact[k] = problem.exact(targets[k], Side::Piecewise, 1.0).u;
          errors[k] = norm(computed[k] - exact[k]);
        }
        const long ie = argmax(errors);
        const Vec2 y = targets[ie];
        // Independent closest point: nearest dense node, then bounded minimization of |x(t) - y|^2.
        std::vector<double> distance2(dense);
        for (int k = 0; k < dense; ++k) {
          const Vec2 d = g_dense[k].x - y;
          distance2[k] = d.x * d.x + d.y * d.y;
        }
        const long jd = argmin(distance2);
        const double period_step = 2 * PI / dense;
        const BoundedMinimum opt = minimize_scalar_bounded(
            [&](double t) {
              const Vec2 d = geometry(t).x - y;
              return d.x * d.x + d.y * d.y;
            },
            t_dense[jd] - period_step, t_dense[jd] + period_step, 1e-15);
        const GeomPoint go = geometry(opt.x);
        const double independent_b = dot(y - go.x, go.normal);

        Json fr = Json::object();
        fr["maximum_error"] = errors[ie];
        fr["maximum_point"] = json_xy(y);
        fr["computed_closest_t"] = theta[ie];
        fr["computed_b"] = b[ie];
        fr["computed_alpha2"] = alpha2[ie];
        fr["dense_refined_closest_t"] = opt.x;
        fr["dense_refined_b"] = independent_b;
        fr["dense_refined_alpha2"] = 1 + independent_b * go.kappa;
        fr["max_saved_fresh_error_difference"] = std::fabs(errors[ie] - row.at("velocity_error_inf").number());

        // Leading correction terms at the maximizing target, lambda = |b|/delta.
        const PressureBoundaryData pd = pressure_densities(geometry, q, f);
        const GeomPoint g0 = geometry(theta[ie]);
        const Vec2 normal = g0.normal, tau = g0.tau;
        const double ks = g0.kappa_s;
        const double aa = std::sqrt(alpha2[ie]);
        const Vec2 f0 = PeriodicSamples(f).vec(theta[ie]);
        const Vec2 qs = PeriodicSamples(pd.q_s).vec(theta[ie]);
        const Vec2 qss = PeriodicSamples(pd.q_ss).vec(theta[ie]);
        const double bv = b[ie];
        const double scale = std::fabs(bv) * scaled_moment_n0(7, std::fabs(bv) / delta);
        const Vec2 sym = normal * dot(tau, qs) + tau * dot(normal, qs);
        const Vec2 c1 = sym / std::pow(aa, 3);
        const Vec2 c2 = -bv / (2 * std::pow(aa, 3)) * normal * dot(normal, qss);
        const Vec2 c3 = bv * bv * ks / (2 * std::pow(aa, 5)) * normal * dot(normal, qs);
        const Vec2 slcorr = scale * normal * dot(normal, f0) / aa / (4 * PI);
        const Vec2 dlcorr = scale * (c1 + c2 + c3) / (2 * PI);
        std::vector<Vec2> q2, f2;
        problem.boundary_data(periodic_nodes(2 * n), 1.0, q2, f2);
        const auto w = eval_stokes_regularized(geometry, q2, f2, {y}, {theta[ie]}, {b[ie]}, delta, 7);
        fr["single_layer_correction"] = json_xy(slcorr);
        fr["double_layer_correction"] = json_xy(dlcorr);
        Json terms = Json::array();
        for (const Vec2& c : {c1, c2, c3}) terms.push_back(json_xy(scale * c / (2 * PI)));
        fr["stresslet_correction_terms"] = terms;
        fr["source_doubling_difference"] = norm(computed[ie] - w[0]);
        fr["raw_combined_error"] = norm(computed[ie] - slcorr - dlcorr - exact[ie]);
        fresh[rule] = fr;
      }
      rows.push_back(row);
      csv_rows.push_back(csv);
    }
  }

  Json out = Json::object();
  out["method"] = "Closest-point geometry on current direct halo; saved full-grid fields; fresh M=128 direct evaluation";
  out["curvature_dense_samples"] = dense;
  out["rows"] = rows;
  out["fresh_M128"] = fresh;
  write_json(output_path(root, "results/paper_revision/geometric_diagnostic.json"), out);
  write_csv(output_path(root, "results/paper_revision/geometric_diagnostic.csv"),
            {"m_grid", "width_rule", "h_c", "n_gamma", "delta", "delta_curvature_inf", "min_alpha_squared",
             "velocity_error_inf", "velocity_max_alpha_squared", "velocity_max_alpha_minus5"},
            csv_rows);

  Json shown = Json::object();
  shown["curvature_inf"] = kappa_inf;
  shown["fresh_M128"] = fresh;
  shown["rows"] = rows;
  std::printf("%s\n", shown.dump(2).c_str());
  return 0;
}
