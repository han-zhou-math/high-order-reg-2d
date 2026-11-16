// experiments/fixed_width_allgrid.py: all-grid runs with the fixed width
// delta = k h_c (k = 2, 3), plus a quadrature check (N versus 2N nodes) of the
// regularized sums at 97 evenly spaced halo targets and the error maxima.
//
// Run from the repository root:  cpp/build/fixed_width_allgrid [--output-root DIR]
// Output: <root>/fixed_width/allgrid/{kite_laplace,starfish_stokes}_k{2,3}_M{128,256,512,1024}.json
// (existing files are skipped).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "regbie/allgrid.hpp"
#include "regbie/fixed_width.hpp"
#include "regbie/io.hpp"
#include "regbie/output.hpp"
#include "regbie/problems.hpp"

using namespace regbie;

namespace {

// np.argmax
long argmax(const std::vector<double>& v) {
  long best = 0;
  for (long k = 0; k < static_cast<long>(v.size()); ++k) {
    if (std::isnan(v[k])) return k;
    if (v[k] > v[best]) best = k;
  }
  return best;
}

// np.max
double max_value(const std::vector<double>& v) {
  double r = v.at(0);
  for (double x : v) {
    if (std::isnan(x)) return x;
    r = std::max(r, x);
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);

  auto process = [&](const std::string& label, const auto& problem) {
    for (int k : {2, 3})
      for (int m : {128, 256, 512, 1024}) {
        const auto path = output_path(root, "results/fixed_width/allgrid/" + label + "_k" + std::to_string(k) + "_M" +
                                                std::to_string(m) + ".json");
        if (std::filesystem::exists(path)) continue;
        const double h = (problem.box[1] - problem.box[0]) / (m + 1);
        const int n = spacing(problem.geometry, h).n_gamma;
        const double delta = k * h;
        MatchedOptions options;
        options.kappa = k;
        options.delta_override = delta;
        options.n_override = n;
        MatchedRun run = matched_run(problem, m, options);
        const MatchedFields& fields = run.fields;
        const auto halo = mask_indices(fields.halo);
        const long count = static_cast<long>(halo.size());

        // np.unique(np.linspace(0, count - 1, 97).astype(int)), plus the halo index of each error maximum.
        std::vector<long> ii;
        const double step = static_cast<double>(count - 1) / 96;
        for (int i = 0; i < 97; ++i) ii.push_back(i == 96 ? count - 1 : static_cast<long>(i * step + 0.0));
        std::vector<double> magnitude(count);
        if (label == "kite_laplace") {
          for (long j = 0; j < count; ++j) magnitude[j] = std::fabs(fields.error(halo[j][0], halo[j][1]));
          ii.push_back(argmax(magnitude));
        } else {
          for (long j = 0; j < count; ++j) magnitude[j] = std::fabs(fields.pressure_error(halo[j][0], halo[j][1]));
          ii.push_back(argmax(magnitude));
          for (long j = 0; j < count; ++j) {
            const double ex = fields.velocity_error[0](halo[j][0], halo[j][1]);
            const double ey = fields.velocity_error[1](halo[j][0], halo[j][1]);
            magnitude[j] = std::sqrt(ex * ex + ey * ey);
          }
          ii.push_back(argmax(magnitude));
        }
        std::sort(ii.begin(), ii.end());
        ii.erase(std::unique(ii.begin(), ii.end()), ii.end());
        std::vector<Vec2> targets;
        for (long j : ii) targets.push_back({fields.xx(halo[j][0], halo[j][1]), fields.yy(halo[j][0], halo[j][1])});

        const ClosestPoints cp = closest_points(targets, problem.geometry, problem.inside);
        const FieldValues v = fixed_width_evaluate(label, targets, cp.t, cp.b, delta, n, "mixed");
        const FieldValues w = fixed_width_evaluate(label, targets, cp.t, cp.b, delta, 2 * n, "mixed");
        Json& report = run.report;
        report["quadrature_check_target_count"] = static_cast<int>(targets.size());
        report["quadrature_check_includes_direct_maxima"] = true;
        Json difference = Json::object();
        std::vector<double> d(targets.size());
        if (label == "kite_laplace") {
          for (size_t j = 0; j < d.size(); ++j) d[j] = std::fabs(v.laplace[j] - w.laplace[j]);
          difference["laplace"] = max_value(d);
        } else {
          for (size_t j = 0; j < d.size(); ++j) d[j] = norm(v.velocity[j] - w.velocity[j]);
          difference["velocity"] = max_value(d);
          for (size_t j = 0; j < d.size(); ++j) d[j] = std::fabs(v.pressure[j] - w.pressure[j]);
          difference["pressure"] = max_value(d);
        }
        report["quadrature_difference_inf"] = difference;
        write_json(path, report);
        std::printf("%s %d %d %s\n", label.c_str(), k, m, difference.dump(-1).c_str());
        std::fflush(stdout);
      }
  };
  process("kite_laplace", laplace_kite());
  process("starfish_stokes", stokes_starfish());
  return 0;
}
