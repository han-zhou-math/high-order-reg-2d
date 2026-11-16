// All-grid part of experiments/run_swapped_production.py (run_allgrid): Cartesian
// reconstructions of the Laplace-kite and Stokes-starfish problems on the matched
// boundary grid, with delta = h_gamma^(5/7) and h_gamma = 2 pi / N.
// The --local part of that script is a separate driver.
//
// Run from the repository root:
//   cpp/build/run_swapped_allgrid [--allgrid] [--levels 64 128 256 512 768 1024] [--output-root DIR]
// Output: <root>/swapped_low_order/allgrid/{laplace,stokes}/M<m>.json (existing files are skipped).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "regbie/allgrid.hpp"
#include "regbie/io.hpp"
#include "regbie/output.hpp"
#include "regbie/problems.hpp"

using namespace regbie;

namespace {

// Integers following `flag` up to the next option (argparse nargs="+").
std::vector<int> int_list(int argc, char** argv, const char* flag, std::vector<int> fallback) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) {
      std::vector<int> values;
      for (int k = i + 1; k < argc && std::strncmp(argv[k], "--", 2) != 0; ++k) values.push_back(std::atoi(argv[k]));
      return values;
    }
  return fallback;
}

// run_swapped_production._grid_parameters
struct GridParameters {
  Spacing match;
  double h_gamma, delta;
};
GridParameters grid_parameters(const Geometry& geometry, const Box& box, int m) {
  const double h_c = (box[1] - box[0]) / (m + 1);
  GridParameters p;
  p.match = spacing(geometry, h_c);
  p.h_gamma = 2 * PI / p.match.n_gamma;
  p.delta = std::pow(p.h_gamma, 5.0 / 7);
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);
  const auto levels = int_list(argc, argv, "--levels", {64, 128, 256, 512, 768, 1024});

  auto run_allgrid = [&](const std::string& label, const auto& problem) {
    const auto output = output_path(root, "results/swapped_low_order/allgrid/" + label);
    std::filesystem::create_directories(output);
    for (int m : levels) {
      const auto destination = output / ("M" + std::to_string(m) + ".json");
      if (std::filesystem::exists(destination)) {
        std::printf("allgrid %s M=%d already complete\n", label.c_str(), m);
        std::fflush(stdout);
        continue;
      }
      const GridParameters g = grid_parameters(problem.geometry, problem.box, m);
      MatchedOptions options;
      options.mode = "bie";
      options.rule = "matched";
      options.delta_override = g.delta;
      options.n_override = g.match.n_gamma;
      MatchedRun run = matched_run(problem, m, options);
      Json& report = run.report;
      report["h_gamma"] = g.h_gamma;
      if (label == "laplace") {
        report["maximum_node"] = maximum_node(run.fields, run.fields.error);
      } else {
        report["pressure_maximum_node"] = maximum_node(run.fields, run.fields.pressure_error);
        report["velocity_maximum_node"] = maximum_node(run.fields, run.fields.velocity_error);
      }
      write_json(destination, report);
      Json summary = Json::object();
      summary["problem"] = label;
      summary["m"] = m;
      summary["n_gamma"] = g.match.n_gamma;
      summary["delta"] = g.delta;
      summary["seconds"] = report.at("total_seconds");
      if (label == "laplace") {
        summary["errors"] = report.at("laplace");
      } else {
        Json errors = Json::object();
        errors["pressure"] = report.at("pressure");
        errors["velocity"] = report.at("velocity");
        errors["gradient"] = report.at("gradient");
        summary["errors"] = errors;
      }
      std::printf("%s\n", summary.dump(-1).c_str());
      std::fflush(stdout);
    }
  };
  run_allgrid("laplace", laplace_kite());
  run_allgrid("stokes", stokes_starfish());
  return 0;
}
