// results/paper_revision/run_missing_practical.py: practical-width all-grid runs
// (delta = 2 h_c on the matched boundary grid) at the levels M = 64 and 768.
//
// Run from the repository root:  cpp/build/run_missing_practical [--output-root DIR]
// Output: <root>/paper_revision/allgrid_practical/{laplace,stokes}/M{64,768}.json
#include <cstdio>
#include <string>
#include <vector>

#include "regbie/allgrid.hpp"
#include "regbie/io.hpp"
#include "regbie/output.hpp"
#include "regbie/problems.hpp"

using namespace regbie;

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);

  auto practical = [&](const std::string& label, const auto& problem) {
    for (int m : {64, 768}) {
      const auto dest = output_path(root, "results/paper_revision/allgrid_practical/" + label);
      const double h = (problem.box[1] - problem.box[0]) / (m + 1);
      MatchedOptions options;
      options.kappa = 2;
      options.delta_override = 2 * h;
      options.n_override = spacing(problem.geometry, h).n_gamma;
      MatchedRun run = matched_run(problem, m, options);
      Json& report = run.report;
      if (label == "laplace") {
        report["error_maximum_node"] = maximum_node(run.fields, run.fields.error);
      } else {
        report["pressure_error_maximum_node"] = maximum_node(run.fields, run.fields.pressure_error);
        report["velocity_error_maximum_node"] = maximum_node(run.fields, run.fields.velocity_error);
      }
      write_json(dest / ("M" + std::to_string(m) + ".json"), report);
      Json shown = Json::object();
      if (label == "laplace") {
        shown["laplace"] = report.at("laplace");
      } else {
        shown["pressure"] = report.at("pressure");
        shown["velocity"] = report.at("velocity");
      }
      std::printf("%s %d %s\n", label.c_str(), m, shown.dump(-1).c_str());
      std::fflush(stdout);
    }
  };
  practical("laplace", laplace_kite());
  practical("stokes", stokes_starfish());
  return 0;
}
