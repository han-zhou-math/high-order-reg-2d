// experiments/matched_allgrid.main: matched-grid reconstructions of the
// five-lobed starfish (Laplace) or the nonconvex kite (Stokes), with direct values
// from the regularized sums (bie) or the exact solution (exact_direct, exact_gradient).
//
// Run from the repository root:
//   cpp/build/matched_allgrid --problem {laplace,stokes} [--mode exact_direct] [--rule matched]
//       [--kappa 1] [--levels 64 128 256 512 768 1024] [--run] [--output-root DIR]
// Output: <root>/matched_allgrid/<problem>_<mode>_<rule>_k<kappa:g>/M<m>.json (existing files are skipped).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

#include "regbie/allgrid.hpp"
#include "regbie/io.hpp"
#include "regbie/output.hpp"

using namespace regbie;

namespace {

const char* option(int argc, char** argv, const char* flag, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
  return fallback;
}

std::vector<int> int_list(int argc, char** argv, const char* flag, std::vector<int> fallback) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], flag) == 0) {
      std::vector<int> values;
      for (int k = i + 1; k < argc && std::strncmp(argv[k], "--", 2) != 0; ++k) values.push_back(std::atoi(argv[k]));
      return values;
    }
  return fallback;
}

bool one_of(const std::string& value, std::initializer_list<const char*> choices) {
  for (const char* c : choices)
    if (value == c) return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);
  const std::string mode = option(argc, argv, "--mode", "exact_direct");
  const std::string problem = option(argc, argv, "--problem", "");
  const std::string rule = option(argc, argv, "--rule", "matched");
  const double kappa = std::atof(option(argc, argv, "--kappa", "1"));
  const auto levels = int_list(argc, argv, "--levels", {64, 128, 256, 512, 768, 1024});
  if (!one_of(problem, {"laplace", "stokes"}) || !one_of(mode, {"bie", "exact_direct", "exact_gradient"}) ||
      !one_of(rule, {"matched", "twice", "old8m"})) {
    std::fprintf(stderr,
                 "usage: matched_allgrid --problem {laplace,stokes} [--mode {bie,exact_direct,exact_gradient}]"
                 " [--rule {matched,twice,old8m}] [--kappa K] [--levels M ...] [--output-root DIR]\n");
    return 2;
  }
  char kappa_text[32];
  std::snprintf(kappa_text, sizeof(kappa_text), "%g", kappa);  // f"{kappa:g}"
  const auto out = output_path(root, "results/matched_allgrid/" + problem + "_" + mode + "_" + rule + "_k" + kappa_text);
  std::filesystem::create_directories(out);

  MatchedOptions options;
  options.mode = mode;
  options.rule = rule;
  options.kappa = kappa;
  for (int m : levels) {
    const auto dest = out / ("M" + std::to_string(m) + ".json");
    if (std::filesystem::exists(dest)) continue;
    const MatchedRun run = problem == "laplace" ? matched_run(laplace_starfish_problem(), m, options)
                                                : matched_run(stokes_kite_problem(), m, options);
    write_json(dest, run.report);
    std::printf("%s\n", run.report.dump(-1).c_str());
    std::fflush(stdout);
  }
  return 0;
}
