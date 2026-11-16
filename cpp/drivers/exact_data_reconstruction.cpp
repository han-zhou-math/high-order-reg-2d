// results/paper_revision/exact_data_reconstruction.py: isolate the Cartesian
// consistency error with smooth exact data. With A + iB = exp(k(z - 0.895)), k = 8,
// the Laplace field is v = A and the Stokes field has stream function y A / k,
// u = (A/k - yB, -yA), p = -2A, inside the interface and zero outside. All direct
// and outer values are exact; the masks and the numerical pressure gradient are the
// original ones (exact_gradient replaces the latter by the exact gradient).
//
// Run from the repository root:  cpp/build/exact_data_reconstruction [--output-root DIR]
// Output: <root>/paper_revision/exact_data_reconstruction.json
#include <cmath>
#include <cstdio>
#include <string>

#include "regbie/experiments/allgrid.hpp"
#include "regbie/io.hpp"
#include "regbie/experiments/output.hpp"
#include "regbie/experiments/problems.hpp"

using namespace regbie;

namespace {

const double K = 8.0;

// (A, B) = exp(K (x - .895)) (cos K y, sin K y)
Vec2 harmonic(Vec2 p) {
  const double ex = std::exp(K * (p.x - .895));
  return {ex * std::cos(K * p.y), ex * std::sin(K * p.y)};
}

}  // namespace

int main(int argc, char** argv) {
  const auto root = output_root(argc, argv);

  LaplaceProblem laplace = laplace_kite();
  laplace.exact = [inside = laplace.inside](Vec2 p, Side) { return inside(p) ? harmonic(p).x : 0.0; };

  StokesProblem stokes = stokes_starfish();
  const InsideFn inside = stokes.inside;
  stokes.exact = [inside](Vec2 p, Side, double) {
    const Vec2 ab = harmonic(p);
    if (!inside(p)) return StokesSolution{{0.0, 0.0}, 0.0};
    return StokesSolution{{ab.x / K - p.y * ab.y, -p.y * ab.x}, -2 * ab.x};
  };
  stokes.exact_gradient = [inside](Vec2 p) {
    const Vec2 ab = harmonic(p);
    return inside(p) ? Vec2{-2 * K * ab.x, 2 * K * ab.y} : Vec2{0.0, 0.0};
  };
  stokes.mu = 1.0;

  Json rows = Json::array();
  auto add = [&](const std::string& label, const auto& problem, const std::string& mode, int m) {
    MatchedOptions options;
    options.mode = mode;
    Json report = matched_run(problem, m, options).report;
    report["diagnostic_label"] = label;
    report["manufactured_scale"] = K;
    rows.push_back(report);
    Json shown = Json::object();
    if (label == "laplace") {
      shown["laplace"] = report.at("laplace");
    } else {
      shown["pressure"] = report.at("pressure");
      shown["velocity"] = report.at("velocity");
    }
    std::printf("%s %d %s\n", label.c_str(), m, shown.dump(-1).c_str());
    std::fflush(stdout);
  };
  for (int m : {64, 128, 256, 512, 768, 1024}) {
    add("laplace", laplace, "exact_direct", m);
    add("stokes", stokes, "exact_direct", m);
    add("stokes_exact_gradient", stokes, "exact_gradient", m);
  }

  Json out = Json::object();
  out["k"] = K;
  out["fields"] =
      "A=exp(k(x-.895))cos(ky), B=exp(k(x-.895))sin(ky); Laplace v=A; Stokes psi=y A/k, p=-2A; outside fields zero";
  out["source"] =
      "All prescribed direct and outer values exact; original stencil masks and numerical pressure gradient retained";
  out["rows"] = rows;
  write_json(output_path(root, "results/paper_revision/exact_data_reconstruction.json"), out);
  return 0;
}
