// Fixed-width diagnostics: delta = kappa h (circle) or kappa h_phys (kite, starfish)
// with p = 7, so the regularization error is fixed while N grows
// (experiments/fixed_width.py, stages circle, complete, sweep and phase).
//
// Errors maximize over the 97 closest-point parameters theta_i = (j_i + phase) h,
// j_i = floor(N (i + 0.37)/97), for each phase and b = lambda delta separately; the
// quadrature difference compares N and 2N nodes at the same delta and targets.
//
// Run from the repository root:
//   cpp/build/fixed_width --stage circle|complete|sweep|phase [--output-root DIR]
// Writes <root>/fixed_width/circle/errors.csv (circle) and
// <root>/fixed_width/{kite_laplace,starfish_stokes}/errors.csv (complete),
// sweep.csv (sweep) or phase_quarter.csv (phase).
// The stage `moments` (90-digit mpmath moment sums) is not ported.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "regbie/fixed_width.hpp"
#include "regbie/io.hpp"
#include "regbie/local_experiments.hpp"
#include "regbie/output.hpp"
#include "regbie/problems.hpp"

using namespace regbie;

namespace {

// A Python number that keeps its type in the CSV (1 -> "1", 1.5 -> "1.5").
struct Num {
  double v;
  bool integer;
  CsvCell cell() const { return integer ? CsvCell(std::int64_t(v)) : CsvCell(v); }
  std::string text() const { return integer ? std::to_string(std::int64_t(v)) : py_repr(v); }
};
Num Int(int v) { return {double(v), true}; }
Num Flt(double v) { return {v, false}; }

// Pointwise error and quadrature difference of one field.
struct FieldErrors {
  std::string field;
  std::vector<double> error, difference;
};

std::vector<FieldErrors> compare(const std::string& label, const std::vector<Vec2>& targets,
                                 const std::vector<double>& theta, const std::vector<double>& b, double delta, int n,
                                 double lam, const std::string& side) {
  const size_t m = targets.size();
  std::vector<FieldErrors> out;
  if (label == "circle") {
    const auto val = fixed_width_evaluate_circle(targets, theta, b, delta, n);
    const auto check = fixed_width_evaluate_circle(targets, theta, b, delta, 2 * n);
    FieldErrors s{"single", std::vector<double>(m), std::vector<double>(m)};
    FieldErrors d{"double", std::vector<double>(m), std::vector<double>(m)};
    for (size_t i = 0; i < m; ++i) {
      s.error[i] = std::fabs(val.single[i] - exact_circle_single(1 + lam * delta, theta[i]));
      s.difference[i] = std::fabs(val.single[i] - check.single[i]);
      d.error[i] = std::fabs(val.double_layer[i] - exact_circle_double(1 + lam * delta, theta[i], side));
      d.difference[i] = std::fabs(val.double_layer[i] - check.double_layer[i]);
    }
    out = {s, d};
  } else if (label == "kite_laplace") {
    const auto val = fixed_width_evaluate(label, targets, theta, b, delta, n, side);
    const auto check = fixed_width_evaluate(label, targets, theta, b, delta, 2 * n, side);
    const Side which = side_from_name(side);
    FieldErrors l{"laplace", std::vector<double>(m), std::vector<double>(m)};
    for (size_t i = 0; i < m; ++i) {
      l.error[i] = std::fabs(val.laplace[i] - laplace_kite().exact(targets[i], which));
      l.difference[i] = std::fabs(val.laplace[i] - check.laplace[i]);
    }
    out = {l};
  } else {
    const auto val = fixed_width_evaluate(label, targets, theta, b, delta, n, side);
    const auto check = fixed_width_evaluate(label, targets, theta, b, delta, 2 * n, side);
    const Side which = side_from_name(side);
    FieldErrors u{"velocity", std::vector<double>(m), std::vector<double>(m)};
    FieldErrors p{"pressure", std::vector<double>(m), std::vector<double>(m)};
    for (size_t i = 0; i < m; ++i) {
      const StokesSolution exact = stokes_starfish().exact(targets[i], which, 1.0);
      u.error[i] = norm(val.velocity[i] - exact.u);
      u.difference[i] = norm(val.velocity[i] - check.velocity[i]);
      p.error[i] = std::fabs(val.pressure[i] - exact.p);
      p.difference[i] = std::fabs(val.pressure[i] - check.pressure[i]);
    }
    out = {u, p};
  }
  return out;
}

void local(const std::filesystem::path& root, const std::string& stage) {
  std::vector<std::string> labels{"circle"};
  if (stage != "circle") labels = {"kite_laplace", "starfish_stokes"};
  std::vector<int> ns;
  for (int j = 0; j < (stage == "circle" ? 8 : 5); ++j) ns.push_back((stage == "circle" ? 128 : 1024) << j);
  std::vector<Num> ks = stage == "circle" ? std::vector<Num>{Int(1), Flt(1.5), Int(2), Int(3), Int(4), Int(6), Int(8)}
                                          : std::vector<Num>{Int(2), Int(3), Int(4), Int(6)};
  if (stage == "sweep" || stage == "phase") {
    ns = {4096};
    ks = {Flt(0.75), Int(1), Flt(1.5), Int(2), Flt(2.5), Int(3), Int(4), Int(5), Int(6), Int(8)};
  }
  // Quarter-step is a supplementary check motivated by circle DLP aliasing.
  if (stage == "phase") ks = {Int(1), Flt(1.5), Int(2), Int(3), Int(4)};
  const std::vector<std::string> fields{"geometry",     "field",           "n_gamma",
                                        "h_gamma",      "h_phys_max",      "delta",
                                        "kappa",        "delta_over_h_phys", "p",
                                        "phase",        "b_over_delta",    "target_count",
                                        "error_inf",    "quadrature_difference_inf", "maximum_target_parameter"};
  const std::string name = stage == "sweep" ? "sweep.csv" : (stage == "phase" ? "phase_quarter.csv" : "errors.csv");
  for (const std::string& label : labels) {
    const Geometry geom = label == "circle" ? Geometry(circle_geometry)
                                            : (label == "kite_laplace" ? laplace_kite().geometry
                                                                       : stokes_starfish().geometry);
    const auto filename = output_path(root, "results/fixed_width/" + label + "/" + name);
    std::vector<CsvRow> rows;
    for (int n : ns) {
      const double h = 2 * PI / n;
      // x(h k): note h*k rather than 2 pi k/n, as in the Python code.
      std::vector<double> tk(n);
      for (int k = 0; k < n; ++k) tk[k] = h * k;
      const auto nodes = sample(geom, tk);
      double hp = 0;
      for (int k = 0; k < n; ++k) hp = std::max(hp, norm(nodes[(k + 1) % n].x - nodes[k].x));
      std::vector<int> indices(97);
      for (int i = 0; i < 97; ++i) indices[i] = static_cast<int>(std::floor(n * (i + 0.37) / 97));
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
      for (const Num& k : ks) {
        const double delta = k.v * (label == "circle" ? h : hp);
        std::vector<Num> phases = label == "circle" ? std::vector<Num>{Int(0), Flt(0.25), Flt(0.5)}
                                                    : std::vector<Num>{Int(0), Flt(0.5)};
        if (stage == "phase") phases = {Flt(0.25)};
        for (const Num& phase : phases) {
          std::vector<double> theta(indices.size());
          for (size_t i = 0; i < indices.size(); ++i) theta[i] = (indices[i] + phase.v) * h;
          const auto close = sample(geom, theta);
          for (const Num& lam : {Flt(-0.5), Int(0), Flt(0.5)}) {
            const std::vector<double> b(theta.size(), lam.v * delta);
            std::vector<Vec2> targets(theta.size());
            for (size_t i = 0; i < theta.size(); ++i) targets[i] = close[i].x + b[i] * close[i].normal;
            const std::string side = side_name(lam.v);
            for (const FieldErrors& fe : compare(label, targets, theta, b, delta, n, lam.v, side)) {
              // np.argmax: first index of the maximum.
              size_t imax = 0;
              for (size_t i = 1; i < fe.error.size(); ++i)
                if (fe.error[i] > fe.error[imax]) imax = i;
              const double dmax = *std::max_element(fe.difference.begin(), fe.difference.end());
              rows.push_back({{"geometry", label},
                              {"field", fe.field},
                              {"n_gamma", std::int64_t(n)},
                              {"h_gamma", h},
                              {"h_phys_max", hp},
                              {"delta", delta},
                              {"kappa", k.cell()},
                              {"delta_over_h_phys", delta / hp},
                              {"p", std::int64_t(7)},
                              {"phase", phase.cell()},
                              {"b_over_delta", lam.cell()},
                              {"target_count", std::int64_t(theta.size())},
                              {"error_inf", fe.error[imax]},
                              {"quadrature_difference_inf", dmax},
                              {"maximum_target_parameter", theta[imax]}});
            }
          }
        }
        write_csv(filename, fields, rows);
        std::printf("%s %s N=%d k=%s\n", stage.c_str(), label.c_str(), n, k.text().c_str());
        std::fflush(stdout);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string stage;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--stage") == 0) stage = argv[i + 1];
  if (stage != "circle" && stage != "complete" && stage != "sweep" && stage != "phase") {
    std::fprintf(stderr, "fixed_width: --stage must be circle, complete, sweep or phase (moments is not ported)\n");
    return 2;
  }
  const auto root = output_root(argc, argv);
  for (const char* folder : {"discrete_moments", "circle", "kite_laplace", "starfish_stokes", "allgrid"})
    std::filesystem::create_directories(output_path(root, std::string("results/fixed_width/") + folder));
  local(root, stage);
  return 0;
}
