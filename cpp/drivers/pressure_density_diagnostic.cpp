// Analytic versus Fourier pressure densities in the scaled-starfish Stokes test
// (experiments/pressure_density_diagnostic.py).
//
// The pressure is S phi - D psi with phi = -grad p . n and psi = -p on the curve.
// Compares phi, psi from boundary_data.pressure_densities (spectral derivatives of
// the velocity and traction) with the exact ones, and the resulting pressures at
// 33 targets b = lambda delta, delta = c h_phys, lambda in {-1/2, 0, 1/2}.
//
// Run from the repository root:  cpp/build/pressure_density_diagnostic [--output-root DIR]
// Writes <root>/paper_revision/pressure_density_diagnostic.csv.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "regbie/io.hpp"
#include "regbie/layers.hpp"
#include "regbie/numerics.hpp"
#include "regbie/output.hpp"
#include "regbie/periodic.hpp"
#include "regbie/problems.hpp"

using namespace regbie;

int main(int argc, char** argv) {
  const auto path = output_path(output_root(argc, argv), "results/paper_revision/pressure_density_diagnostic.csv");
  const Geometry geometry = stokes_starfish_geometry;
  const StokesProblem& problem = stokes_starfish();
  const std::vector<std::string> fields{"n_gamma", "c_delta", "h_physical", "delta", "phase", "signed_lambda",
                                        "density_method", "error_inf", "evaluation_seconds",
                                        "density_phi_difference_inf", "density_psi_difference_inf",
                                        "field_difference_inf"};
  std::vector<CsvRow> rows;
  for (int n : {4096, 8192, 16384}) {
    const auto t = periodic_nodes(n);
    const auto nodes = sample(geometry, t);
    std::vector<Vec2> q, f;
    problem.boundary_data(t, 1.0, q, f);
    const PressureBoundaryData numerical = pressure_densities(geometry, q, f);
    std::vector<double> analytic_phi(n), analytic_psi(n);
    double density_error_phi = 0, density_error_psi = 0, physical_h = 0;
    for (int k = 0; k < n; ++k) {
      analytic_phi[k] = -dot(stokes_grad_pressure(nodes[k].x), nodes[k].normal);
      analytic_psi[k] = -stokes_pressure(nodes[k].x);
      density_error_phi = std::max(density_error_phi, std::fabs(numerical.phi[k] - analytic_phi[k]));
      density_error_psi = std::max(density_error_psi, std::fabs(numerical.psi[k] - analytic_psi[k]));
      physical_h = std::max(physical_h, norm(nodes[(k + 1) % n].x - nodes[k].x));
    }
    for (double c_delta : {2.0, 4.0}) {
      const double delta = c_delta * physical_h;
      for (double phase : {0.0, 0.5}) {
        std::vector<double> theta(33);
        for (int i = 0; i < 33; ++i) theta[i] = 2 * PI * i / 33 + phase * 2 * PI / n;
        const auto close = sample(geometry, theta);
        for (double lam : {-0.5, 0.0, 0.5}) {
          const std::vector<double> b(theta.size(), lam * delta);
          std::vector<Vec2> targets(theta.size());
          for (size_t i = 0; i < theta.size(); ++i) targets[i] = close[i].x + b[i] * close[i].normal;
          // Exact pressure: p inside, p/2 on the curve, 0 outside.
          const double multiplier = lam < 0 ? 1.0 : (lam == 0 ? 0.5 : 0.0);
          std::vector<double> exact(theta.size());
          for (size_t i = 0; i < theta.size(); ++i) exact[i] = multiplier * stokes_pressure(targets[i]);
          struct Method {
            const char* name;
            const std::vector<double>& phi;
            const std::vector<double>& psi;
          };
          const Method methods[2] = {{"fourier", numerical.phi, numerical.psi},
                                     {"analytic", analytic_phi, analytic_psi}};
          std::vector<double> values[2];
          for (int j = 0; j < 2; ++j) {
            const double start = perf_counter();
            const std::vector<double>& phi = methods[j].phi;
            const std::vector<double>& psi = methods[j].psi;
            const auto sl = single_layer(geometry, phi, targets, b, delta, 7);
            const auto dl = double_layer(geometry, psi, PeriodicSamples(psi).at(theta), targets, b, delta, 7);
            std::vector<double> value(theta.size());
            double error = 0;
            for (size_t i = 0; i < theta.size(); ++i) {
              value[i] = sl[i] - dl[i];
              error = std::max(error, std::fabs(value[i] - exact[i]));
            }
            values[j] = value;
            rows.push_back({{"n_gamma", std::int64_t(n)},
                            {"c_delta", c_delta},
                            {"h_physical", physical_h},
                            {"delta", delta},
                            {"phase", phase},
                            {"signed_lambda", lam},
                            {"density_method", std::string(methods[j].name)},
                            {"error_inf", error},
                            {"evaluation_seconds", perf_counter() - start},
                            {"density_phi_difference_inf", density_error_phi},
                            {"density_psi_difference_inf", density_error_psi},
                            {"field_difference_inf", 0.0}});
          }
          double diff = 0;
          for (size_t i = 0; i < theta.size(); ++i) diff = std::max(diff, std::fabs(values[0][i] - values[1][i]));
          rows[rows.size() - 1]["field_difference_inf"] = diff;
          rows[rows.size() - 2]["field_difference_inf"] = diff;
        }
      }
      std::printf("%d %s finished\n", n, py_repr(c_delta).c_str());
      std::fflush(stdout);
    }
    write_csv(path, fields, rows);
  }
  return 0;
}
