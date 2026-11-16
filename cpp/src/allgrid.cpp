#include "regbie/allgrid.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include "regbie/layers.hpp"
#include "regbie/numerics.hpp"
#include "regbie/periodic.hpp"

namespace regbie {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
using VectorGrid = std::array<Grid, 2>;

// np.max: the largest value, NaN if any value is NaN.
double nan_max(const std::vector<double>& v) {
  double r = v.at(0);
  for (double x : v) {
    if (std::isnan(x)) return x;
    if (x > r) r = x;
  }
  return r;
}

// np.argmax: index of the first maximum (or of the first NaN).
long nan_argmax(const std::vector<double>& v) {
  long best = 0;
  for (long k = 0; k < static_cast<long>(v.size()); ++k) {
    if (std::isnan(v[k])) return k;
    if (v[k] > v[best]) best = k;
  }
  return best;
}

std::vector<double> flatten(const Grid& g) { return std::vector<double>(g.data(), g.data() + g.size()); }

std::vector<Vec2> flatten(const VectorGrid& g) {
  std::vector<Vec2> v(g[0].size());
  for (long k = 0; k < g[0].size(); ++k) v[k] = {g[0].data()[k], g[1].data()[k]};
  return v;
}

// g[mask] and g[mask] = v in NumPy (row-major) order.
std::vector<double> gather(const Grid& g, const Mask& mask) {
  std::vector<double> v;
  for (const auto& ji : mask_indices(mask)) v.push_back(g(ji[0], ji[1]));
  return v;
}
std::vector<Vec2> gather(const VectorGrid& g, const Mask& mask) {
  std::vector<Vec2> v;
  for (const auto& ji : mask_indices(mask)) v.push_back({g[0](ji[0], ji[1]), g[1](ji[0], ji[1])});
  return v;
}
void scatter(Grid& g, const Mask& mask, const std::vector<double>& v) {
  const auto idx = mask_indices(mask);
  for (size_t k = 0; k < idx.size(); ++k) g(idx[k][0], idx[k][1]) = v[k];
}
void scatter(VectorGrid& g, const Mask& mask, const std::vector<Vec2>& v) {
  const auto idx = mask_indices(mask);
  for (size_t k = 0; k < idx.size(); ++k) {
    g[0](idx[k][0], idx[k][1]) = v[k].x;
    g[1](idx[k][0], idx[k][1]) = v[k].y;
  }
}

Mask mask_or(const Mask& a, const Mask& b) {
  Mask c(a.rows(), a.cols());
  for (long k = 0; k < a.size(); ++k) c.data()[k] = a.data()[k] || b.data()[k];
  return c;
}
Mask mask_not(const Mask& a) {
  Mask c(a.rows(), a.cols());
  for (long k = 0; k < a.size(); ++k) c.data()[k] = !a.data()[k];
  return c;
}
int count(const Mask& a) { return static_cast<int>(a.count()); }

// full[1:-1, 1:-1]
Grid interior(const Grid& full) { return full.block(1, 1, full.rows() - 2, full.cols() - 2); }

VectorGrid constant_vector(long rows, long cols, double value) {
  return {Grid(Grid::Constant(rows, cols, value)), Grid(Grid::Constant(rows, cols, value))};
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
  std::vector<double> c(a.size());
  for (size_t k = 0; k < a.size(); ++k) c[k] = a[k] - b[k];
  return c;
}
std::vector<Vec2> subtract(const std::vector<Vec2>& a, const std::vector<Vec2>& b) {
  std::vector<Vec2> c(a.size());
  for (size_t k = 0; k < a.size(); ++k) c[k] = a[k] - b[k];
  return c;
}
std::vector<double> component(const std::vector<Vec2>& v, int c) {
  std::vector<double> out(v.size());
  for (size_t k = 0; k < v.size(); ++k) out[k] = c == 0 ? v[k].x : v[k].y;
  return out;
}
std::vector<double> abs_values(const std::vector<double>& v) {
  std::vector<double> a(v.size());
  for (size_t k = 0; k < v.size(); ++k) a[k] = std::fabs(v[k]);
  return a;
}
std::vector<double> lengths(const std::vector<Vec2>& v) {
  std::vector<double> a(v.size());
  for (size_t k = 0; k < v.size(); ++k) a[k] = norm(v[k]);
  return a;
}
// h sqrt(sum v^2)
double l2(const std::vector<double>& v, double h) {
  std::vector<double> sq(v.size());
  for (size_t k = 0; k < v.size(); ++k) sq[k] = v[k] * v[k];
  return h * std::sqrt(pairwise_sum(sq.data(), static_cast<long>(sq.size())));
}

std::vector<Vec2> interior_points(const GridCoords& grid) {
  const long m = grid.xx.rows() - 2;
  std::vector<Vec2> p;
  p.reserve(m * m);
  for (long j = 0; j < m; ++j)
    for (long i = 0; i < m; ++i) p.push_back(grid.point(j + 1, i + 1));
  return p;
}

}  // namespace

// NumPy pairwise summation: blocks of at most 128 with eight accumulators.
double pairwise_sum(const double* a, long n) {
  if (n < 8) {
    double res = -0.0;
    for (long i = 0; i < n; ++i) res += a[i];
    return res;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    long i;
    for (i = 8; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return res;
  }
  long n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

BoundaryPoints boundary_points(const GridCoords& grid) {
  BoundaryPoints b;
  b.mask = outer_ring(static_cast<int>(grid.xx.rows()), static_cast<int>(grid.xx.cols()));
  b.points = masked_points(grid, b.mask);
  return b;
}

// ---------------------------------------------------------------------------
// allgrid.solve_laplace / solve_stokes

AllgridLaplaceResult solve_laplace(const LaplaceProblem& problem, int m, int n_gamma, double delta, int order,
                                   int oversample_factor) {
  const double start = perf_counter();
  const GridCoords grid = grid_coordinates(m, problem.box);
  const double h = grid.h;
  const Mask c_delta = classify_crossing(grid, LAPLACE_OFFSETS, problem.inside);
  const Mask e_l = stencil_halo(c_delta);
  const auto targets = masked_points(grid, e_l);
  std::vector<double> f, g;
  problem.boundary_densities(periodic_nodes(n_gamma), f, g);
  const PeriodicSamples fi(f), gi(g);
  const ClosestPoints cp = closest_points(targets, problem.geometry, problem.inside);
  const double bie_start = perf_counter();
  const auto direct =
      eval_laplace_regularized(problem.geometry, fi, gi, targets, cp.t, cp.b, delta, n_gamma, order);
  const int n_check = oversample_factor * n_gamma;
  std::vector<double> f_check, g_check;
  problem.boundary_densities(periodic_nodes(n_check), f_check, g_check);
  const auto direct_check = eval_laplace_regularized(problem.geometry, PeriodicSamples(f_check),
                                                     PeriodicSamples(g_check), targets, cp.t, cp.b, delta,
                                                     n_check, order);
  const double oversampled_error = nan_max(abs_values(subtract(direct_check, direct)));
  const BoundaryPoints outer = boundary_points(grid);
  const auto boundary_values = unregularized_laplace(problem.geometry, fi, gi, outer.points, n_gamma);
  const double bie_seconds = perf_counter() - bie_start;

  // Right side: nine-point Laplacian of the direct values on the crossing set, zero elsewhere.
  Grid direct_full = Grid::Constant(m + 2, m + 2, NaN);
  scatter(direct_full, e_l, direct);
  const Grid lap = nine_point_laplacian(direct_full, h);
  Grid rhs = Grid::Zero(m, m);
  for (const auto& ji : mask_indices(c_delta)) rhs(ji[0], ji[1]) = lap(ji[0], ji[1]);
  Grid boundary = Grid::Zero(m + 2, m + 2);
  scatter(boundary, outer.mask, boundary_values);
  const double dst_start = perf_counter();
  Grid solved = dst_solve_nine_point(rhs, boundary, h);
  const double dst_seconds = perf_counter() - dst_start;
  scatter(solved, e_l, direct);

  Grid error(m + 2, m + 2);
  for (int j = 0; j < m + 2; ++j)
    for (int i = 0; i < m + 2; ++i) error(j, i) = solved(j, i) - problem.exact(grid.point(j, i), Side::Piecewise);

  AllgridLaplaceResult r;
  r.values = solved;
  r.error = error;
  r.xx = grid.xx;
  r.yy = grid.yy;
  r.c_delta = c_delta;
  r.halo = e_l;
  r.h = h;
  r.error_inf = nan_max(abs_values(flatten(error)));
  r.error_l2 = l2(flatten(error), h);
  r.oversampled_error_inf = oversampled_error;
  r.direct_targets = count(e_l);
  r.bie_seconds = bie_seconds;
  r.dst_seconds = dst_seconds;
  r.total_seconds = perf_counter() - start;
  return r;
}

AllgridStokesResult solve_stokes(const StokesProblem& problem, int m, int n_gamma, double delta, int order,
                                 int oversample_factor) {
  const double start = perf_counter();
  const GridCoords grid = grid_coordinates(m, problem.box);
  const double h = grid.h, mu = problem.mu;
  const int rows = m + 2;
  const Mask c_delta = classify_crossing(grid, LAPLACE_OFFSETS, problem.inside);
  const Mask c_gradient = classify_crossing(grid, GRADIENT_OFFSETS, problem.inside);
  const Mask c_union = mask_or(c_delta, c_gradient);
  const Mask halo = stencil_halo(c_union);
  const auto targets = masked_points(grid, halo);
  std::vector<Vec2> q, f;
  problem.boundary_data(periodic_nodes(n_gamma), mu, q, f);
  const PressureBoundaryData pdata = pressure_densities(problem.geometry, q, f, mu);
  const PeriodicSamples phi(pdata.phi), psi(pdata.psi);
  const ClosestPoints cp = closest_points(targets, problem.geometry, problem.inside);
  const double bie_start = perf_counter();
  const auto p_direct = eval_pressure_regularized(problem.geometry, q, f, targets, cp.t, cp.b, delta, order, mu);
  const int n_check = oversample_factor * n_gamma;
  std::vector<Vec2> q_check, f_check;
  problem.boundary_data(periodic_nodes(n_check), mu, q_check, f_check);
  const auto p_direct_check =
      eval_pressure_regularized(problem.geometry, q_check, f_check, targets, cp.t, cp.b, delta, order, mu);
  const BoundaryPoints outer = boundary_points(grid);
  const auto p_boundary = unregularized_laplace(problem.geometry, phi, psi, outer.points, n_gamma, -1.0);

  // Pressure: Lap p = 0 on each side, with the jump carried by the crossing set.
  Grid p_full = Grid::Constant(rows, rows, NaN);
  scatter(p_full, halo, p_direct);
  const Grid lap_p = nine_point_laplacian(p_full, h);
  Grid p_rhs = Grid::Zero(m, m);
  for (const auto& ji : mask_indices(c_delta)) p_rhs(ji[0], ji[1]) = lap_p(ji[0], ji[1]);
  Grid p_bc = Grid::Zero(rows, rows);
  scatter(p_bc, outer.mask, p_boundary);
  double dst_start = perf_counter();
  const Grid p_solved = dst_solve_nine_point(p_rhs, p_bc, h);
  const double p_dst = perf_counter() - dst_start;
  Grid p_all = p_solved;
  scatter(p_all, halo, p_direct);

  const auto u_direct = eval_stokes_regularized(problem.geometry, q, f, targets, cp.t, cp.b, delta, order, mu);
  const auto u_direct_check =
      eval_stokes_regularized(problem.geometry, q_check, f_check, targets, cp.t, cp.b, delta, order, mu);
  const double p_check = nan_max(abs_values(subtract(p_direct_check, p_direct)));
  const double u_check = nan_max(lengths(subtract(u_direct_check, u_direct)));
  const auto u_boundary = unregularized_stokes(problem.geometry, q, f, outer.points, mu);
  const double bie_seconds = perf_counter() - bie_start;

  // Velocity: mu Lap u = grad p away from the interface, nine-point Laplacian of
  // the direct values on the union of the crossing sets.
  VectorGrid u_full = constant_vector(rows, rows, NaN);
  scatter(u_full, halo, u_direct);
  const VectorGrid l_u{nine_point_laplacian(u_full[0], h), nine_point_laplacian(u_full[1], h)};
  const Gradient p_gradient = grad4(p_solved, h);
  VectorGrid u_rhs{Grid(m, m), Grid(m, m)};
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i) {
      if (c_union(j, i)) {
        u_rhs[0](j, i) = l_u[0](j, i);
        u_rhs[1](j, i) = l_u[1](j, i);
      } else {
        u_rhs[0](j, i) = p_gradient.gx(j, i) / mu;
        u_rhs[1](j, i) = p_gradient.gy(j, i) / mu;
      }
    }
  dst_start = perf_counter();
  VectorGrid u_solved;
  for (int k = 0; k < 2; ++k) {
    Grid bc = Grid::Zero(rows, rows);
    scatter(bc, outer.mask, component(u_boundary, k));
    u_solved[k] = dst_solve_nine_point(u_rhs[k], bc, h);
  }
  const double u_dst = perf_counter() - dst_start;
  scatter(u_solved, halo, u_direct);

  Grid pe(rows, rows);
  VectorGrid ue{Grid(rows, rows), Grid(rows, rows)};
  for (int j = 0; j < rows; ++j)
    for (int i = 0; i < rows; ++i) {
      const StokesSolution s = problem.exact(grid.point(j, i), Side::Piecewise, 1.0);
      pe(j, i) = p_all(j, i) - s.p;
      ue[0](j, i) = u_solved[0](j, i) - s.u.x;
      ue[1](j, i) = u_solved[1](j, i) - s.u.y;
    }
  // Gradient error at the interior nodes whose gradient stencil stays on one side.
  const auto inner = interior_points(grid);
  std::vector<double> ge_safe;
  for (const auto& ji : mask_indices(mask_not(c_gradient))) {
    const Vec2 g = problem.exact_gradient(inner[ji[0] * m + ji[1]]);
    ge_safe.push_back(norm(Vec2{p_gradient.gx(ji[0], ji[1]), p_gradient.gy(ji[0], ji[1])} - g));
  }
  // np.sum(ue**2) runs over the interleaved (rows, rows, 2) array.
  std::vector<double> ue_flat;
  ue_flat.reserve(2 * rows * rows);
  for (long k = 0; k < ue[0].size(); ++k) {
    ue_flat.push_back(ue[0].data()[k]);
    ue_flat.push_back(ue[1].data()[k]);
  }

  AllgridStokesResult r;
  r.pressure = p_all;
  r.velocity = u_solved;
  r.pressure_error = pe;
  r.velocity_error = ue;
  r.xx = grid.xx;
  r.yy = grid.yy;
  r.h = h;
  r.c_delta = c_delta;
  r.c_gradient = c_gradient;
  r.halo = halo;
  r.pressure_error_inf = nan_max(abs_values(flatten(pe)));
  r.pressure_error_l2 = l2(flatten(pe), h);
  r.velocity_error_inf = nan_max(lengths(flatten(ue)));
  r.velocity_error_l2 = l2(ue_flat, h);
  r.gradient_error_inf = nan_max(ge_safe);
  r.gradient_error_l2 = l2(ge_safe, h);
  r.oversampled_error_inf = std::max(p_check, u_check);
  r.pressure_oversampled_error_inf = p_check;
  r.velocity_oversampled_error_inf = u_check;
  r.direct_targets = count(halo);
  r.bie_seconds = bie_seconds;
  r.dst_seconds = p_dst + u_dst;
  r.total_seconds = perf_counter() - start;
  return r;
}

// ---------------------------------------------------------------------------
// matched_allgrid

Spacing spacing(const Geometry& geometry, double h, std::optional<int> n_given, double ratio) {
  if (h <= 0 || ratio <= 0) throw std::invalid_argument("grid spacings must be positive");
  const bool automatic = !n_given;
  long n;
  if (automatic) {
    // max |x'(t)| on np.linspace(0, 2 pi, 16384, endpoint=False)
    const int samples = 16384;
    const double step = 2 * PI / samples;
    double speed = geometry(0.0).speed;
    for (int i = 1; i < samples; ++i) speed = std::max(speed, geometry(i * step).speed);
    n = next_fast_len(static_cast<long>(std::ceil(2 * PI * speed / (ratio * h))));
    while (n % 2) n = next_fast_len(n + 1);
  } else {
    n = *n_given;
  }
  std::vector<double> chord;
  double chord_max = 0;
  for (int attempt = 0;; ++attempt) {
    if (attempt == 8) throw std::runtime_error("unable to match discrete chord spacing");
    // |x_{k+1} - x_k| on the n-point periodic grid.
    const auto x = sample(geometry, static_cast<int>(n));
    chord.assign(n, 0.0);
    for (long k = 0; k < n; ++k) chord[k] = norm(x[(k + 1) % n].x - x[k].x);
    chord_max = nan_max(chord);
    if (!automatic || (0.8 <= chord_max / (ratio * h) && chord_max / (ratio * h) <= 1.2)) break;
    n = next_fast_len(std::max(4L, static_cast<long>(std::ceil(n * chord_max / (ratio * h)))));
    while (n % 2) n = next_fast_len(n + 1);
  }
  Spacing s;
  s.n_gamma = static_cast<int>(n);
  s.chord_max = chord_max;
  s.chord_mean = pairwise_sum(chord.data(), n) / n;
  s.spacing_ratio = chord_max / h;
  return s;
}

Json norms(const std::vector<double>& v, double h) {
  const auto a = abs_values(v);
  Json out = Json::object();
  out["inf"] = nan_max(a);
  out["l2"] = l2(a, h);
  return out;
}

Json norms(const std::vector<Vec2>& v, double h) {
  const auto a = lengths(v);
  Json out = Json::object();
  out["inf"] = nan_max(a);
  out["l2"] = l2(a, h);
  return out;
}

namespace {

struct Reconstruction {
  Grid solved, hybrid, source_error, full;
};

// matched_allgrid.run for both problem types (Stokes when Problem is StokesProblem).
template <class Problem>
MatchedRun run_impl(const Problem& problem, int m, const MatchedOptions& o) {
  constexpr bool sto = std::is_same<Problem, StokesProblem>::value;
  const double start = perf_counter();
  const GridCoords grid = grid_coordinates(m, problem.box);
  const double h = grid.h;
  const int rows = m + 2;
  std::optional<int> n_rule = o.n_override;
  if (!n_rule && o.rule == "old8m") n_rule = 8 * m;
  const Spacing pars = spacing(problem.geometry, h, n_rule, o.rule == "twice" ? 0.5 : 1.0);
  const int n = pars.n_gamma;
  const double delta = o.delta_override ? *o.delta_override : o.kappa.number() * std::pow(h, 5.0 / 7);
  if (delta <= 0) throw std::invalid_argument("smoothing width must be positive");
  const Mask cd = classify_crossing(grid, LAPLACE_OFFSETS, problem.inside);
  const Mask cg = sto ? classify_crossing(grid, GRADIENT_OFFSETS, problem.inside) : cd;
  const Mask cu = mask_or(cd, cg);
  const Mask halo = stencil_halo(cu);
  const BoundaryPoints outer = boundary_points(grid);
  const auto halo_points = masked_points(grid, halo);

  // Exact (piecewise) fields on the full grid: pe is the pressure or the Laplace solution.
  Grid pe(rows, rows);
  VectorGrid ue = constant_vector(rows, rows, 0.0);
#pragma omp parallel for schedule(dynamic, 4)
  for (int j = 0; j < rows; ++j)
    for (int i = 0; i < rows; ++i) {
      if constexpr (sto) {
        const StokesSolution s = problem.exact(grid.point(j, i), Side::Piecewise, 1.0);
        pe(j, i) = s.p;
        ue[0](j, i) = s.u.x;
        ue[1](j, i) = s.u.y;
      } else {
        pe(j, i) = problem.exact(grid.point(j, i), Side::Piecewise);
      }
    }
  const auto direct_exact = gather(pe, halo);
  std::vector<Vec2> u_direct_exact;
  if (sto) u_direct_exact = gather(ue, halo);

  Json report = Json::object();
  report["m_grid"] = m;
  report["h_c"] = h;
  report["geometry"] = problem.name;
  report["mode"] = o.mode;
  report["rule"] = o.rule;
  report["kappa"] = o.kappa;
  report["delta"] = delta;
  report["delta_over_h"] = delta / h;
  report["delta_over_chord"] = delta / pars.chord_max;
  report["n_gamma"] = pars.n_gamma;
  report["chord_max"] = pars.chord_max;
  report["chord_mean"] = pars.chord_mean;
  report["spacing_ratio"] = pars.spacing_ratio;
  report["direct_targets"] = count(halo);
  report["crossing_count"] = count(cd);
  report["gradient_crossing_count"] = count(cg);
  report["union_count"] = count(cu);

  // Direct values on the halo and outer boundary values: regularized/unregularized
  // sums (bie) or the exact solution (exact_direct, exact_gradient).
  std::vector<double> direct, pb;
  std::vector<Vec2> ud, ub;
  if (o.mode == "bie") {
    const auto t = periodic_nodes(n);
    const ClosestPoints cp = closest_points(halo_points, problem.geometry, problem.inside);
    if constexpr (sto) {
      std::vector<Vec2> q, f;
      problem.boundary_data(t, 1.0, q, f);
      const PressureBoundaryData pd = pressure_densities(problem.geometry, q, f);
      direct = eval_pressure_regularized(problem.geometry, q, f, halo_points, cp.t, cp.b, delta, 7);
      ud = eval_stokes_regularized(problem.geometry, q, f, halo_points, cp.t, cp.b, delta, 7);
      pb = unregularized_laplace(problem.geometry, PeriodicSamples(pd.phi), PeriodicSamples(pd.psi),
                                 outer.points, n, -1.0);
      ub = unregularized_stokes(problem.geometry, q, f, outer.points);
    } else {
      std::vector<double> f, g;
      problem.boundary_densities(t, f, g);
      const PeriodicSamples fi(f), gi(g);
      direct = eval_laplace_regularized(problem.geometry, fi, gi, halo_points, cp.t, cp.b, delta, n, 7);
      pb = unregularized_laplace(problem.geometry, fi, gi, outer.points, n);
    }
  } else {
    direct = direct_exact;
    pb = gather(pe, outer.mask);
    if (sto) {
      ud = u_direct_exact;
      ub = gather(ue, outer.mask);
    }
  }
  report["direct_seconds"] = perf_counter() - start;

  // Nine-point solve with the Laplacian of the direct values on `cross` as source;
  // the hybrid field keeps the direct values on the halo.
  auto reconstruct = [&](const std::vector<double>& d, const Grid& exact_field, const std::vector<double>& bc,
                         const Mask& cross) {
    Reconstruction r;
    r.full = Grid::Constant(rows, rows, NaN);
    scatter(r.full, halo, d);
    const Grid ld = nine_point_laplacian(r.full, h), le = nine_point_laplacian(exact_field, h);
    Grid rhs = Grid::Zero(m, m);
    for (const auto& ji : mask_indices(cross)) rhs(ji[0], ji[1]) = ld(ji[0], ji[1]);
    Grid boundary = Grid::Zero(rows, rows);
    scatter(boundary, outer.mask, bc);
    r.solved = dst_solve_nine_point(rhs, boundary, h);
    r.hybrid = r.solved;
    scatter(r.hybrid, halo, d);
    r.source_error = ld - le;
    return r;
  };
  const Reconstruction p = reconstruct(direct, pe, pb, cd);
  const Grid pa_error = p.hybrid - pe;
  const std::string key = sto ? "pressure" : "laplace";
  report[key] = norms(flatten(pa_error), h);
  report["direct_" + key] = norms(subtract(direct, direct_exact), h);
  report["crossing_direct_" + key] = norms(gather(interior(Grid(p.full - pe)), cd), h);
  report[key + "_source"] = norms(gather(p.source_error, cd), h);
  report["outer_" + key] = norms(subtract(pb, gather(pe, outer.mask)), h);

  MatchedRun run;
  run.fields.xx = grid.xx;
  run.fields.yy = grid.yy;
  run.fields.error = pa_error;
  run.fields.c_delta = cd;
  run.fields.c_gradient = cg;
  run.fields.halo = halo;

  if constexpr (sto) {
    const Mask safe = mask_not(cu);
    VectorGrid ge{Grid(m, m), Grid(m, m)};
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i) {
        const Vec2 g = problem.exact_gradient(grid.point(j + 1, i + 1));
        ge[0](j, i) = g.x;
        ge[1](j, i) = g.y;
      }
    Gradient gp = grad4(p.solved, h);
    if (o.mode == "exact_gradient")
      for (const auto& ji : mask_indices(safe)) {
        gp.gx(ji[0], ji[1]) = ge[0](ji[0], ji[1]);
        gp.gy(ji[0], ji[1]) = ge[1](ji[0], ji[1]);
      }
    report["pressure_reconstructed"] = norms(flatten(Grid(p.solved - pe)), h);
    report["gradient"] = norms(gather(VectorGrid{Grid(gp.gx - ge[0]), Grid(gp.gy - ge[1])}, safe), h);
    report["direct_velocity"] = norms(subtract(ud, u_direct_exact), h);
    VectorGrid uf = constant_vector(rows, rows, NaN);
    scatter(uf, halo, ud);
    const VectorGrid lu{nine_point_laplacian(uf[0], h), nine_point_laplacian(uf[1], h)};
    const VectorGrid lue{nine_point_laplacian(ue[0], h), nine_point_laplacian(ue[1], h)};
    report["velocity_source_crossing"] =
        norms(gather(VectorGrid{Grid(lu[0] - lue[0]), Grid(lu[1] - lue[1])}, cu), h);
    report["crossing_direct_velocity"] =
        norms(gather(VectorGrid{interior(Grid(uf[0] - ue[0])), interior(Grid(uf[1] - ue[1]))}, cu), h);
    // Source: Laplacian of the direct velocity on the crossing union, grad p / mu elsewhere.
    VectorGrid rhs{Grid(m, m), Grid(m, m)};
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < m; ++i) {
        if (cu(j, i)) {
          rhs[0](j, i) = lu[0](j, i);
          rhs[1](j, i) = lu[1](j, i);
        } else {
          rhs[0](j, i) = gp.gx(j, i) / problem.mu;
          rhs[1](j, i) = gp.gy(j, i) / problem.mu;
        }
      }
    VectorGrid ua;
    for (int k = 0; k < 2; ++k) {
      Grid bc = Grid::Zero(rows, rows);
      scatter(bc, outer.mask, component(ub, k));
      ua[k] = dst_solve_nine_point(rhs[k], bc, h);
      scatter(ua[k], halo, component(ud, k));
    }
    const VectorGrid u_error{Grid(ua[0] - ue[0]), Grid(ua[1] - ue[1])};
    report["velocity"] = norms(flatten(u_error), h);
    run.fields.pressure_error = pa_error;
    run.fields.velocity_error = u_error;

    // Pressure error along the grid rows nearest to the requested heights.
    Json stripe = Json::array();
    for (double target : {0.0, h, -h, 2 * h, -2 * h, 0.1, -0.1}) {
      int j = 0;
      for (int r = 1; r < rows; ++r)
        if (std::fabs(grid.yy(r, 0) - target) < std::fabs(grid.yy(j, 0) - target)) j = r;
      std::vector<double> e(rows);
      for (int i = 0; i < rows; ++i) e[i] = pa_error(j, i);
      int zeros = 0;
      bool finite = true;
      for (double x : e) {
        zeros += x == 0;
        finite = finite && std::isfinite(x);
      }
      Json row = Json::object();
      row["requested_y"] = target;
      row["actual_y"] = grid.yy(j, 0);
      row["row"] = j;
      row["max_abs"] = nan_max(abs_values(e));
      row["zeros"] = zeros;
      row["finite"] = finite;
      stripe.push_back(row);
      run.fields.profile_rows.push_back(j);
    }
    report["stripe_rows"] = stripe;
    run.fields.pressure_profiles.resize(static_cast<long>(run.fields.profile_rows.size()), rows);
    for (size_t k = 0; k < run.fields.profile_rows.size(); ++k)
      run.fields.pressure_profiles.row(k) = pa_error.row(run.fields.profile_rows[k]);
  }
  report["total_seconds"] = perf_counter() - start;
  run.report = report;
  return run;
}

Json node(const MatchedFields& fields, const Grid& magnitude) {
  const long k = nan_argmax(flatten(magnitude));
  const int j = static_cast<int>(k / magnitude.cols()), i = static_cast<int>(k % magnitude.cols());
  Json out = Json::object();
  out["index"] = std::vector<int>{j, i};
  out["point"] = std::vector<double>{fields.xx(j, i), fields.yy(j, i)};
  out["error"] = magnitude(j, i);
  out["in_direct_halo"] = static_cast<bool>(fields.halo(j, i));
  return out;
}

}  // namespace

MatchedRun matched_run(const LaplaceProblem& problem, int m, const MatchedOptions& options) {
  return run_impl(problem, m, options);
}

MatchedRun matched_run(const StokesProblem& problem, int m, const MatchedOptions& options) {
  return run_impl(problem, m, options);
}

Json maximum_node(const MatchedFields& fields, const Grid& error) {
  return node(fields, Grid(error.abs()));
}

Json maximum_node(const MatchedFields& fields, const std::array<Grid, 2>& error) {
  // sqrt(e_x^2 + e_y^2), as np.linalg.norm
  return node(fields, Grid((error[0] * error[0] + error[1] * error[1]).sqrt()));
}

// ---------------------------------------------------------------------------
// experiments/problems.py: five-lobed starfish (Laplace) and nonconvex kite (Stokes),
// with the manufactured fields of manufactured_laplace.py and manufactured_stokes.py.

namespace {

using cplx = std::complex<double>;
const cplx I(0.0, 1.0);
cplx to_complex(Vec2 p) { return {p.x, p.y}; }

// z**n for an integer 1 <= n < 100 as NumPy computes it (repeated squaring).
cplx ipow(cplx z, int n) {
  if (n == 1) return z;
  if (n == 2) return z * z;
  if (n == 3) return z * (z * z);
  cplx result(1.0, 0.0), p = z;
  for (int mask = 1;; mask <<= 1) {
    if (n & mask) result = result * p;
    if (n < (mask << 1)) break;
    p = p * p;
  }
  return result;
}

// Starfish Laplace: F- analytic inside, F+ analytic outside (config.STARFISH_*).
const cplx SM_A1(1.65, 0.15), SM_A2(-1.45, 0.65);
const cplx SP_Z1(0.18, 0.12), SP_Z2(-0.22, 0.08), SP_Z3(0.05, -0.25);
const cplx SP_R1(0.40, 0.08), SP_R2(0.0, -0.18), SP_R3(0.08, 0.0);
const double SM_EXPONENTIAL = 0.35, SM_EXPONENT = 0.30, SM_Z4 = 0.08;
const cplx SM_Z3(0.0, 0.12), SM_R1(0.25, 0.0), SM_R2(0.0, -0.16);

cplx starfish_fminus(cplx z) {
  return SM_EXPONENTIAL * std::exp(SM_EXPONENT * z) + SM_Z4 * ipow(z, 4) + SM_Z3 * ipow(z, 3) +
         SM_R1 / (z - SM_A1) + SM_R2 / (z - SM_A2);
}
cplx starfish_dfminus(cplx z) {
  return SM_EXPONENTIAL * SM_EXPONENT * std::exp(SM_EXPONENT * z) + 4 * SM_Z4 * ipow(z, 3) +
         3.0 * SM_Z3 * ipow(z, 2) - SM_R1 / ipow(z - SM_A1, 2) - SM_R2 / ipow(z - SM_A2, 2);
}
cplx starfish_fplus(cplx z) {
  return SP_R1 / (z - SP_Z1) + SP_R2 / ipow(z - SP_Z2, 2) + SP_R3 / ipow(z - SP_Z3, 3);
}
cplx starfish_dfplus(cplx z) {
  return -SP_R1 / ipow(z - SP_Z1, 2) - 2.0 * SP_R2 / ipow(z - SP_Z2, 3) - 3.0 * SP_R3 / ipow(z - SP_Z3, 4);
}
Vec2 harmonic_gradient(cplx derivative) { return {derivative.real(), -derivative.imag()}; }

// Kite Stokes: Goursat functions F, G (config.KITE_*); psi = Re(conj(z) F + G).
const cplx KF_A1(2.10, 0.40), KF_A2(-2.20, 0.50), KG_A3(0.40, 2.30), KG_A4(-0.60, -2.20);
const double KF_Z4 = 0.10, KF_EXPONENTIAL = 0.12, KF_EXPONENT = 0.25;
const cplx KF_Z2(0.06, 0.04), KF_R1(0.15, 0.0), KF_R2(0.0, -0.10);
const double KG_Z3 = 0.09, KG_EXPONENTIAL = 0.08;
const cplx KG_Z5(0.0, -0.06), KG_EXPONENT(0.0, -0.20), KG_R1(0.10, 0.0), KG_R2(0.0, 0.07);

cplx kite_F(cplx z) {
  return KF_Z4 * ipow(z, 4) + KF_Z2 * ipow(z, 2) + KF_EXPONENTIAL * std::exp(KF_EXPONENT * z) +
         KF_R1 / (z - KF_A1) + KF_R2 / (z - KF_A2);
}
cplx kite_dF(cplx z) {
  return 4 * KF_Z4 * ipow(z, 3) + 2.0 * KF_Z2 * z + KF_EXPONENTIAL * KF_EXPONENT * std::exp(KF_EXPONENT * z) -
         KF_R1 / ipow(z - KF_A1, 2) - KF_R2 / ipow(z - KF_A2, 2);
}
cplx kite_ddF(cplx z) {
  return 12 * KF_Z4 * ipow(z, 2) + 2.0 * KF_Z2 +
         KF_EXPONENTIAL * (KF_EXPONENT * KF_EXPONENT) * std::exp(KF_EXPONENT * z) +
         2.0 * KF_R1 / ipow(z - KF_A1, 3) + 2.0 * KF_R2 / ipow(z - KF_A2, 3);
}
cplx kite_dG(cplx z) {
  return 5.0 * KG_Z5 * ipow(z, 4) + 3 * KG_Z3 * ipow(z, 2) + KG_EXPONENTIAL * KG_EXPONENT * std::exp(KG_EXPONENT * z) -
         KG_R1 / ipow(z - KG_A3, 2) - KG_R2 / ipow(z - KG_A4, 2);
}
cplx kite_ddG(cplx z) {
  return 20.0 * KG_Z5 * ipow(z, 3) + 6 * KG_Z3 * z +
         KG_EXPONENTIAL * ipow(KG_EXPONENT, 2) * std::exp(KG_EXPONENT * z) + 2.0 * KG_R1 / ipow(z - KG_A3, 3) +
         2.0 * KG_R2 / ipow(z - KG_A4, 3);
}

Vec2 kite_velocity(Vec2 p) {
  const cplx z = to_complex(p);
  const cplx a = kite_F(z), b = kite_dF(z), q = kite_dG(z);
  const double psi_x = (a + std::conj(z) * b + q).real();
  const double psi_y = (-I * a + I * std::conj(z) * b + I * q).real();
  return {psi_y, -psi_x};
}
std::array<std::array<double, 2>, 2> kite_grad_velocity(Vec2 p) {
  const cplx z = to_complex(p);
  const cplx b = kite_dF(z), c = kite_ddF(z), q = kite_ddG(z);
  const double psi_xx = (2.0 * b + std::conj(z) * c + q).real();
  const double psi_xy = (I * (std::conj(z) * c + q)).real();
  const double psi_yy = (2.0 * b - std::conj(z) * c - q).real();
  return {{{psi_xy, psi_yy}, {-psi_xx, -psi_xy}}};
}
double kite_pressure(Vec2 p, double mu) { return -4 * mu * kite_dF(to_complex(p)).imag(); }
Vec2 kite_grad_pressure(Vec2 p, double mu) {
  const cplx c = kite_ddF(to_complex(p));
  return {-4 * mu * c.imag(), -4 * mu * c.real()};
}

}  // namespace

const LaplaceProblem& laplace_starfish_problem() {
  static const LaplaceProblem problem = [] {
    LaplaceProblem pr;
    pr.name = "five_lobed_starfish";
    pr.geometry = laplace_starfish_geometry;
    pr.inside = [](Vec2 p) { return starfish_level(p) < 0; };
    pr.boundary_densities = [](const std::vector<double>& t, std::vector<double>& f, std::vector<double>& g) {
      f.resize(t.size());
      g.resize(t.size());
      for (size_t k = 0; k < t.size(); ++k) {
        const GeomPoint gp = laplace_starfish_geometry(t[k]);
        const cplx z = to_complex(gp.x);
        f[k] = dot(harmonic_gradient(starfish_dfplus(z)) - harmonic_gradient(starfish_dfminus(z)), gp.normal);
        g[k] = starfish_fminus(z).real() - starfish_fplus(z).real();
      }
    };
    pr.exact = [](Vec2 p, Side side) {
      const cplx z = to_complex(p);
      switch (side) {
        case Side::Inside: return starfish_fminus(z).real();
        case Side::Outside: return starfish_fplus(z).real();
        case Side::Average: return 0.5 * (starfish_fminus(z).real() + starfish_fplus(z).real());
        default: return starfish_level(p) < 0 ? starfish_fminus(z).real() : starfish_fplus(z).real();
      }
    };
    pr.box = {-1.6, 1.6, -1.6, 1.6};
    return pr;
  }();
  return problem;
}

const StokesProblem& stokes_kite_problem() {
  static const StokesProblem problem = [] {
    StokesProblem pr;
    pr.name = "nonconvex_kite";
    pr.geometry = stokes_kite_geometry;
    auto inside = std::make_shared<CurveInside>(pr.geometry);
    pr.inside = [inside](Vec2 p) { return (*inside)(p); };
    pr.boundary_data = [](const std::vector<double>& t, double mu, std::vector<Vec2>& q, std::vector<Vec2>& f) {
      q.resize(t.size());
      f.resize(t.size());
      for (size_t k = 0; k < t.size(); ++k) {
        const GeomPoint gp = stokes_kite_geometry(t[k]);
        q[k] = kite_velocity(gp.x);
        const auto grad = kite_grad_velocity(gp.x);
        const double p = kite_pressure(gp.x, mu);
        // f = sigma n, sigma = -p I + mu (grad u + grad u^T)
        double sigma[2][2];
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j) sigma[i][j] = (i == j ? -p : 0.0) + mu * (grad[i][j] + grad[j][i]);
        f[k] = {sigma[0][0] * gp.normal.x + sigma[0][1] * gp.normal.y,
                sigma[1][0] * gp.normal.x + sigma[1][1] * gp.normal.y};
      }
    };
    pr.exact = [inside](Vec2 p, Side side, double mu) {
      const Vec2 u = kite_velocity(p);
      const double pr_ = kite_pressure(p, mu);
      switch (side) {
        case Side::Inside: return StokesSolution{u, pr_};
        case Side::Outside: return StokesSolution{{0.0, 0.0}, 0.0};
        case Side::Average: return StokesSolution{0.5 * u, 0.5 * pr_};
        default: return (*inside)(p) ? StokesSolution{u, pr_} : StokesSolution{{0.0, 0.0}, 0.0};
      }
    };
    pr.exact_gradient = [inside](Vec2 p) { return (*inside)(p) ? kite_grad_pressure(p, 1.0) : Vec2{0.0, 0.0}; };
    pr.box = {-2.2, 1.6, -1.9, 1.9};
    return pr;
  }();
  return problem;
}

}  // namespace regbie
