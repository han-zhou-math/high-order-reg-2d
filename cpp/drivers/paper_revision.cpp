// Focused reproducible studies for the review revision (experiments/paper_revision.py).
//
// Uses the scaled kite/starfish geometry. The independent comparator integrates the
// ORIGINAL kernels on target-dependent, geometrically graded Gauss-Legendre panels,
// with analytic densities and geometry. The plot stage stays in Python.
//
// Run from the repository root:
//   cpp/build/paper_revision --stage all [--output-root DIR]
// Stages: all, ablation, width, benchmark, limits, manufactured_limits, summary.
// The validation (metadata.json) runs before every stage, as in the Python main().
#include <fftw3.h>
#include <omp.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/io.hpp"
#include "regbie/kernels.hpp"
#include "regbie/layers.hpp"
#include "regbie/numerics.hpp"
#include "regbie/experiments/output.hpp"
#include "regbie/periodic.hpp"
#include "regbie/experiments/problems.hpp"
#include "regbie/types.hpp"

using namespace regbie;
namespace fs = std::filesystem;

namespace {

fs::path ROOT;  // <output root>/paper_revision

void require(bool ok, const std::string& what, double value) {
  if (!ok) throw std::runtime_error(what + ": " + py_repr(value));
}

// Timed sections run on one thread, as the Python timings (OPENBLAS_NUM_THREADS=1).
class SingleThread {
 public:
  SingleThread() : saved_(omp_get_max_threads()) { omp_set_num_threads(1); }
  ~SingleThread() { omp_set_num_threads(saved_); }

 private:
  int saved_;
};

// ---------------------------------------------------------------- output rows

// An insertion-ordered dict row (the Python dict(...) passed to csv.DictWriter).
using Row = std::vector<std::pair<std::string, CsvCell>>;
CsvCell Int(long long v) { return CsvCell(static_cast<std::int64_t>(v)); }
CsvCell Str(const std::string& v) { return CsvCell(v); }

void save(const std::string& name, const std::vector<Row>& rows) {
  std::vector<std::string> fields;
  for (const auto& kv : rows.at(0)) fields.push_back(kv.first);
  std::vector<CsvRow> out;
  for (const auto& row : rows) {
    CsvRow r;
    for (const auto& kv : row) r[kv.first] = kv.second;
    out.push_back(r);
  }
  write_csv(ROOT / name, fields, out);
}

Json to_json(const Row& row) {
  Json object = Json::object();
  for (const auto& [key, cell] : row) {
    if (const auto* s = std::get_if<std::string>(&cell)) object[key] = Json(*s);
    else if (const auto* d = std::get_if<double>(&cell)) object[key] = Json(*d);
    else if (const auto* i = std::get_if<std::int64_t>(&cell)) object[key] = Json(static_cast<long long>(*i));
    else object[key] = Json(std::get<bool>(cell));
  }
  return object;
}

void print_line(const std::string& text) {
  std::printf("%s\n", text.c_str());
  std::fflush(stdout);
}

// ---------------------------------------------------------------- basic helpers

// Independent q, f, unrelated by a PDE representation.
struct StokesDensity {
  Vec2 q, f;
};
StokesDensity stokes_densities(double t) {
  return {{std::sin(2 * t) + .25 * std::cos(5 * t), .6 * std::cos(3 * t) - .2 * std::sin(t)},
          {.7 * std::cos(2 * t) + .15 * std::sin(4 * t), .5 * std::sin(3 * t) + .2 * std::cos(t)}};
}

// max |value - reference|, with the Euclidean norm of vector differences.
double max_error(const std::vector<double>& value, const std::vector<double>& reference) {
  double m = 0.0;
  for (size_t i = 0; i < value.size(); ++i) m = std::max(m, std::fabs(value[i] - reference[i]));
  return m;
}
double max_error(const std::vector<Vec2>& value, const std::vector<Vec2>& reference) {
  double m = 0.0;
  for (size_t i = 0; i < value.size(); ++i) m = std::max(m, norm(value[i] - reference[i]));
  return m;
}

template <class T>
std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b) {
  std::vector<T> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  return out;
}

template <class T>
std::vector<T> concat(std::vector<T> a, const std::vector<T>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

bool all_finite(const std::vector<double>& v) {
  return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
}
bool all_finite(const std::vector<Vec2>& v) {
  return std::all_of(v.begin(), v.end(), [](Vec2 x) { return std::isfinite(x.x) && std::isfinite(x.y); });
}

double side_indicator(double b) { return b < 0 ? 1.0 : (b > 0 ? 0.0 : 0.5); }

// theta_i = 2 pi (i + 0.37)/count, targets x(theta_i) + offset n(theta_i).
struct TargetSet {
  std::vector<double> theta, b;
  std::vector<Vec2> y;
};
TargetSet target_set(const Geometry& geometry, double offset, int count = 49) {
  TargetSet s;
  for (int i = 0; i < count; ++i) {
    const double theta = 2 * PI * (i + .37) / count;
    const GeomPoint g = geometry(theta);
    s.theta.push_back(theta);
    s.b.push_back(offset);
    s.y.push_back(g.x + offset * g.normal);
  }
  return s;
}

// Largest chord |x(t_{k+1}) - x(t_k)| of the n-point grid.
double physical_spacing(const Geometry& geometry, int n) {
  const auto nodes = sample(geometry, n);
  double h = 0.0;
  for (int k = 0; k < n; ++k) h = std::max(h, norm(nodes[(k + 1) % n].x - nodes[k].x));
  return h;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

// Ordered stage timings; the median is taken key by key.
using Timing = std::vector<std::pair<std::string, double>>;
Timing median_timing(const std::vector<Timing>& times) {
  Timing out = times.at(0);
  for (size_t j = 0; j < out.size(); ++j) {
    std::vector<double> values;
    for (const auto& t : times) values.push_back(t[j].second);
    out[j].second = median(values);
  }
  return out;
}

// ---------------------------------------------------------------- regularized Stokes

// Separate corrected/raw SLP, DLP and nonoverlapping stage timings.
// Density derivatives and off-node interpolation follow the actual FFT path.
// evaluation_seconds excludes projection; callers add it when applicable.
// coefficient_seconds times the coefficients evaluated once per target in the
// kernel stage plus the scaled_moment_n0 calls (a coefficient evaluation and a
// closed-form moment sum), i.e. the calls coefficient_counter() times in Python.
struct StokesResult {
  std::vector<Vec2> sl, dl, raw_sl, raw_dl;
  double prepare_seconds = 0, local_seconds = 0, kernel_seconds = 0, evaluation_seconds = 0,
         coefficient_seconds = 0;
};

StokesResult stokes_regularized(int n, const std::vector<Vec2>& targets, const std::vector<double>& theta,
                                const std::vector<double>& b, double delta, int order) {
  StokesResult out;
  const double start = perf_counter();
  const auto t = periodic_nodes(n);
  const auto nodes = sample(stokes_starfish_geometry, t);
  std::vector<Vec2> q(n), f(n);
  for (int k = 0; k < n; ++k) {
    const StokesDensity d = stokes_densities(t[k]);
    q[k] = d.q;
    f[k] = d.f;
  }
  // q_s = q'/|x'|, q_ss = q_s'/|x'|.
  auto qs = spectral_derivative(q);
  for (int k = 0; k < n; ++k) qs[k] = qs[k] / nodes[k].speed;
  auto qss = spectral_derivative(qs);
  for (int k = 0; k < n; ++k) qss[k] = qss[k] / nodes[k].speed;
  const PeriodicSamples qi(q), fi(f), qsi(qs), qssi(qss);
  out.prepare_seconds = perf_counter() - start;

  // Local corrections: |b| N0 (n0 n0.f0)/(4 pi alpha) and |b| N0 c2/(2 pi), alpha = sqrt(1 + b kappa).
  const double local_start = perf_counter();
  const size_t m = targets.size();
  const auto close = sample(stokes_starfish_geometry, theta);
  const auto q0 = qi.vec_at(theta), f0 = fi.vec_at(theta), qs0 = qsi.vec_at(theta), qss0 = qssi.vec_at(theta);
  std::vector<Vec2> correction_sl(m), correction_dl(m);
  std::vector<double> coefficient_seconds(m, 0.0);
  for (size_t i = 0; i < m; ++i) {
    const GeomPoint& c = close[i];
    const double alpha = std::sqrt(1 + b[i] * c.kappa);
    require(std::isfinite(alpha) && alpha > 0, "alpha must be positive", alpha);
    const double moment_start = perf_counter();
    const double scale = std::fabs(b[i]) * scaled_moment_n0(order, std::fabs(b[i]) / delta);
    coefficient_seconds[i] += perf_counter() - moment_start;
    const double alpha3 = std::pow(alpha, 3.0), alpha5 = std::pow(alpha, 5.0);
    const Vec2 b0 = c.normal * dot(c.normal, f0[i]) / alpha;
    const Vec2 sym = c.normal * dot(c.tau, qs0[i]) + c.tau * dot(c.normal, qs0[i]);
    Vec2 c2 = sym / alpha3;
    c2 -= (b[i] / (2 * alpha3)) * c.normal * dot(c.normal, qss0[i]);
    c2 += (b[i] * b[i] * c.kappa_s / (2 * alpha5)) * c.normal * dot(c.normal, qs0[i]);
    correction_sl[i] = scale * b0 / (4 * PI);
    correction_dl[i] = scale * c2 / (2 * PI);
  }
  out.local_seconds = perf_counter() - local_start;

  // Trapezoidal sums of the regularized Stokeslet and the subtracted stresslet.
  const double kernel_start = perf_counter();
  const double d2 = std::pow(delta, 2.0), d4 = std::pow(delta, 4.0);
  std::vector<double> weights(n);
  for (int k = 0; k < n; ++k) weights[k] = 2 * PI * nodes[k].speed / n;
  out.sl.resize(m);
  out.dl.resize(m);
  out.raw_sl.resize(m);
  out.raw_dl.resize(m);
#pragma omp parallel for schedule(dynamic, 1)
  for (long i = 0; i < static_cast<long>(m); ++i) {
    const double coefficient_start = perf_counter();
    const Coefficients cs = coefficients(order, std::fabs(b[i]) / delta);
    coefficient_seconds[i] += perf_counter() - coefficient_start;
    Sum2 sl, dl;
    for (int k = 0; k < n; ++k) {
      const Vec2 r = nodes[k].x - targets[i];
      const double r2 = dot(r, r);
      const RadialFactors rf = radial_factors(r2 / d2, cs);
      const double gf = regularized_kernel(std::sqrt(r2), delta, cs);
      const double r_f = dot(r, f[k]);
      sl.add(((r * r_f * rf.a) / (4 * PI * d2) - gf * f[k] / 2.0) * weights[k]);
      const double rq = dot(r, q[k] - q0[i]);
      const double rn = dot(r, nodes[k].normal);
      dl.add((r * (rq * rn * (-rf.ap) / (PI * d4))) * weights[k]);
    }
    out.raw_sl[i] = sl.value();
    out.raw_dl[i] = dl.value() + side_indicator(b[i]) * q0[i];
    out.sl[i] = out.raw_sl[i] + correction_sl[i];
    out.dl[i] = out.raw_dl[i] + correction_dl[i];
  }
  out.kernel_seconds = perf_counter() - kernel_start;
  out.evaluation_seconds = perf_counter() - start;
  for (double s : coefficient_seconds) out.coefficient_seconds += s;
  return out;
}

// ---------------------------------------------------------------- reference quadratures

// numpy.polynomial.legendre.legval by Clenshaw recursion.
double legval(double x, const std::vector<double>& c) {
  if (c.size() == 1) return c[0] + 0.0 * x;
  double c0, c1;
  if (c.size() == 2) {
    c0 = c[0];
    c1 = c[1];
  } else {
    int nd = static_cast<int>(c.size());
    c0 = c[nd - 2];
    c1 = c[nd - 1];
    for (size_t i = 3; i <= c.size(); ++i) {
      const double tmp = c0;
      nd = nd - 1;
      c0 = c[c.size() - i] - (c1 * (nd - 1)) / nd;
      c1 = tmp + (c1 * x * (2 * nd - 1)) / nd;
    }
  }
  return c0 + c1 * x;
}

// numpy.polynomial.legendre.legder (first derivative).
std::vector<double> legder(std::vector<double> c) {
  const int n = static_cast<int>(c.size()) - 1;
  std::vector<double> der(n);
  for (int j = n; j > 2; --j) {
    der[j - 1] = (2 * j - 1) * c[j];
    c[j - 2] += c[j];
  }
  if (n > 1) der[1] = 3 * c[2];
  der[0] = c[1];
  return der;
}

struct GaussRule {
  std::vector<double> x, w;
};

// numpy.polynomial.legendre.leggauss: eigenvalues of the symmetric companion matrix
// (off-diagonal k/sqrt((2k-1)(2k+1))), one Newton step, normalized weights, symmetrization.
GaussRule leggauss(int deg) {
  std::vector<double> c(deg + 1, 0.0);
  c[deg] = 1.0;
  std::vector<double> scl(deg);
  for (int j = 0; j < deg; ++j) scl[j] = 1.0 / std::sqrt(2 * j + 1);
  Eigen::VectorXd diag = Eigen::VectorXd::Zero(deg), sub(deg - 1);
  for (int k = 1; k < deg; ++k) sub[k - 1] = k * scl[k - 1] * scl[k];
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
  solver.computeFromTridiagonal(diag, sub, Eigen::EigenvaluesOnly);
  std::vector<double> x(deg), df(deg), fm(deg), w(deg);
  const auto dc = legder(c);
  const std::vector<double> tail(c.begin() + 1, c.end());
  for (int i = 0; i < deg; ++i) {
    x[i] = solver.eigenvalues()[i];
    const double dy = legval(x[i], c);
    df[i] = legval(x[i], dc);
    x[i] -= dy / df[i];
  }
  double fmax = 0.0, dmax = 0.0;
  for (int i = 0; i < deg; ++i) {
    fm[i] = legval(x[i], tail);
    fmax = std::max(fmax, std::fabs(fm[i]));
    dmax = std::max(dmax, std::fabs(df[i]));
  }
  for (int i = 0; i < deg; ++i) w[i] = 1 / ((fm[i] / fmax) * (df[i] / dmax));
  GaussRule rule{std::vector<double>(deg), std::vector<double>(deg)};
  double sum = 0.0;
  for (int i = 0; i < deg; ++i) {
    rule.w[i] = (w[i] + w[deg - 1 - i]) / 2;
    rule.x[i] = (x[i] - x[deg - 1 - i]) / 2;
    sum += rule.w[i];
  }
  const double factor = 2. / sum;
  for (double& wi : rule.w) wi *= factor;
  return rule;
}

// Cached as functools.lru_cache.
const GaussRule& gauss_rule(int order) {
  static std::map<int, GaussRule> cache;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(order);
  if (it == cache.end()) it = cache.emplace(order, leggauss(order)).first;
  return it->second;
}

// numpy.linspace(start, stop, num): start + k*step with the endpoint set exactly.
std::vector<double> linspace(double start, double stop, int num) {
  const double step = (stop - start) / (num - 1);
  std::vector<double> y(num);
  for (int k = 0; k < num; ++k) y[k] = k * step + start;
  y[num - 1] = stop;
  return y;
}

// GL panels in t - theta, with dyadically increasing width away from zero.
// Central width is |b|/|x'(theta)|. At b = 0 use 32 equal smooth panels.
// split = 2 bisects every panel for an independent refinement diagnostic.
struct ParameterNodes {
  std::vector<double> t, w;
};
ParameterNodes graded_nodes(double theta, double b, double speed, int order, int split = 1) {
  const auto uniform = linspace(-PI, PI, 33);
  std::vector<double> edges;
  if (b == 0) {
    edges = uniform;
  } else {
    const double width = std::min(PI, std::fabs(b) / speed);
    std::vector<double> positive{0.0, width};
    while (positive.back() < PI) positive.push_back(std::min(PI, 2 * positive.back()));
    for (size_t j = positive.size() - 1; j >= 1; --j) edges.push_back(-positive[j]);
    edges.insert(edges.end(), positive.begin(), positive.end());
  }
  // Bound the far-panel width as well: the five-lobed geometry has
  // significant variation away from the nearest point (np.unique of the union).
  edges.insert(edges.end(), uniform.begin(), uniform.end());
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  if (split == 2) {
    const size_t count = edges.size();
    for (size_t j = 0; j + 1 < count; ++j) edges.push_back(.5 * (edges[j] + edges[j + 1]));
    std::sort(edges.begin(), edges.end());
  }
  const GaussRule& rule = gauss_rule(order);
  ParameterNodes out;
  for (size_t j = 0; j + 1 < edges.size(); ++j) {
    const double mid = .5 * (edges[j + 1] + edges[j]);
    const double half = .5 * (edges[j + 1] - edges[j]);
    for (size_t k = 0; k < rule.x.size(); ++k) {
      out.t.push_back(theta + (mid + half * rule.x[k]));
      out.w.push_back(half * rule.w[k]);
    }
  }
  return out;
}

// Independent original-kernel graded GL evaluator; returns the pair and the work.
template <class T>
struct PanelResult {
  std::vector<T> sl, dl;
  long total_nodes = 0;
};

// Stokes: S f = int (-log r f/2 + r (r.f)/r^2)/(4 pi), D q = chi q0 + int r (r.(q - q0))(r.n)/(pi r^4).
PanelResult<Vec2> panel_original_stokes(const std::vector<Vec2>& targets, const std::vector<double>& theta,
                                        const std::vector<double>& b, int order = 32, int split = 1) {
  const size_t m = targets.size();
  PanelResult<Vec2> out;
  out.sl.resize(m);
  out.dl.resize(m);
  std::vector<long> counts(m);
#pragma omp parallel for schedule(dynamic, 1)
  for (long i = 0; i < static_cast<long>(m); ++i) {
    const double speed0 = stokes_starfish_geometry(theta[i]).speed;
    const ParameterNodes nodes = graded_nodes(theta[i], b[i], speed0, order, split);
    const Vec2 q0 = stokes_densities(theta[i]).q;
    Sum2 sl, dl;
    for (size_t k = 0; k < nodes.t.size(); ++k) {
      const GeomPoint g = stokes_starfish_geometry(nodes.t[k]);
      const double w = g.speed * nodes.w[k];
      const Vec2 r = g.x - targets[i];
      const double r2 = dot(r, r);
      const StokesDensity d = stokes_densities(nodes.t[k]);
      const double rf = dot(r, d.f);
      sl.add((((-.5 * std::log(r2)) * d.f + r * (rf / r2)) / (4 * PI)) * w);
      const double rq = dot(r, d.q - q0);
      const double rn = dot(r, g.normal);
      dl.add((r * (rq * rn / (PI * (r2 * r2)))) * w);
    }
    out.sl[i] = sl.value();
    out.dl[i] = dl.value() + side_indicator(b[i]) * q0;
    counts[i] = static_cast<long>(nodes.t.size());
  }
  for (long c : counts) out.total_nodes += c;
  return out;
}

// Laplace: S f = int log(r) f/(2 pi), D g = chi g0 + int (r.n)/r^2 (g - g0)/(2 pi).
PanelResult<double> panel_original_laplace(const LaplaceProblem& problem, const std::vector<Vec2>& targets,
                                           const std::vector<double>& theta, const std::vector<double>& b,
                                           int order = 32, int split = 1) {
  const size_t m = targets.size();
  PanelResult<double> out;
  out.sl.resize(m);
  out.dl.resize(m);
  std::vector<long> counts(m);
#pragma omp parallel for schedule(dynamic, 1)
  for (long i = 0; i < static_cast<long>(m); ++i) {
    const double speed0 = problem.geometry(theta[i]).speed;
    const ParameterNodes nodes = graded_nodes(theta[i], b[i], speed0, order, split);
    std::vector<double> f, g, f0, g0;
    problem.boundary_densities(nodes.t, f, g);
    problem.boundary_densities({theta[i]}, f0, g0);
    Sum sl, dl;
    for (size_t k = 0; k < nodes.t.size(); ++k) {
      const GeomPoint p = problem.geometry(nodes.t[k]);
      const double w = p.speed * nodes.w[k];
      const Vec2 r = p.x - targets[i];
      const double r2 = dot(r, r);
      sl.add(.5 * std::log(r2) * f[k] * w);
      dl.add(dot(r, p.normal) / r2 * (g[k] - g0[0]) * w);
    }
    out.sl[i] = sl.value() / (2 * PI);
    out.dl[i] = dl.value() / (2 * PI) + side_indicator(b[i]) * g0[0];
    counts[i] = static_cast<long>(nodes.t.size());
  }
  for (long c : counts) out.total_nodes += c;
  return out;
}

// Original kernels on a fine shifted uniform grid, without density subtraction.
std::pair<std::vector<Vec2>, std::vector<Vec2>> original_uniform_stokes(const std::vector<Vec2>& targets,
                                                                        int n = 65536) {
  std::vector<double> t(n);
  for (int k = 0; k < n; ++k) t[k] = 2 * PI * (k + .173) / n;
  const auto nodes = sample(stokes_starfish_geometry, t);
  std::vector<StokesDensity> d(n);
  std::vector<double> w(n);
  for (int k = 0; k < n; ++k) {
    d[k] = stokes_densities(t[k]);
    w[k] = 2 * PI * nodes[k].speed / n;
  }
  std::vector<Vec2> sl(targets.size()), dl(targets.size());
#pragma omp parallel for schedule(dynamic, 1)
  for (long i = 0; i < static_cast<long>(targets.size()); ++i) {
    Sum2 s, v;
    for (int k = 0; k < n; ++k) {
      const Vec2 r = nodes[k].x - targets[i];
      const double r2 = dot(r, r);
      const double rf = dot(r, d[k].f);
      s.add((((-.5 * std::log(r2)) * d[k].f + r * (rf / r2)) / (4 * PI)) * w[k]);
      const double rq = dot(r, d[k].q), rn = dot(r, nodes[k].normal);
      v.add((r * (rq * rn / (PI * (r2 * r2)))) * w[k]);
    }
    sl[i] = s.value();
    dl[i] = v.value();
  }
  return {sl, dl};
}

// ---------------------------------------------------------------- stages

int source_count(double delta) {
  return std::max(8192, 1 << static_cast<int>(std::ceil(std::log2(32 / delta))));
}

void ablation() {
  std::vector<Row> rows;
  Json reference_checks = Json::array();
  for (double delta : {.04, .02, .01, .005, .0025, .00125}) {
    const int n = source_count(delta);
    for (double lam : {-.5, .5}) {
      const TargetSet s = target_set(stokes_starfish_geometry, lam * delta);
      const auto ref = panel_original_stokes(s.y, s.theta, s.b, 48);
      const auto fine = panel_original_stokes(s.y, s.theta, s.b, 64, 2);
      const double ref_check = std::max(max_error(ref.sl, fine.sl), max_error(ref.dl, fine.dl));
      require(ref_check < 2e-11, "reference refinement", ref_check);
      if (delta == .01) {
        const auto uniform = original_uniform_stokes(s.y);
        const double independent =
            std::max(max_error(uniform.first, fine.sl), max_error(uniform.second, fine.dl));
        require(independent < 2e-11, "uniform reference check", independent);
        Json check = Json::object();
        check["delta"] = delta;
        check["lam"] = lam;
        check["uniform_check"] = independent;
        reference_checks.push_back(check);
      }
      for (int order : {3, 5, 7}) {
        const StokesResult value = stokes_regularized(n, s.y, s.theta, s.b, delta, order);
        const StokesResult finer = stokes_regularized(2 * n, s.y, s.theta, s.b, delta, order);
        const struct {
          const char* name;
          const std::vector<Vec2>&value, &raw, &reference, &check;
        } operators[2] = {{"single", value.sl, value.raw_sl, fine.sl, finer.sl},
                          {"double", value.dl, value.raw_dl, fine.dl, finer.dl}};
        for (const auto& op : operators)
          rows.push_back({{"operator", Str(op.name)},
                          {"p", Int(order)},
                          {"delta", delta},
                          {"lambda_signed", lam},
                          {"n_gamma", Int(n)},
                          {"target_count", Int(s.y.size())},
                          {"error_inf", max_error(op.value, op.reference)},
                          {"raw_error_inf", max_error(op.raw, op.reference)},
                          {"source_doubling_inf", max_error(op.value, op.check)},
                          {"reference_refinement_inf", ref_check}});
      }
      save("stokes_ablation.csv", rows);
      print_line("ablation " + py_repr(delta) + " " + py_repr(lam));
    }
  }
  write_json(ROOT / "reference_checks.json", reference_checks);
}

void width() {
  std::vector<Row> rows;
  const int n = 2048;
  const double hp = physical_spacing(stokes_starfish_geometry, n);
  // Targets stay fixed throughout the sweep, independently of delta.
  const TargetSet s = target_set(stokes_starfish_geometry, -.003);
  const auto ref = panel_original_stokes(s.y, s.theta, s.b, 64, 2);
  for (int order : {3, 5, 7}) {
    for (double width_ratio : {.35, .5, .75, 1., 1.5, 2., 3., 4., 6., 8.}) {
      const double delta = width_ratio * hp;
      const StokesResult value = stokes_regularized(n, s.y, s.theta, s.b, delta, order);
      const StokesResult fine = stokes_regularized(2 * n, s.y, s.theta, s.b, delta, order);
      const struct {
        const char* name;
        const std::vector<Vec2>&value, &fine, &reference;
      } operators[2] = {{"single", value.sl, fine.sl, ref.sl}, {"double", value.dl, fine.dl, ref.dl}};
      for (const auto& op : operators)
        rows.push_back({{"operator", Str(op.name)},
                        {"p", Int(order)},
                        {"n_gamma", Int(n)},
                        {"h_phys_max", hp},
                        {"delta", delta},
                        {"delta_over_h_phys", width_ratio},
                        {"b", -.003},
                        {"target_count", Int(s.y.size())},
                        {"error_inf", max_error(op.value, op.reference)},
                        {"source_doubling_inf", max_error(op.value, op.fine)}});
      save("width_sweep.csv", rows);
    }
    print_line("width " + std::to_string(order));
  }
}

// Seconds of the per-target coefficient evaluations made by single_layer and
// double_layer inside eval_laplace_regularized (two per target), replayed and timed
// separately because the library calls are not instrumented.
double laplace_coefficient_seconds(const std::vector<double>& b, double delta, int order) {
  double seconds = 0.0;
  for (double bb : b)
    for (int layer = 0; layer < 2; ++layer) {
      const double start = perf_counter();
      const Coefficients cs = coefficients(order, std::fabs(bb) / delta);
      seconds += perf_counter() - start;
      (void)cs;
    }
  return seconds;
}

void benchmark() {
  std::vector<Row> rows;
  const int repeats = 3;
  for (const std::string name : {"laplace", "stokes"}) {
    const bool stokes = name == "stokes";
    const LaplaceProblem& laplace = laplace_kite();
    const Geometry& geometry = stokes ? stokes_starfish().geometry : laplace.geometry;
    const InsideFn& inside = stokes ? stokes_starfish().inside : laplace.inside;
    const TargetSet s1 = target_set(geometry, -.001, 25), s2 = target_set(geometry, .001, 24);
    const auto theta = concat(s1.theta, s2.theta), b = concat(s1.b, s2.b);
    const auto targets = concat(s1.y, s2.y);
    const long count = static_cast<long>(targets.size());
    std::vector<double> reference_laplace;
    std::vector<Vec2> reference_stokes;
    if (!stokes) {
      for (const Vec2& y : s1.y) reference_laplace.push_back(laplace.exact(y, Side::Inside));
      for (const Vec2& y : s2.y) reference_laplace.push_back(laplace.exact(y, Side::Outside));
    } else {
      const auto ref = panel_original_stokes(targets, theta, b, 64, 2);
      reference_stokes = add(ref.sl, ref.dl);
    }
    for (int n : {256, 512, 1024, 2048, 4096, 8192, 16384}) {
      const double hp = physical_spacing(geometry, n);
      const double delta = 2 * hp;
      for (int order : {3, 5, 7}) {
        std::vector<Timing> times;
        std::vector<double> value_laplace;
        std::vector<Vec2> value_stokes;
        // Complete independent repeats: include target projection, density
        // samples, FFT differentiation/interpolation and coefficients.
        for (int repeat = 0; repeat < repeats; ++repeat) {
          SingleThread single;
          const double start = perf_counter();
          const ClosestPoints cp = closest_points(targets, geometry, inside);
          const double projection_seconds = perf_counter() - start;
          double prepare_seconds = 0., local_seconds = 0., kernel_seconds = 0., coefficient_seconds = 0.;
          if (stokes) {
            const StokesResult r = stokes_regularized(n, targets, cp.t, cp.b, delta, order);
            value_stokes = add(r.sl, r.dl);
            prepare_seconds = r.prepare_seconds;
            local_seconds = r.local_seconds;
            kernel_seconds = r.kernel_seconds;
            coefficient_seconds = r.coefficient_seconds;
          } else {
            const double prepare_start = perf_counter();
            const auto t = periodic_nodes(n);
            std::vector<double> f, g;
            laplace.boundary_densities(t, f, g);
            const PeriodicSamples fi(f), gi(g);
            prepare_seconds = perf_counter() - prepare_start;
            const double kernel_start = perf_counter();
            value_laplace = eval_laplace_regularized(geometry, fi, gi, targets, cp.t, cp.b, delta, n, order);
            local_seconds = 0.;
            kernel_seconds = perf_counter() - kernel_start;
          }
          const double total_seconds = perf_counter() - start;
          if (!stokes) coefficient_seconds = laplace_coefficient_seconds(cp.b, delta, order);
          times.push_back({{"total_seconds", total_seconds},
                           {"projection_seconds", projection_seconds},
                           {"prepare_seconds", prepare_seconds},
                           {"local_seconds", local_seconds},
                           {"kernel_seconds", kernel_seconds},
                           {"coefficient_seconds", coefficient_seconds}});
        }
        const double error = stokes ? max_error(value_stokes, reference_stokes)
                                    : max_error(value_laplace, reference_laplace);
        Row row{{"problem", Str(name)},
                {"method", Str("E1-p" + std::to_string(order))},
                {"n_gamma", Int(n)},
                {"total_kernel_nodes", Int(static_cast<long long>(n) * count)},
                {"nodes_per_target", Int(n)},
                {"delta", delta},
                {"delta_over_h_phys", 2.},
                {"target_count", Int(count)},
                {"error_inf", error},
                {"repetitions", Int(repeats)}};
        for (const auto& kv : median_timing(times)) row.emplace_back(kv.first, kv.second);
        rows.push_back(row);
      }
      save("accuracy_work.csv", rows);
      print_line("benchmark " + name + " " + std::to_string(n));
    }
    for (int gl_order : {4, 6, 8, 12, 16, 24, 32}) {
      std::vector<Timing> timings;
      PanelResult<Vec2> panel_stokes;
      PanelResult<double> panel_laplace;
      for (int repeat = 0; repeat < repeats; ++repeat) {
        SingleThread single;
        const double start = perf_counter();
        const ClosestPoints cp = closest_points(targets, geometry, inside);
        const double projection_seconds = perf_counter() - start;
        const double kernel_start = perf_counter();
        if (stokes) panel_stokes = panel_original_stokes(targets, cp.t, cp.b, gl_order);
        else panel_laplace = panel_original_laplace(laplace, targets, cp.t, cp.b, gl_order);
        const double total_seconds = perf_counter() - start;
        const double kernel_seconds = perf_counter() - kernel_start;
        timings.push_back({{"total_seconds", total_seconds},
                           {"projection_seconds", projection_seconds},
                           {"prepare_seconds", 0.},
                           {"local_seconds", 0.},
                           {"coefficient_seconds", 0.},
                           {"kernel_seconds", kernel_seconds}});
      }
      const long nodes = stokes ? panel_stokes.total_nodes : panel_laplace.total_nodes;
      const double error = stokes ? max_error(add(panel_stokes.sl, panel_stokes.dl), reference_stokes)
                                  : max_error(add(panel_laplace.sl, panel_laplace.dl), reference_laplace);
      Row row{{"problem", Str(name)},
              {"method", Str("graded-GL-" + std::to_string(gl_order))},
              {"n_gamma", Int(0)},
              {"total_kernel_nodes", Int(nodes)},
              {"nodes_per_target", static_cast<double>(nodes) / count},
              {"delta", 0.},
              {"delta_over_h_phys", 0.},
              {"target_count", Int(count)},
              {"error_inf", error},
              {"repetitions", Int(repeats)}};
      for (const auto& kv : median_timing(timings)) row.emplace_back(kv.first, kv.second);
      rows.push_back(row);
    }
    save("accuracy_work.csv", rows);
  }
}

// Signed offsets b = lambda delta -> 0 against the on-curve values (the DLP plus its jump).
void limits() {
  std::vector<Row> rows;
  const int n = 8192;
  const double delta = .01;
  const TargetSet on = target_set(stokes_starfish_geometry, 0., 33);
  const auto frames = sample(stokes_starfish_geometry, on.theta);
  std::vector<Vec2> q0;
  for (double t : on.theta) q0.push_back(stokes_densities(t).q);
  for (int order : {3, 5, 7}) {
    const StokesResult surface = stokes_regularized(n, on.y, on.theta, on.b, delta, order);
    for (double lam_abs : {.5, .01, 1e-4, 1e-6, 1e-8, 1e-10}) {
      for (int sign : {-1, 1}) {
        const std::vector<double> b(on.theta.size(), sign * lam_abs * delta);
        std::vector<Vec2> targets(on.y.size()), dl_limit(on.y.size());
        for (size_t i = 0; i < targets.size(); ++i) {
          targets[i] = on.y[i] + b[i] * frames[i].normal;
          dl_limit[i] = surface.dl[i] + (-sign * .5) * q0[i];
        }
        const StokesResult r = stokes_regularized(n, targets, on.theta, b, delta, order);
        require(all_finite(r.sl) && all_finite(r.dl), "nonfinite tiny-lambda value", sign * lam_abs);
        rows.push_back({{"operator", Str("single")},
                        {"p", Int(order)},
                        {"lambda_signed", sign * lam_abs},
                        {"delta", delta},
                        {"n_gamma", Int(n)},
                        {"trace_difference_inf", max_error(r.sl, surface.sl)}});
        rows.push_back({{"operator", Str("double")},
                        {"p", Int(order)},
                        {"lambda_signed", sign * lam_abs},
                        {"delta", delta},
                        {"n_gamma", Int(n)},
                        {"trace_difference_inf", max_error(r.dl, dl_limit)}});
      }
    }
  }
  save("tiny_lambda.csv", rows);
}

// True manufactured-field errors at very small nonzero signed offsets.
void manufactured_limits() {
  const int n = 8192, order = 7;
  const double delta = .01;
  std::vector<Row> rows;
  std::vector<double> offsets{0.};
  for (double lam : {.5, .01, 1e-4, 1e-6, 1e-8, 1e-10}) {
    offsets.push_back(-lam);
    offsets.push_back(lam);
  }
  auto add_row = [&](const std::string& geometry, const char* field, double lam, long count, double error,
                     double doubling) {
    rows.push_back({{"geometry", Str(geometry)},
                    {"field", Str(field)},
                    {"p", Int(order)},
                    {"n_gamma", Int(n)},
                    {"delta", delta},
                    {"lambda_signed", lam},
                    {"target_count", Int(count)},
                    {"error_inf", error},
                    {"source_doubling_inf", doubling}});
  };
  // Offset targets x(theta) + lambda delta n(theta) and the matching branch of the exact field.
  auto offset_targets = [&](const TargetSet& on, const std::vector<GeomPoint>& frames, double lam,
                            std::vector<double>& b, std::vector<Vec2>& targets, Side& side) {
    b.assign(on.theta.size(), lam * delta);
    targets.resize(on.y.size());
    for (size_t i = 0; i < targets.size(); ++i) targets[i] = on.y[i] + b[i] * frames[i].normal;
    side = lam < 0 ? Side::Inside : (lam > 0 ? Side::Outside : Side::Average);
  };

  {
    const LaplaceProblem& problem = laplace_kite();
    const TargetSet on = target_set(problem.geometry, 0., 33);
    const auto frames = sample(problem.geometry, on.theta);
    for (double lam : offsets) {
      std::vector<double> b;
      std::vector<Vec2> targets;
      Side side;
      offset_targets(on, frames, lam, b, targets, side);
      std::vector<double> values[2];
      for (int level = 0; level < 2; ++level) {
        const int sources = (level + 1) * n;
        std::vector<double> f, g;
        problem.boundary_densities(periodic_nodes(sources), f, g);
        values[level] = eval_laplace_regularized(problem.geometry, PeriodicSamples(f), PeriodicSamples(g),
                                                 targets, on.theta, b, delta, sources, order);
      }
      std::vector<double> exact;
      for (const Vec2& y : targets) exact.push_back(problem.exact(y, side));
      require(all_finite(values[0]), "nonfinite Laplace value", lam);
      add_row(problem.name, "laplace", lam, static_cast<long>(targets.size()), max_error(values[0], exact),
              max_error(values[0], values[1]));
    }
    print_line("manufactured small-lambda " + problem.name);
    save("manufactured_tiny_lambda.csv", rows);
  }
  {
    const StokesProblem& problem = stokes_starfish();
    const TargetSet on = target_set(problem.geometry, 0., 33);
    const auto frames = sample(problem.geometry, on.theta);
    for (double lam : offsets) {
      std::vector<double> b;
      std::vector<Vec2> targets;
      Side side;
      offset_targets(on, frames, lam, b, targets, side);
      std::vector<Vec2> velocity[2];
      std::vector<double> pressure[2];
      for (int level = 0; level < 2; ++level) {
        std::vector<Vec2> q, f;
        problem.boundary_data(periodic_nodes((level + 1) * n), problem.mu, q, f);
        velocity[level] = eval_stokes_regularized(problem.geometry, q, f, targets, on.theta, b, delta, order);
        pressure[level] = eval_pressure_regularized(problem.geometry, q, f, targets, on.theta, b, delta, order);
      }
      std::vector<Vec2> exact_u;
      std::vector<double> exact_p;
      for (const Vec2& y : targets) {
        const StokesSolution e = problem.exact(y, side, problem.mu);
        exact_u.push_back(e.u);
        exact_p.push_back(e.p);
      }
      require(all_finite(velocity[0]), "nonfinite velocity", lam);
      add_row(problem.name, "velocity", lam, static_cast<long>(targets.size()), max_error(velocity[0], exact_u),
              max_error(velocity[0], velocity[1]));
      require(all_finite(pressure[0]), "nonfinite pressure", lam);
      add_row(problem.name, "pressure", lam, static_cast<long>(targets.size()), max_error(pressure[0], exact_p),
              max_error(pressure[0], pressure[1]));
    }
    print_line("manufactured small-lambda " + problem.name);
    save("manufactured_tiny_lambda.csv", rows);
  }
}

std::string cpu_model() {
  utsname u{};
  uname(&u);
  std::string model = u.machine;  // platform.processor() fallback
  std::ifstream cpuinfo("/proc/cpuinfo");
  for (std::string line; std::getline(cpuinfo, line);)
    if (line.rfind("model name", 0) == 0) {
      std::string value = line.substr(line.find(':') + 1);
      const auto first = value.find_first_not_of(" \t"), last = value.find_last_not_of(" \t\r");
      model = first == std::string::npos ? "" : value.substr(first, last - first + 1);
      break;
    }
  return model;
}

// The separate SLP/DLP evaluation must agree with the library Stokes evaluator.
void validate() {
  const TargetSet s = target_set(stokes_starfish_geometry, -.01, 17);
  const auto t = periodic_nodes(1024);
  std::vector<Vec2> q(t.size()), f(t.size());
  for (size_t k = 0; k < t.size(); ++k) {
    q[k] = stokes_densities(t[k]).q;
    f[k] = stokes_densities(t[k]).f;
  }
  const StokesResult separate = stokes_regularized(1024, s.y, s.theta, s.b, .03, 7);
  const auto current = eval_stokes_regularized(stokes_starfish_geometry, q, f, s.y, s.theta, s.b, .03, 7);
  const double discrepancy = max_error(add(separate.sl, separate.dl), current);
  require(discrepancy < 1e-13, "evaluator discrepancy", discrepancy);
  utsname u{};
  uname(&u);
  Json metadata = Json::object();
  // The Python version keys hold the C++ toolchain: compiler, Eigen, FFTW.
  metadata["python"] = std::string("C++17 ") + __VERSION__;
  metadata["numpy"] = "Eigen " + std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) +
                      "." + std::to_string(EIGEN_MINOR_VERSION);
  metadata["scipy"] = std::string(fftw_version);
  metadata["platform"] = std::string(u.sysname) + "-" + u.release + "-" + u.machine;
  metadata["cpu_model"] = cpu_model();
  metadata["curve_names"] = Json(Json::Array{Json(laplace_kite().name), Json(stokes_starfish().name)});
  metadata["existing_evaluator_discrepancy"] = discrepancy;
  metadata["note"] =
      "Single-thread OpenMP in timed sections. Timing medians over three complete evaluations, no cached "
      "projection.";
  write_json(ROOT / "metadata.json", metadata);
}

// Finite-range rates and fastest tested tolerance configurations.
void summarize() {
  const CsvTable ablation = read_csv(ROOT / "stokes_ablation.csv");
  auto num = [](const std::map<std::string, std::string>& row, const char* key) { return std::stod(row.at(key)); };
  std::vector<Row> rates;
  Json rates_json = Json::array();
  for (double lam : {-.5, .5}) {
    for (int order : {3, 5, 7}) {
      for (const std::string op : {"single", "double"}) {
        std::vector<const std::map<std::string, std::string>*> resolved;
        for (const auto& row : ablation.rows)
          if (std::stoi(row.at("p")) == order && row.at("operator") == op && num(row, "lambda_signed") == lam &&
              num(row, "error_inf") >
                  20 * std::max(num(row, "source_doubling_inf"), num(row, "reference_refinement_inf")))
            resolved.push_back(&row);
        if (resolved.size() < 2) throw std::runtime_error("fewer than two resolved ablation rows");
        const auto& coarse = *resolved[resolved.size() - 2];
        const auto& fine = *resolved.back();
        rates.push_back({{"operator", Str(op)},
                         {"p", Int(order)},
                         {"lambda_signed", lam},
                         {"delta_coarse", num(coarse, "delta")},
                         {"delta_fine", num(fine, "delta")},
                         {"rate", std::log2(num(coarse, "error_inf") / num(fine, "error_inf"))},
                         {"raw_rate", std::log2(num(coarse, "raw_error_inf") / num(fine, "raw_error_inf"))},
                         {"error_fine", num(fine, "error_inf")}});
        rates_json.push_back(to_json(rates.back()));
      }
    }
  }
  save("ablation_rates.csv", rates);

  // Fastest configuration of each family reaching the tolerance; CSV values stay strings.
  const CsvTable work = read_csv(ROOT / "accuracy_work.csv");
  std::vector<Row> threshold;
  Json threshold_json = Json::array();
  for (const std::string problem : {"laplace", "stokes"}) {
    for (double tolerance : {1e-6, 1e-9, 1e-12}) {
      for (const std::string family : {"E1", "graded"}) {
        const std::map<std::string, std::string>* best = nullptr;
        for (const auto& row : work.rows)
          if (row.at("problem") == problem && row.at("method").rfind(family, 0) == 0 &&
              num(row, "error_inf") <= tolerance &&
              (!best || num(row, "total_seconds") < num(*best, "total_seconds")))
            best = &row;
        if (!best) continue;
        Row row{{"tolerance", tolerance}};
        for (const auto& field : work.fields) row.emplace_back(field, Str(best->at(field)));
        threshold.push_back(row);
        threshold_json.push_back(to_json(row));
      }
    }
  }
  save("threshold.csv", threshold);

  double reference_max = -HUGE_VAL, quadrature_max = -HUGE_VAL;
  for (const auto& row : ablation.rows) {
    reference_max = std::max(reference_max, num(row, "reference_refinement_inf"));
    quadrature_max = std::max(quadrature_max, num(row, "source_doubling_inf"));
  }
  Json summary = Json::object();
  summary["ablation_rates"] = rates_json;
  summary["threshold"] = threshold_json;
  summary["reference_refinement_max"] = reference_max;
  summary["quadrature_difference_max"] = quadrature_max;
  write_json(ROOT / "summary.json", summary);
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> choices{"all", "ablation", "width", "benchmark", "limits",
                                         "manufactured_limits", "summary"};
  std::string stage = "all";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--stage" && i + 1 < argc) stage = argv[++i];
    else if (arg.rfind("--stage=", 0) == 0) stage = arg.substr(8);
    else if (arg == "--output-root" && i + 1 < argc) ++i;
    else {
      std::fprintf(stderr, "usage: paper_revision [--stage STAGE] [--output-root DIR]\n");
      return 2;
    }
  }
  if (std::find(choices.begin(), choices.end(), stage) == choices.end()) {
    std::fprintf(stderr, "invalid --stage %s (the plot stage stays in Python)\n", stage.c_str());
    return 2;
  }
  ROOT = output_path(output_root(argc, argv), "results/paper_revision");
  fs::create_directories(ROOT);
  validate();
  const std::vector<std::pair<std::string, void (*)()>> stages{
      {"ablation", ablation}, {"width", width},         {"benchmark", benchmark},
      {"limits", limits},     {"manufactured_limits", manufactured_limits}, {"summary", summarize}};
  for (const auto& [name, function] : stages)
    if (stage == name || stage == "all") function();
  return 0;
}
