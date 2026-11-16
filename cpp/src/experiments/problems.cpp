#include "regbie/experiments/problems.hpp"

#include <complex>
#include <memory>

namespace regbie {

using cplx = std::complex<double>;
namespace {
const cplx I(0.0, 1.0);
cplx to_complex(Vec2 p) { return {p.x, p.y}; }

// Laplace: F- is analytic inside the kite and F+ is analytic outside it.
const cplx LM_A(2.4, 1.1), LM_B(-2.5, 0.8);
const cplx LP_A(0.08, 0.10), LP_B(-0.22, -0.08);

cplx laplace_fminus(cplx z) {
  return 0.12 + 0.20 * z + cplx(0.07, -0.04) * z * z + 0.10 / (z - LM_A) - cplx(0, 0.08) / (z - LM_B);
}
cplx laplace_dfminus(cplx z) {
  return 0.20 + 2.0 * cplx(0.07, -0.04) * z - 0.10 / ((z - LM_A) * (z - LM_A)) +
         cplx(0, 0.08) / ((z - LM_B) * (z - LM_B));
}
cplx laplace_fplus(cplx z) { return cplx(0.16, 0.03) / (z - LP_A) - cplx(0.07, -0.04) / (z - LP_B); }
cplx laplace_dfplus(cplx z) {
  return -cplx(0.16, 0.03) / ((z - LP_A) * (z - LP_A)) + cplx(0.07, -0.04) / ((z - LP_B) * (z - LP_B));
}
// grad Re F = (Re F', -Im F').
Vec2 harmonic_gradient(cplx derivative) { return {derivative.real(), -derivative.imag()}; }

// Stokes: Goursat functions of degree at most two plus exterior poles.
const cplx SF_A(2.5, 1.0), SF_B(-2.4, 1.2);
const cplx SG_A(2.3, -1.4), SG_B(-2.6, -0.9);

cplx stokes_f(cplx z) {
  return 0.04 + cplx(0.055, 0.025) * z + cplx(0.018, -0.012) * z * z + 0.035 / (z - SF_A) -
         cplx(0, 0.025) / (z - SF_B);
}
cplx stokes_df(cplx z) {
  return cplx(0.055, 0.025) + 2.0 * cplx(0.018, -0.012) * z - 0.035 / ((z - SF_A) * (z - SF_A)) +
         cplx(0, 0.025) / ((z - SF_B) * (z - SF_B));
}
cplx stokes_ddf(cplx z) {
  return 2.0 * cplx(0.018, -0.012) + 0.070 / ((z - SF_A) * (z - SF_A) * (z - SF_A)) -
         cplx(0, 0.050) / ((z - SF_B) * (z - SF_B) * (z - SF_B));
}
cplx stokes_dg(cplx z) {
  return cplx(0.045, -0.020) + 2.0 * cplx(0.014, 0.010) * z - cplx(0, 0.030) / ((z - SG_A) * (z - SG_A)) +
         0.022 / ((z - SG_B) * (z - SG_B));
}
cplx stokes_ddg(cplx z) {
  return 2.0 * cplx(0.014, 0.010) + cplx(0, 0.060) / ((z - SG_A) * (z - SG_A) * (z - SG_A)) -
         0.044 / ((z - SG_B) * (z - SG_B) * (z - SG_B));
}
}  // namespace

double kite_scale() {
  static const double s = coordinate_scale(stokes_kite_geometry);
  return s;
}
double starfish_scale() {
  static const double s = coordinate_scale(laplace_starfish_geometry);
  return s;
}

GeomPoint laplace_kite_geometry(double t) {
  static const Geometry g = scaled_geometry(stokes_kite_geometry, kite_scale());
  return g(t);
}
GeomPoint stokes_starfish_geometry(double t) {
  static const Geometry g = scaled_geometry(laplace_starfish_geometry, starfish_scale());
  return g(t);
}

Vec2 stokes_velocity(Vec2 p) {
  const cplx z = to_complex(p);
  const cplx a = stokes_f(z), b = stokes_df(z), q = stokes_dg(z);
  const double psi_x = (a + std::conj(z) * b + q).real();
  const double psi_y = (-I * a + I * std::conj(z) * b + I * q).real();
  return {psi_y, -psi_x};
}

std::array<std::array<double, 2>, 2> stokes_grad_velocity(Vec2 p) {
  const cplx z = to_complex(p);
  const cplx b = stokes_df(z), c = stokes_ddf(z), q = stokes_ddg(z);
  const double psi_xx = (2.0 * b + std::conj(z) * c + q).real();
  const double psi_xy = (I * (std::conj(z) * c + q)).real();
  const double psi_yy = (2.0 * b - std::conj(z) * c - q).real();
  return {{{psi_xy, psi_yy}, {-psi_xx, -psi_xy}}};
}

double stokes_pressure(Vec2 p, double mu) { return -4 * mu * stokes_df(to_complex(p)).imag(); }

Vec2 stokes_grad_pressure(Vec2 p, double mu) {
  const cplx v = stokes_ddf(to_complex(p));
  return {-4 * mu * v.imag(), -4 * mu * v.real()};
}

const LaplaceProblem& laplace_kite() {
  static const LaplaceProblem problem = [] {
    LaplaceProblem pr;
    pr.name = "scaled_kite_low_order_r0895";
    pr.geometry = laplace_kite_geometry;
    auto inside = std::make_shared<CurveInside>(pr.geometry);
    pr.inside = [inside](Vec2 p) { return (*inside)(p); };
    pr.boundary_densities = [](const std::vector<double>& t, std::vector<double>& f, std::vector<double>& g) {
      f.resize(t.size());
      g.resize(t.size());
      for (size_t k = 0; k < t.size(); ++k) {
        const GeomPoint gp = laplace_kite_geometry(t[k]);
        const cplx z = to_complex(gp.x);
        const Vec2 grad_minus = harmonic_gradient(laplace_dfminus(z));
        const Vec2 grad_plus = harmonic_gradient(laplace_dfplus(z));
        f[k] = dot(grad_plus - grad_minus, gp.normal);
        g[k] = laplace_fminus(z).real() - laplace_fplus(z).real();
      }
    };
    pr.exact = [inside](Vec2 p, Side side) {
      const cplx z = to_complex(p);
      const double minus = laplace_fminus(z).real(), plus = laplace_fplus(z).real();
      switch (side) {
        case Side::Inside: return minus;
        case Side::Outside: return plus;
        case Side::Average: return 0.5 * (minus + plus);
        default: return (*inside)(p) ? minus : plus;
      }
    };
    pr.box = {-1.5, 1.5, -1.5, 1.5};
    return pr;
  }();
  return problem;
}

const StokesProblem& stokes_starfish() {
  static const StokesProblem problem = [] {
    StokesProblem pr;
    pr.name = "scaled_starfish_low_order_r0895";
    pr.geometry = stokes_starfish_geometry;
    auto inside = std::make_shared<CurveInside>(pr.geometry);
    pr.inside = [inside](Vec2 p) { return (*inside)(p); };
    pr.boundary_data = [](const std::vector<double>& t, double mu, std::vector<Vec2>& q, std::vector<Vec2>& f) {
      q.resize(t.size());
      f.resize(t.size());
      for (size_t k = 0; k < t.size(); ++k) {
        const GeomPoint gp = stokes_starfish_geometry(t[k]);
        q[k] = stokes_velocity(gp.x);
        const auto grad = stokes_grad_velocity(gp.x);
        const double p = stokes_pressure(gp.x, mu);
        // sigma = -p I + mu (grad u + grad u^T); f = sigma n.
        double sigma[2][2];
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j) sigma[i][j] = (i == j ? -p : 0.0) + mu * (grad[i][j] + grad[j][i]);
        f[k] = {sigma[0][0] * gp.normal.x + sigma[0][1] * gp.normal.y,
                sigma[1][0] * gp.normal.x + sigma[1][1] * gp.normal.y};
      }
    };
    pr.exact = [inside](Vec2 p, Side side, double mu) {
      const Vec2 u = stokes_velocity(p);
      const double pr_ = stokes_pressure(p, mu);
      switch (side) {
        case Side::Inside: return StokesSolution{u, pr_};
        case Side::Outside: return StokesSolution{{0.0, 0.0}, 0.0};
        case Side::Average: return StokesSolution{0.5 * u, 0.5 * pr_};
        default: return (*inside)(p) ? StokesSolution{u, pr_} : StokesSolution{{0.0, 0.0}, 0.0};
      }
    };
    pr.exact_gradient = [inside](Vec2 p) { return (*inside)(p) ? stokes_grad_pressure(p) : Vec2{0.0, 0.0}; };
    pr.box = {-1.5, 1.5, -1.5, 1.5};
    return pr;
  }();
  return problem;
}

}  // namespace regbie
