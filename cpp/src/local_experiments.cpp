#include "regbie/local_experiments.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace regbie {

std::vector<TargetSet> local_targets(const Geometry& geometry, double delta, const std::vector<double>& signed_lambdas,
                                     int count) {
  std::vector<double> theta(count);
  for (int i = 0; i < count; ++i) theta[i] = 2 * PI * (i + 0.37) / count;
  const auto frame0 = sample(geometry, theta);
  std::vector<TargetSet> sets;
  for (double lam : signed_lambdas) {
    TargetSet s;
    s.theta = theta;
    s.lam = lam;
    const double b = lam * delta;
    s.b.assign(count, b);
    s.targets.resize(count);
    for (int i = 0; i < count; ++i) s.targets[i] = frame0[i].x + b * frame0[i].normal;
    sets.push_back(std::move(s));
  }
  return sets;
}

std::string side_name(double lam) { return lam == 0 ? "average" : (lam < 0 ? "inside" : "outside"); }

Side side_from_name(const std::string& side) {
  if (side == "inside") return Side::Inside;
  if (side == "outside") return Side::Outside;
  if (side == "average") return Side::Average;
  return Side::Piecewise;
}

// CIRCLE_COS3 = 0.4, CIRCLE_SIN5 = 0.3
double circle_density(double t) { return 1 + 0.4 * std::cos(3 * t) + 0.3 * std::sin(5 * t); }

double exact_circle_single(double r, double theta) {
  const double safe = r > 0 ? r : 1.0;
  const double radial3 = r <= 1 ? std::pow(r, 3.0) : std::pow(safe, -3.0);
  const double radial5 = r <= 1 ? std::pow(r, 5.0) : std::pow(safe, -5.0);
  return std::log(std::max(r, 1.0)) - 0.4 / 6 * radial3 * std::cos(3 * theta) -
         0.3 / 10 * radial5 * std::sin(5 * theta);
}

double exact_circle_double(double r, double theta, const std::string& side) {
  if (side == "average") return 0.5;
  const double safe = r > 0 ? r : 1.0;
  if (r < 1) return 1 + 0.4 / 2 * std::pow(r, 3.0) * std::cos(3 * theta) + 0.3 / 2 * std::pow(r, 5.0) * std::sin(5 * theta);
  if (r > 1) return -0.4 / 2 * std::pow(safe, -3.0) * std::cos(3 * theta) - 0.3 / 2 * std::pow(safe, -5.0) * std::sin(5 * theta);
  return 0.5;
}

namespace {

using cplx = std::complex<double>;
const cplx I(0.0, 1.0);

// z^k by repeated products.
cplx ipow(cplx z, int k) {
  cplx out(1.0, 0.0);
  for (int j = 0; j < k; ++j) out *= z;
  return out;
}

// Laplace starfish fields (manufactured_laplace.Fminus, Fplus and derivatives).
const cplx FM_A1(1.65, 0.15), FM_A2(-1.45, 0.65);
const cplx FP_Z1(0.18, 0.12), FP_Z2(-0.22, 0.08), FP_Z3(0.05, -0.25);
const cplx FP_R1(0.40, 0.08), FP_R2(0.0, -0.18), FP_R3(0.08, 0.0);

cplx fminus(cplx z) {
  return 0.35 * std::exp(0.30 * z) + 0.08 * z * z * z * z + cplx(0, 0.12) * z * z * z + 0.25 / (z - FM_A1) +
         cplx(0, -0.16) / (z - FM_A2);
}
cplx dfminus(cplx z) {
  return 0.35 * 0.30 * std::exp(0.30 * z) + 4 * 0.08 * z * z * z + 3.0 * cplx(0, 0.12) * z * z -
         0.25 / ((z - FM_A1) * (z - FM_A1)) - cplx(0, -0.16) / ((z - FM_A2) * (z - FM_A2));
}
cplx fplus(cplx z) {
  return FP_R1 / (z - FP_Z1) + FP_R2 / ipow(z - FP_Z2, 2) + FP_R3 / ipow(z - FP_Z3, 3);
}
cplx dfplus(cplx z) {
  return -FP_R1 / ipow(z - FP_Z1, 2) - 2.0 * FP_R2 / ipow(z - FP_Z2, 3) - 3.0 * FP_R3 / ipow(z - FP_Z3, 4);
}

// Kite Goursat functions (manufactured_stokes.F, G and derivatives).
const cplx KF_A1(2.10, 0.40), KF_A2(-2.20, 0.50), KG_A3(0.40, 2.30), KG_A4(-0.60, -2.20);
const cplx KF_Z2(0.06, 0.04), KF_R1(0.15, 0), KF_R2(0, -0.10);
const cplx KG_Z5(0, -0.06), KG_EXP(0, -0.20), KG_R1(0.10, 0), KG_R2(0, 0.07);

cplx kite_f(cplx z) {
  return 0.10 * ipow(z, 4) + KF_Z2 * z * z + 0.12 * std::exp(0.25 * z) + KF_R1 / (z - KF_A1) + KF_R2 / (z - KF_A2);
}
cplx kite_df(cplx z) {
  return 0.40 * ipow(z, 3) + 2.0 * KF_Z2 * z + 0.12 * 0.25 * std::exp(0.25 * z) - KF_R1 / ipow(z - KF_A1, 2) -
         KF_R2 / ipow(z - KF_A2, 2);
}
cplx kite_ddf(cplx z) {
  return 1.2 * z * z + 2.0 * KF_Z2 + 0.12 * 0.0625 * std::exp(0.25 * z) + 2.0 * KF_R1 / ipow(z - KF_A1, 3) +
         2.0 * KF_R2 / ipow(z - KF_A2, 3);
}
cplx kite_dg(cplx z) {
  return 5.0 * KG_Z5 * ipow(z, 4) + 0.27 * z * z + 0.08 * KG_EXP * std::exp(KG_EXP * z) -
         KG_R1 / ipow(z - KG_A3, 2) - KG_R2 / ipow(z - KG_A4, 2);
}
cplx kite_ddg(cplx z) {
  return 20.0 * KG_Z5 * ipow(z, 3) + 0.54 * z + 0.08 * KG_EXP * KG_EXP * std::exp(KG_EXP * z) +
         2.0 * KG_R1 / ipow(z - KG_A3, 3) + 2.0 * KG_R2 / ipow(z - KG_A4, 3);
}

bool finite(double v) { return std::isfinite(v); }
bool finite(cplx v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

// validation.validate_boundary_finiteness(samples=512)
void validate_boundary_finiteness(int samples = 512) {
  const auto t = periodic_nodes(samples);
  bool ok = true;
  for (double tk : t) {
    // Laplace starfish: F-, F+ and the densities f = [du/dn], g = [u].
    const GeomPoint gl = laplace_starfish_geometry(tk);
    const cplx zl(gl.x.x, gl.x.y);
    const cplx dm = dfminus(zl), dp = dfplus(zl);
    const double lf = dot(Vec2{dp.real(), -dp.imag()} - Vec2{dm.real(), -dm.imag()}, gl.normal);
    const double lg = fminus(zl).real() - fplus(zl).real();
    ok = ok && finite(fminus(zl)) && finite(fplus(zl)) && finite(lf) && finite(lg);
    // Kite Stokes: velocity q and traction f = sigma n.
    const GeomPoint gk = stokes_kite_geometry(tk);
    const cplx z(gk.x.x, gk.x.y), zc = std::conj(z);
    const cplx a = kite_f(z), b = kite_df(z), p = kite_dg(z), c = kite_ddf(z), q = kite_ddg(z);
    const Vec2 u{(-I * a + I * zc * b + I * p).real(), -(a + zc * b + p).real()};
    const double psi_xx = (2.0 * b + zc * c + q).real(), psi_xy = (I * (zc * c + q)).real(),
                 psi_yy = (2.0 * b - zc * c - q).real();
    const double grad[2][2] = {{psi_xy, psi_yy}, {-psi_xx, -psi_xy}};
    const double pressure = -4 * b.imag();
    double sigma[2][2];
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) sigma[i][j] = (i == j ? -pressure : 0.0) + grad[i][j] + grad[j][i];
    const Vec2 f{sigma[0][0] * gk.normal.x + sigma[0][1] * gk.normal.y,
                 sigma[1][0] * gk.normal.x + sigma[1][1] * gk.normal.y};
    ok = ok && finite(u.x) && finite(u.y) && finite(f.x) && finite(f.y);
  }
  if (!ok) throw std::runtime_error("nonfinite manufactured boundary data");
}

// validation.validate_starfish_poles: F+ poles inside, F- poles outside the starfish.
void validate_starfish_poles() {
  bool ok = true;
  for (cplx z : {FP_Z1, FP_Z2, FP_Z3}) ok = ok && starfish_level({z.real(), z.imag()}) < 0;
  for (cplx z : {FM_A1, FM_A2}) ok = ok && starfish_level({z.real(), z.imag()}) > 0;
  if (!ok) throw std::runtime_error("a Laplace pole is on the wrong side of the starfish");
}

// validation.validate_kite_poles: every Goursat pole lies outside the kite.
void validate_kite_poles() {
  const CurveInside inside(stokes_kite_geometry, 4096);
  for (cplx z : {KF_A1, KF_A2, KG_A3, KG_A4})
    if (inside({z.real(), z.imag()})) throw std::runtime_error("a Goursat pole lies inside the kite");
}

}  // namespace

void validate_all() {
  validate_boundary_finiteness();
  validate_starfish_poles();
  validate_kite_poles();
}

}  // namespace regbie
