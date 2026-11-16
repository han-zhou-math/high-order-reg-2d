// Basic types shared by the library: points in the plane and row-major grids.
#pragma once
#include <cmath>
#include <vector>

#include <Eigen/Dense>

namespace regbie {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double EULER_GAMMA = 0.577215664901532860606512090082402431;

struct Vec2 {
  double x = 0.0, y = 0.0;
};

inline Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
inline Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
inline Vec2 operator-(Vec2 a) { return {-a.x, -a.y}; }
inline Vec2 operator*(double s, Vec2 a) { return {s * a.x, s * a.y}; }
inline Vec2 operator*(Vec2 a, double s) { return {s * a.x, s * a.y}; }
inline Vec2 operator/(Vec2 a, double s) { return {a.x / s, a.y / s}; }
inline Vec2& operator+=(Vec2& a, Vec2 b) { a.x += b.x; a.y += b.y; return a; }
inline Vec2& operator-=(Vec2& a, Vec2 b) { a.x -= b.x; a.y -= b.y; return a; }
inline double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
inline double cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
// sqrt(x^2+y^2), as np.linalg.norm (not hypot), so results match NumPy to roundoff.
inline double norm(Vec2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }

// Compensated (Neumaier) summation for the quadrature sums. NumPy sums pairwise,
// with roundoff O(eps log n); a plain loop gives O(eps n), which shows up as a
// larger roundoff floor at n ~ 10^4. The compensated sum is at least as accurate.
struct Sum {
  double s = 0.0, c = 0.0;
  void add(double x) {
    const double t = s + x;
    c += (std::fabs(s) >= std::fabs(x)) ? (s - t) + x : (x - t) + s;
    s = t;
  }
  double value() const { return s + c; }
};

struct Sum2 {
  Sum x, y;
  void add(Vec2 v) { x.add(v.x); y.add(v.y); }
  Vec2 value() const { return {x.value(), y.value()}; }
};

// Grids follow np.meshgrid(x, y): row j is y_j, column i is x_i.
using Grid = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Mask = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Periodic nodes t_k = 2*pi*k/n, k=0..n-1, computed as in NumPy: (2*pi*k)/n.
inline std::vector<double> periodic_nodes(int n) {
  std::vector<double> t(n);
  for (int k = 0; k < n; ++k) t[k] = 2.0 * PI * k / n;
  return t;
}

}  // namespace regbie
