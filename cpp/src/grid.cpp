#include "regbie/grid.hpp"

#include <fftw3.h>

#include <cmath>
#include <mutex>
#include <stdexcept>

#include "regbie/periodic.hpp"

namespace regbie {

const Offsets LAPLACE_OFFSETS = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1},
                                 {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
const Offsets GRADIENT_OFFSETS = {{0, 0}, {1, 0}, {-1, 0}, {2, 0}, {-2, 0},
                                  {0, 1}, {0, -1}, {0, 2}, {0, -2}};

namespace {
// np.linspace(start, stop, num): start + i*step, with the last value set to stop.
std::vector<double> linspace(double start, double stop, int num) {
  std::vector<double> v(num);
  const double step = (stop - start) / (num - 1);
  for (int i = 0; i < num; ++i) v[i] = i * step + start;
  v[num - 1] = stop;
  return v;
}
}  // namespace

GridCoords grid_coordinates(int m, const Box& box) {
  const double xmin = box[0], xmax = box[1], ymin = box[2], ymax = box[3];
  if (std::fabs((xmax - xmin) - (ymax - ymin)) > 1e-13)
    throw std::invalid_argument("the DST implementation requires a square box");
  GridCoords g;
  g.h = (xmax - xmin) / (m + 1);
  const auto x = linspace(xmin, xmax, m + 2), y = linspace(ymin, ymax, m + 2);
  g.xx.resize(m + 2, m + 2);
  g.yy.resize(m + 2, m + 2);
  for (int j = 0; j < m + 2; ++j)
    for (int i = 0; i < m + 2; ++i) {
      g.xx(j, i) = x[i];
      g.yy(j, i) = y[j];
    }
  return g;
}

Mask classify_crossing(const GridCoords& grid, const Offsets& offsets, const InsideFn& inside) {
  const int m = static_cast<int>(grid.xx.rows()) - 2;
  const double h = grid.h;
  Mask crossing = Mask::Constant(m, m, false);
#pragma omp parallel for schedule(dynamic, 4)
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i) {
      const Vec2 center = grid.point(j + 1, i + 1);
      const bool center_inside = inside(center);
      bool cross = false;
      for (const auto& o : offsets) {
        const Vec2 step{o[0] * h, o[1] * h};
        for (double fraction : {0.25, 0.5, 0.75, 1.0}) {
          if (cross) break;
          cross = inside(center + fraction * step) != center_inside;
        }
        if (cross) break;
      }
      crossing(j, i) = cross;
    }
  return crossing;
}

Mask stencil_halo(const Mask& mask, const Offsets& offsets) {
  const int m = static_cast<int>(mask.rows()), n = static_cast<int>(mask.cols());
  Mask halo = Mask::Constant(m + 2, n + 2, false);
  for (const auto& o : offsets) {
    const int di = o[0], dj = o[1];
    for (int j = 0; j < m; ++j)
      for (int i = 0; i < n; ++i)
        if (mask(j, i)) halo(1 + dj + j, 1 + di + i) = true;
  }
  return halo;
}

Grid nine_point_laplacian(const Grid& v, double h) {
  const int rows = static_cast<int>(v.rows()) - 2, cols = static_cast<int>(v.cols()) - 2;
  Grid out(rows, cols);
  for (int j = 0; j < rows; ++j)
    for (int i = 0; i < cols; ++i) {
      const int r = j + 1, c = i + 1;
      const double axis = v(r, c + 1) + v(r, c - 1) + v(r + 1, c) + v(r - 1, c);
      const double diagonal = v(r + 1, c + 1) + v(r + 1, c - 1) + v(r - 1, c + 1) + v(r - 1, c - 1);
      out(j, i) = (4 * axis + diagonal - 20 * v(r, c)) / (6 * h * h);
    }
  return out;
}

namespace {
// Nodes and weights of the five-point first derivative at index (one-sided near the ends).
void first_derivative_weights(int index, int size, int nodes[5], double weights[5]) {
  int first;
  if (2 <= index && index <= size - 3) first = index - 2;
  else if (index < 2) first = 0;
  else first = size - 5;
  Eigen::Matrix<double, 5, 5> matrix;
  for (int k = 0; k < 5; ++k) {
    nodes[k] = first + k;
    const double offset = nodes[k] - index;
    for (int degree = 0; degree < 5; ++degree) matrix(degree, k) = std::pow(offset, degree);
  }
  Eigen::Matrix<double, 5, 1> rhs;
  rhs << 0., 1., 0., 0., 0.;
  const Eigen::Matrix<double, 5, 1> w = matrix.partialPivLu().solve(rhs);
  for (int k = 0; k < 5; ++k) weights[k] = w(k);
}
}  // namespace

Gradient grad4(const Grid& v, double h) {
  const int ny = static_cast<int>(v.rows()), nx = static_cast<int>(v.cols());
  Gradient g;
  g.gx.resize(ny - 2, nx - 2);
  g.gy.resize(ny - 2, nx - 2);
  int nodes[5];
  double w[5];
  for (int i = 1; i < nx - 1; ++i) {
    first_derivative_weights(i, nx, nodes, w);
    for (int j = 1; j < ny - 1; ++j) {
      double s = 0.0;
      for (int k = 0; k < 5; ++k) s += v(j, nodes[k]) * w[k];
      g.gx(j - 1, i - 1) = s / h;
    }
  }
  for (int j = 1; j < ny - 1; ++j) {
    first_derivative_weights(j, ny, nodes, w);
    for (int i = 1; i < nx - 1; ++i) {
      double s = 0.0;
      for (int k = 0; k < 5; ++k) s += w[k] * v(nodes[k], i);
      g.gy(j - 1, i - 1) = s / h;
    }
  }
  return g;
}

Grid dst_solve_nine_point(const Grid& rhs, const Grid& boundary, double h) {
  const int m = static_cast<int>(rhs.rows());
  if (rhs.cols() != m || boundary.rows() != m + 2 || boundary.cols() != m + 2)
    throw std::invalid_argument("incompatible square grid and boundary array");
  Grid boundary_only = boundary;
  boundary_only.block(1, 1, m, m).setZero();
  Grid adjusted = rhs - nine_point_laplacian(boundary_only, h);
  // Orthonormal DST-I in both directions = FFTW RODFT00 / sqrt(2(m+1)) per direction.
  Grid transformed(m, m), interior(m, m);
  fftw_plan forward, backward;
  {
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    forward = fftw_plan_r2r_2d(m, m, adjusted.data(), transformed.data(), FFTW_RODFT00, FFTW_RODFT00,
                               FFTW_ESTIMATE);
    backward = fftw_plan_r2r_2d(m, m, transformed.data(), interior.data(), FFTW_RODFT00, FFTW_RODFT00,
                                FFTW_ESTIMATE);
  }
  fftw_execute(forward);
  std::vector<double> c(m);
  for (int k = 0; k < m; ++k) c[k] = std::cos((k + 1) * PI / (m + 1));
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < m; ++i) {
      const double eigenvalue = (8 * (c[i] + c[j]) + 4 * c[i] * c[j] - 20) / (6 * h * h);
      transformed(j, i) /= eigenvalue;
    }
  fftw_execute(backward);
  {
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
  }
  const double scale = 1.0 / (4.0 * (m + 1) * (m + 1));
  Grid full = boundary;
  full.block(1, 1, m, m) = interior * scale;
  return full;
}

Mask outer_ring(int rows, int cols) {
  Mask mask = Mask::Constant(rows, cols, false);
  mask.row(0).setConstant(true);
  mask.row(rows - 1).setConstant(true);
  mask.col(0).setConstant(true);
  mask.col(cols - 1).setConstant(true);
  return mask;
}

std::vector<Vec2> masked_points(const GridCoords& grid, const Mask& mask) {
  std::vector<Vec2> points;
  for (int j = 0; j < mask.rows(); ++j)
    for (int i = 0; i < mask.cols(); ++i)
      if (mask(j, i)) points.push_back(grid.point(j, i));
  return points;
}

std::vector<std::array<int, 2>> mask_indices(const Mask& mask) {
  std::vector<std::array<int, 2>> idx;
  for (int j = 0; j < mask.rows(); ++j)
    for (int i = 0; i < mask.cols(); ++i)
      if (mask(j, i)) idx.push_back({j, i});
  return idx;
}

}  // namespace regbie
