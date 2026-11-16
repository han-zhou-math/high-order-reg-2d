// Cartesian-grid tools: stencils, nine-point Laplacian, fourth-order gradient and
// the fast sine-transform Dirichlet solver (all_grid_reconstruction.py, allgrid.py).
#pragma once
#include <array>
#include <vector>

#include "regbie/geometry.hpp"
#include "regbie/types.hpp"

namespace regbie {

using Offsets = std::vector<std::array<int, 2>>;  // (di, dj): di along x (columns), dj along y (rows)
extern const Offsets LAPLACE_OFFSETS;   // nine-point stencil
extern const Offsets GRADIENT_OFFSETS;  // fourth-order gradient stencil

using Box = std::array<double, 4>;  // xmin, xmax, ymin, ymax

// (m+2) x (m+2) grid on a square box with spacing h = (xmax - xmin)/(m+1).
struct GridCoords {
  Grid xx, yy;
  double h = 0;
  Vec2 point(int row, int col) const { return {xx(row, col), yy(row, col)}; }
};
GridCoords grid_coordinates(int m, const Box& box);

// Interior centers whose stencil, sampled at fractions 1/4, 1/2, 3/4, 1 of each
// offset, meets the other phase (allgrid.classify_crossing). Shape m x m.
Mask classify_crossing(const GridCoords& grid, const Offsets& offsets, const InsideFn& inside);

// Full-grid mask of all stencil nodes of the masked interior centers.
Mask stencil_halo(const Mask& mask, const Offsets& offsets = LAPLACE_OFFSETS);

// Nine-point Laplacian of full-grid values at the interior nodes (m x m).
Grid nine_point_laplacian(const Grid& full, double h);

// Fourth-order gradient at the interior nodes, one-sided near the outer boundary.
struct Gradient {
  Grid gx, gy;
};
Gradient grad4(const Grid& full, double h);

// Solve the nine-point Dirichlet problem: rhs is m x m, boundary is (m+2) x (m+2)
// (only its outer ring is used). Returns the full grid with the boundary values.
Grid dst_solve_nine_point(const Grid& rhs, const Grid& boundary, double h);

// Mask of the outer ring of an (rows x cols) grid, and its points in row-major order
// (the order of NumPy boolean indexing).
Mask outer_ring(int rows, int cols);
std::vector<Vec2> masked_points(const GridCoords& grid, const Mask& mask);

// Row-major list of (row, col) of the true entries of a mask.
std::vector<std::array<int, 2>> mask_indices(const Mask& mask);

}  // namespace regbie
