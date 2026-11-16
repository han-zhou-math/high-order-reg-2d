// Umbrella header of the regbie library.
//
// User interface:  regbie::Curve (curve.hpp) and regbie::LayerPotentials (potentials.hpp).
// Lower-level routines: kernels.hpp (regularized kernels and coefficients),
// layers.hpp (trapezoidal sums with given closest points and signed distances),
// periodic.hpp, geometry.hpp, grid.hpp (Cartesian-grid tools), io.hpp.
#pragma once

#include "regbie/curve.hpp"
#include "regbie/geometry.hpp"
#include "regbie/grid.hpp"
#include "regbie/io.hpp"
#include "regbie/kernels.hpp"
#include "regbie/layers.hpp"
#include "regbie/numerics.hpp"
#include "regbie/periodic.hpp"
#include "regbie/potentials.hpp"
#include "regbie/special.hpp"
#include "regbie/types.hpp"
