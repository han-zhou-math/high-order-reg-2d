// Field evaluation of the fixed-width diagnostics (experiments/fixed_width.evaluate),
// with the p = 7 kernels at prescribed closest parameters theta and signed distances b.
#pragma once
#include <string>
#include <vector>

#include "regbie/types.hpp"

namespace regbie {

struct FieldValues {
  std::vector<double> laplace, pressure;
  std::vector<Vec2> velocity;
};

// label "kite_laplace" fills laplace (regbie::laplace_kite()); "starfish_stokes" fills
// pressure and velocity (regbie::stokes_starfish(), mu = 1). The densities are sampled
// at the n nodes 2 pi k/n. side is accepted for the Python signature and not used,
// as in fixed_width.evaluate.
FieldValues fixed_width_evaluate(const std::string& label, const std::vector<Vec2>& targets,
                                 const std::vector<double>& theta, const std::vector<double>& b, double delta, int n,
                                 const std::string& side);

// label "circle" of fixed_width.evaluate: single and double layers of the circle
// density 1 + 0.4 cos 3t + 0.3 sin 5t on the unit circle.
struct CircleLayerValues {
  std::vector<double> single, double_layer;
};
CircleLayerValues fixed_width_evaluate_circle(const std::vector<Vec2>& targets, const std::vector<double>& theta,
                                              const std::vector<double>& b, double delta, int n);

}  // namespace regbie
