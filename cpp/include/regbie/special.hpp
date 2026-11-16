// Special functions used by the kernels (replacing scipy.special).
#pragma once

namespace regbie {

// Exponential integral E1(x), x > 0 (scipy.special.exp1).
double exp1(double x);

// Scaled complementary error function exp(x^2) erfc(x), x >= 0 (scipy.special.erfcx).
double erfcx(double x);

}  // namespace regbie
