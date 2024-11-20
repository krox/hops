#pragma once

#include "hops/view.h"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace hops {

// out = alpha * a * b
void mul(View out, double alpha, ConstView a, ConstView b);

// out += alpha * a * b
void add_mul(View out, double alpha, ConstView a, ConstView b);

} // namespace hops
