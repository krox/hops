#pragma once

#include "hops/view.h"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace hops {

// out = alpha * a * b
template <class T>
void mul(View<T> out, T alpha, View<const T> a, View<const T> b);

// out += alpha * a * b
template <class T>
void add_mul(View<T> out, T alpha, View<const T> a, View<const T> b);

} // namespace hops
