#pragma once

#include "hops/parallel_kernel.h"
#include "hops/view.h"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace hops {

// implementation note: the 'type_identity' wrappers are used to guide template
// deduction. We only want to deduce based on the first argument.

// out = alpha * a * b
template <class T>
void mul(View<T> out, real_type_t<T> alpha,
         View<const std::type_identity_t<T>> a,
         View<const std::type_identity_t<T>> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization before
	// dispatch to the proper kernel.

	static auto kernel = hops::ParallelKernel(
	    "auto& out, auto alpha, auto a, auto b", "out = alpha * a * b;");

	kernel.launch(out, alpha, a, b);
}

// out += alpha * a * b
template <class T>
void add_mul(View<T> out, real_type_t<T> alpha,
             View<const std::type_identity_t<T>> a,
             View<const std::type_identity_t<T>> b)
{
	static auto kernel = hops::ParallelKernel(
	    "auto&out, auto alpha, auto a, auto b", "out += alpha * a * b;");
	kernel.launch(out, alpha, a, b);
}

} // namespace hops
