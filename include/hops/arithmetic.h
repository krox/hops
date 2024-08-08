#pragma once

#include "hops/view.h"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace hops {

// pointer + stride.
// Meant as a parameter type to make kernel function signatures a bit more
// readable.
template <std::floating_point T> struct DevicePtr
{
	T *ptr_ = nullptr;
	ptrdiff_t stride_ = 1;
};

// backend kernels. These are not meant to be called directly by the user.
// also, probably not the final form of the interface.
namespace kernels {

template <bool accumulate, class T>
void mul_1d(DevicePtr<T> out, size_t n, T alpha, DevicePtr<const T> a,
            DevicePtr<const T> b);
} // namespace kernels

// out {=,+=} alpha * a * b
template <bool accumulate, class T>
void mul(View<T> out, T alpha, View<const T> a, View<const T> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.shape() == a.shape());
	assert(out.shape() == b.shape());

	if (out.rank() == 1)
	{
		kernels::mul_1d<accumulate>({out.data(), out.stride()[0]}, out.size(),
		                            alpha, {a.data(), a.stride()[0]},
		                            {b.data(), b.stride()[0]});
	}
	else
		throw std::runtime_error("Not implemented yet");
}

} // namespace hops
