#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace hops {

// pointer + stride.
// Meant as a parameter type to make kernel function signatures a bit more
// readable.
template <std::floating_point T> struct DevicePtr
{
	T *ptr_ = nullptr;
	ptrdiff_t stride_ = 1;

	DevicePtr() = default;
	DevicePtr(T *ptr) : ptr_(ptr), stride_(1) {}
	DevicePtr(T *ptr, ptrdiff_t stride) : ptr_(ptr), stride_(stride) {}

	T &operator()(size_t i) const { return ptr_[i * stride_]; }
};

// out {=,+=} alpha * a * b
template <bool accumulate, class T>
void mul(DevicePtr<T> out, size_t n, T alpha, DevicePtr<const T> a,
         DevicePtr<const T> b);

} // namespace hops
