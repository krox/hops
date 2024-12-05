#pragma once
#include "util/vector.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hops {

// map nested type to its innermost type. In numerical contexts, this should be
// one of 'float' or 'double'.
template <class T> struct real_type : public ::std::type_identity<T>
{};
template <class T>
    requires requires { typename T::value_type; }
struct real_type<T> : public ::std::type_identity<typename T::value_type>
{};
template <class T> using real_type_t = typename real_type<T>::type;

// compatible with CUDA's 'dim3' type
struct dim3
{
	uint32_t x = 1, y = 1, z = 1;

	constexpr dim3() noexcept = default;
	constexpr dim3(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept
	    : x(x), y(y), z(z)
	{}
};

} // namespace hops