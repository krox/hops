#pragma once
#include "util/vector.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hops {

enum class Precision : int8_t
{
	float32,
	float64
	// future: float16, bfloat16, fixed-point, integers?
	// probably not: double-double
	// definitely not: complex
};

enum class Complexity : int8_t
{
	real,
	complex
	// possible: quaternion, conjugated_complex
	// in principle, but probably too annoying: imaginary, negative_{real,imag}
};

class Type
{
	Precision prec_ = Precision::float64;
	Complexity comp_ = Complexity::real;

  public:
	// default is double-precision real (consistent with numpy)
	Type() = default;

	// implicit conversion from Precision (defaulting to real, scalar)
	Type(Precision prec) : prec_(prec) {}

	// full constructor
	Type(Precision prec, Complexity comp) : prec_(prec), comp_(comp) {}

	Precision precision() const { return prec_; }
	Complexity complexity() const { return comp_; }

	bool real() const { return comp_ == Complexity::real; }

	// future prove...
	int height() const { return 1; }
	int width() const { return 1; }
};

inline size_t real_dim(Complexity c)
{
	switch (c)
	{
	case Complexity::real:
		return 1;
	case Complexity::complex:
		return 2;
	default:
		assert(false);
	}
}

inline int real_dim(Type t)
{
	return real_dim(t.complexity()) * t.width() * t.height();
}

// number of bytes for a given precision
inline size_t bytes(Precision p)
{
	switch (p)
	{
	case Precision::float32:
		return 4;
	case Precision::float64:
		return 8;
	default:
		assert(false);
	}
}

inline size_t bytes(Type t) { return bytes(t.precision()) * real_dim(t); }

inline std::string cuda(Type t)
{
	assert(t.width() == 1 && t.height() == 1);

	if (t.complexity() == Complexity::real)
		switch (t.precision())
		{
		case Precision::float32:
			return "float";
		case Precision::float64:
			return "double";
		default:
			assert(false);
		}
	if (t.complexity() == Complexity::complex)
		switch (t.precision())
		{
		case Precision::float32:
			return "std::complex<float>";
		case Precision::float64:
			return "std::complex<double>";
		default:
			assert(false);
		}
	assert(false);
}

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