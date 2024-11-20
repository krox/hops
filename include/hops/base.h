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

// number of bytes for a given precision
/*inline size_t bytes(Precision p)
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
}*/

// cuda type name for a given precision
inline std::string_view cuda(Precision p)
{
	switch (p)
	{
	case Precision::float32:
		return "float";
	case Precision::float64:
		return "double";
	default:
		assert(false);
	}
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

// parameter type for kernels processing arrays in parallel.
//  * On the CUDA side, this is templated over the data type (float/double)
//  * probably gonna refactor this quite a bit eventually
class parallel
{
	void *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
	parallel(void *data, ptrdiff_t stride_x, ptrdiff_t stride_y,
	         ptrdiff_t stride_z)
	    : data_(data), stride_x_(stride_x), stride_y_(stride_y),
	      stride_z_(stride_z)
	{}
};

class const_parallel
{
	void const *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
	const_parallel(void const *data, ptrdiff_t stride_x, ptrdiff_t stride_y,
	               ptrdiff_t stride_z)
	    : data_(data), stride_x_(stride_x), stride_y_(stride_y),
	      stride_z_(stride_z)
	{}
};

} // namespace hops