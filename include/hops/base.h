#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace hops {

// compatible with CUDA's 'dim3' type
struct dim3
{
	uint32_t x = 1, y = 1, z = 1;

	constexpr dim3() noexcept = default;
	constexpr dim3(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept
	    : x(x), y(y), z(z)
	{}
};

// essentially a 'static_vector<ptrdiff_t, max_rank>', used for
// multi-dimensional indices/shapes/strides
class Index
{
  public:
	static constexpr size_t max_rank = 7;

	Index() noexcept : rank_(0) {}
	Index(size_t i0) noexcept : rank_(1) { data_[0] = ptrdiff_t(i0); }
	Index(int i0) noexcept : rank_(1) { data_[0] = ptrdiff_t(i0); }
	Index(ptrdiff_t i0) noexcept : rank_(1) { data_[0] = i0; }
	Index(ptrdiff_t i0, ptrdiff_t i1) noexcept : rank_(2)
	{
		data_[0] = i0;
		data_[1] = i1;
	}
	Index(ptrdiff_t i0, ptrdiff_t i1, ptrdiff_t i2) noexcept : rank_(3)
	{
		data_[0] = i0;
		data_[1] = i1;
		data_[2] = i2;
	}

	ptrdiff_t operator[](size_t i) const { return data_[i]; }
	ptrdiff_t &operator[](size_t i) { return data_[i]; }
	int rank() const { return int(rank_); }

	bool operator==(const Index &other) const = default;

  private:
	std::array<ptrdiff_t, max_rank> data_ = {};
	int rank_ = 0;
};

// parameter type for kernels processing arrays in parallel.
// IMPORTANT: needs to be identical to the class in the cuda code.
// TODO: make the strides (optional?) template parameters
template <class T> class parallel
{
	T *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
	parallel(T *data, ptrdiff_t stride_x, ptrdiff_t stride_y,
	         ptrdiff_t stride_z)
	    : data_(data), stride_x_(stride_x), stride_y_(stride_y),
	      stride_z_(stride_z)
	{}
};

} // namespace hops