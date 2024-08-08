#pragma once

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace hops {

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
	int rank_;
};

// Non-owning view of a homogeneous array, typically in device memory.
// To be used as paramter type in hops arithmetic functions.
// NOTE: with run-time rank, this is not exactly a light-weight type. This can
// be mitigated by the (future) two-step plan/execute workflow.
template <class T> class View
{
	T *data_ = nullptr;
	size_t size_ = 0;
	Index shape_ = {};
	Index stride_ = {};

  public:
	View() = default;
	explicit View(T *data, Index shape, Index stride)
	    : data_(data), shape_(shape), stride_(stride)
	{
		assert(shape.rank() == stride.rank());
		size_ = 1;
		for (int i = 0; i < rank(); ++i)
			size_ *= shape[i];
	}

	static View contiguous(T *data, Index shape)
	{
		Index stride = shape;
		stride[shape.rank() - 1] = 1;
		for (int i = shape.rank() - 1; i >= 0; --i)
		{
			stride[i] = shape[i + 1] * stride[i + 1];
		}
		return View(data, shape, stride);
	}

	// implicit const cast
	operator View<const T>() const
	{
		return View<const T>(data_, shape_, stride_);
	}

	T *data() const { return data_; }
	size_t size() const { return size_; }
	int rank() const { return shape_.rank(); }
	Index shape() const { return shape_; }
	Index stride() const { return stride_; }

	// subset view
	View<T> step(Index stepsize) const
	{
		assert(stepsize.rank() == rank());
		Index new_shape = shape_;
		Index new_stride = stride_;
		for (int i = 0; i < rank(); ++i)
		{
			new_shape[i] = (shape_[i] + stepsize[i] - 1) / stepsize[i];
			new_stride[i] *= stepsize[i];
		}
		return View(data_, new_shape, new_stride);
	}
};

} // namespace hops