#pragma once

#include "hops/base.h"
#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace hops {

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
	ptrdiff_t stride(int i) const { return stride_[i]; }
	ptrdiff_t shape(int i) const { return shape_[i]; }

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

	// helper functions for view->parallel conversion when calling a kernel
	parallel<T> ewise() const
	{
		assert(rank() == 1);
		return parallel<T>(data_, stride(0), 0, 0);
	}
};

} // namespace hops