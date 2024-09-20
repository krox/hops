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
template <class T> class View : public Cartesian
{
	T *data_ = nullptr;

  public:
	View() = default;
	explicit View(T *data, Cartesian const &cart) : Cartesian(cart), data_(data)
	{}
	explicit View(T *data, Index shape, Index stride)
	    : Cartesian(shape, stride), data_(data)
	{}

	static View contiguous(T *data, Index shape)
	{
		return View(data, Cartesian::contiguous(shape));
	}

	// implicit const cast
	operator View<const T>() const { return View<const T>(data_, *this); }

	T *data() const { return data_; }

	// subset view
	View<T> step(Index stepsize) const
	{
		assert(stepsize.rank() == rank());
		Index new_shape = shape();
		Index new_stride = stride();
		for (int i = 0; i < rank(); ++i)
		{
			new_shape[i] = (shape(i) + stepsize[i] - 1) / stepsize[i];
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