#pragma once

#include "hops/base.h"
#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace hops {

// 'View' might be the most central type in the hops library. It is implemented
// in a hirarchy like this:
// View : ConstView : Layout : Cartesian

// Multi-dimensional index/size/stride type.
// Implementation notes: This type is designed with convenience in mind.
// Ideally, moving from one-dimensional to multi-dimensional code should be as
// simple as replacing 'int' with 'Index'. This implies:
//   * Implicit conversion from integers and overloaded (elem-wise) arithmetic.
//   * Internal type is signed 64-bit to cover all usecases (sizes and strides
//     and indices) with a single type.
//   * No memory allocation so that 'Index' can reasonably be passed by value.
//     At the same time large enough 'max_rank' to cover quite complex
//     situations (case in point: between lattice/lorentz/spin/color indices,
//     some lattice QCD objects could conceivably have ~10 dimennsions).
//   * This type is probably not cheap enough to be used in hot inner loops. In
//     particular inside kernels one should use 'dim3' instead.

static constexpr int max_ndim = 7; // feel free to increase if needed

class Index : public util::static_vector<int64_t, max_ndim>
{
  public:
	Index() = default;

	Index(int64_t i0) noexcept : util::static_vector<int64_t, max_ndim>({i0}) {}
	Index(int64_t i0, int64_t i1) noexcept
	    : util::static_vector<int64_t, max_ndim>({i0, i1})
	{}
	Index(int64_t i0, int i1, int64_t i2) noexcept
	    : util::static_vector<int64_t, max_ndim>({i0, i1, i2})
	{}
	Index(size_t i0) noexcept : Index(int(i0)) {}
	Index(int i0) noexcept : Index(int64_t(i0)) {}

	int ndim() const { return (int)size(); }
};

inline Index &operator+=(Index &a, Index b)
{
	assert(a.ndim() == b.ndim());
	for (int i = 0; i < a.ndim(); ++i)
		a[i] += b[i];
	return a;
}
inline Index &operator-=(Index &a, Index b)
{
	assert(a.ndim() == b.ndim());
	for (int i = 0; i < a.ndim(); ++i)
		a[i] -= b[i];
	return a;
}
inline Index &operator*=(Index &a, Index b)
{
	assert(a.ndim() == b.ndim());
	for (int i = 0; i < a.ndim(); ++i)
		a[i] *= b[i];
	return a;
}
inline Index &operator/=(Index &a, Index b)
{
	assert(a.ndim() == b.ndim());
	for (int i = 0; i < a.ndim(); ++i)
		a[i] /= b[i];
	return a;
}
inline Index operator+(Index a, Index b)
{
	auto r = a;
	r += b;
	return r;
}
inline Index operator-(Index a, Index b)
{
	auto r = a;
	r -= b;
	return r;
}
inline Index operator*(Index a, Index b)
{
	auto r = a;
	r *= b;
	return r;
}
inline Index operator/(Index a, Index b)
{
	auto r = a;
	r /= b;
	return r;
}

inline bool operator==(Index a, Index b)
{
	if (a.ndim() != b.ndim())
		return false;
	for (int i = 0; i < a.ndim(); ++i)
		if (a[i] != b[i])
			return false;
	return true;
}

inline Index make_row_major_strides(Index shape)
{
	Index strides;
	strides.resize(shape.ndim());
	int64_t stride = 1;
	for (int i = shape.ndim() - 1; i >= 0; --i)
	{
		strides[i] = stride;
		stride *= shape[i];
	}
	return strides;
}

// shape+strides. Base class for View, just enough to do some index
// calculations. Think about this as an linear mapping from multiple dimensions
// to one dimension.
class Cartesian
{
	Index shape_ = 0;
	Index stride_ = 0;

  public:
	// default is dimension=1, size=0
	Cartesian() = default;

	Cartesian(Index shape, Index stride) : shape_(shape), stride_(stride)
	{
		assert(shape.ndim() == stride.ndim());
	}

	// index calculation. Theoretically the core of the 'Cartesian' class,
	// practically never used in hot loops of course, but sometimes nice for
	// debugging.
	int64_t operator()(Index index) const
	{
		assert(index.ndim() == shape_.ndim());
		int64_t offset = 0;
		for (int i = 0; i < index.ndim(); ++i)
		{
			assert(index[i] >= 0);
			assert(index[i] < shape_[i]);
			offset += index[i] * stride_[i];
		}
		return offset;
	}

	int ndim() const { return shape_.ndim(); }

	// by convention, dimensions beyond 'ndim()' have size=1, stride=0
	Index const &shape() const { return shape_; }
	Index const &stride() const { return stride_; }
	int64_t shape(int i) const
	{
		if (i < ndim())
			return shape_[i];
		else
			return 1;
	}
	int64_t stride(int i) const
	{
		if (i < ndim())
			return stride_[i];
		else
			return 0;
	}
	int64_t size() const
	{
		int64_t s = 1;
		for (int i = 0; i < shape_.ndim(); ++i)
			s *= shape_[i];
		return s;
	}

	auto step(this auto const &self, Index stepsize)
	{
		assert(stepsize.ndim() == self.ndim());
		auto ret = self;
		for (int i = 0; i < ret.ndim(); ++i)
		{
			ret.shape_[i] = (ret.shape_[i] + stepsize[i] - 1) / stepsize[i];
			ret.stride_[i] *= stepsize[i];
		}
		return ret;
	}
};

// shape+strides+type, Base class for 'View', just enough to serve as key when
// looking compiled kernels or tuning parameters.
class Layout : public Cartesian
{
	Precision prec_ = Precision::float64;

  public:
	Layout() = default;
	explicit Layout(Precision prec, Index shape, Index stride)
	    : Cartesian(shape, stride), prec_(prec)
	{}

	Precision precision() const { return prec_; }
};

// Non-owning view of a homogeneous array, typically in device memory.
// To be used as paramter type in hops arithmetic functions.
// NOTE: with run-time rank, this is not exactly a light-weight type. This can
// be mitigated by the (future) two-step plan/execute workflow.
class ConstView : public Layout
{
  protected:
	void *data_ = nullptr;

  public:
	ConstView() = default;
	explicit ConstView(void *data, Precision prec, Index shape, Index stride)
	    : Layout(prec, shape, stride), data_(data)
	{}

	explicit ConstView(float *data, Index shape, Index stride)
	    : Layout(Precision::float32, shape, stride), data_(data)
	{}
	explicit ConstView(double *data, Index shape, Index stride)
	    : Layout(Precision::float64, shape, stride), data_(data)
	{}
	explicit ConstView(float *data, size_t n)
	    : Layout(Precision::float32, Index(n), Index(1)), data_(data)
	{}
	explicit ConstView(double *data, size_t n)
	    : Layout(Precision::float64, Index(n), Index(1)), data_(data)
	{}

	void const *data() const { return data_; }
};

class View : public ConstView
{
  public:
	using ConstView::ConstView;

	void *data() const { return data_; }
};

} // namespace hops