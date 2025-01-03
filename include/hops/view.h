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

// shape and strides.
// This non-templated base class of 'View' is sufficient for some index
// operations, and can serve as key when looking up cached kernel-tuning
// parameters in the future.
class Layout
{
	Index shape_ = 0;
	Index stride_ = 0;

  public:
	// default is dimension=1, size=0, double-precision, real
	Layout() = default;

	// full constructor
	Layout(Index shape, Index stride) : shape_(shape), stride_(stride)
	{
		assert(shape.ndim() == stride.ndim());
	}

	// index calculation. Theoretically the core of the 'Cartesian' class,
	// practically never used in hot loops of course, but sometimes nice for
	// debugging.
	/*int64_t operator()(Index index) const
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
	}*/

	int ndim() const { return shape_.ndim(); }

	// for convenience, dimensions beyond 'ndim()' are reported as having
	// size=1 and stride=0
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
		assert(stepsize.ndim() <= self.ndim());
		auto ret = self;
		for (int i = 0; i < stepsize.ndim(); ++i)
		{
			ret.shape_[i] = (ret.shape_[i] + stepsize[i] - 1) / stepsize[i];
			ret.stride_[i] *= stepsize[i];
		}
		return ret;
	}
};

// Non-owning view of a homogeneous array, typically in device memory.
// To be used as paramter type in hops arithmetic functions.
//   * 'View' only handles the physical layout of data, i.e., mapping an
//     abstract index space to memory locations. It does not know the semantics
//     of the indices. E.g. there is no notion of differentiating between
//     'array'-, 'vector'-, 'matrix'-style axes.
//   * 'View' does not explicitly place any restrictions on the type 'T'. Though
//     for maximum flexibility one should use elementary types (i.e. 'float',
//     'double'). For example instead of
//         View<Matrix3<Complex<double>>>(shape={10,10})
//     one should use
//         View<double>(shape={10,10,3,3,2})
//     which allow potentially more efficient data layouts. Of course, this
//     places an increased burden on the the kernel to interpret the data
//     correctly.
template <class T> class View : public Layout
{
	T *data_ = nullptr;

  public:
	View() = default;

	// full constructor
	explicit View(T *data, Index shape, Index stride)
	    : Layout(shape, stride), data_(data)
	{}

	// convenience constructors assuming "row-major", contiguous layout
	explicit View(T *data, Index shape)
	    : Layout(shape, make_row_major_strides(shape)), data_(data)
	{}

	// implicit View<T> to View<const T>  conversion
	operator View<const T>() const
	{
		return View<const T>(data_, shape(), stride());
	}

	T *data() const { return data_; }
};

} // namespace hops