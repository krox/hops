#pragma once

// basic device memory management.
// Practically just a shallow wrapper around cuMem{Alloc,Free},
// cuMemcpy{HtoD,DtoH,..} and friends.

#include "hops/view.h"
#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace hops {

// un-typed backend implementation
void *device_malloc(size_t n);
void device_free(void *ptr);
void device_memcpy(void *dst, void const *src, size_t n);
void device_memcpy_2d(void *dst, size_t dpitch, void const *src, size_t spitch,
                      size_t width, size_t height);
void device_memclear(void *ptr, size_t n);
void device_memcpy_to_host(void *dst, void const *src, size_t n);
void device_memcpy_from_host(void *dst, void const *src, size_t n);

// RAII wrapper for device memory
//   * The template parameter 'T' is just for the convenience of not having to
//     cast pointers and multiply by 'sizeof(T)' all the time, and '.view()'
//     returning the correct type. No constructors/destructors are ever called
//     on the device memory. So if an application wants to, just using
//     'device_buffer<char>' and casting pointers manually would be fine.
template <class T> class device_buffer
{
	T *data_ = nullptr;
	size_t size_ = 0;

  public:
	device_buffer() = default;
	constexpr device_buffer(std::nullptr_t) noexcept {}
	explicit device_buffer(size_t n)
	    : data_(static_cast<T *>(device_malloc(n * sizeof(T)))), size_(n)
	{}

	// special members (move-only)
	device_buffer(device_buffer &&other) noexcept
	    : data_(std::exchange(other.data_, nullptr)),
	      size_(std::exchange(other.size_, 0))
	{}

	device_buffer &operator=(device_buffer &&other) noexcept
	{
		if (this != &other)
		{
			reset();
			data_ = std::exchange(other.data_, nullptr);
			size_ = std::exchange(other.size_, 0);
		}
		return *this;
	}

	~device_buffer() { reset(); }

	void reset() noexcept
	{
		device_free(data_);
		data_ = nullptr;
		size_ = 0;
	}

	friend void swap(device_buffer &a, device_buffer &b) noexcept
	{
		using std::swap;
		swap(a.data_, b.data_);
		swap(a.size_, b.size_);
	}

	// basic field accessors
	constexpr explicit operator bool() const noexcept
	{
		return data_ != nullptr;
	}
	constexpr size_t size() const noexcept { return size_; }
	constexpr size_t bytes() const noexcept { return size_ * sizeof(T); }
	constexpr T *data() noexcept { return data_; }
	constexpr T const *data() const noexcept { return data_; }

	// converting to a 'View' (for use in kernels and such).
	constexpr View<T> view() noexcept { return View(data_, size_); }
	constexpr View<const T> view() const noexcept { return View(data_, size_); }

	/*constexpr View<T> view(Index shape) noexcept
	{
	    auto r = View<T>::contiguous(data_, shape);
	    assert(r.size() == size());
	    return r;
	}
	constexpr View<const T> view(Index shape) const noexcept
	{
	    auto r = View<const T>::contiguous(data_, shape);
	    assert(r.size() == size());
	    return r;
	}*/

	// convenience memory transfer functions
	static device_buffer from_host(std::span<T const> src)
	{
		device_buffer dst(src.size());
		device_memcpy_from_host(dst.data(), src.data(), dst.bytes());
		return dst;
	}
	std::vector<T> to_host() const
	{
		std::vector<T> dst(size());
		device_memcpy_to_host(dst.data(), data_, bytes());
		return dst;
	}

	// do we even need indexing/iterators?
};

} // namespace hops