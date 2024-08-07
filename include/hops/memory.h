#pragma once

// basic device memory management.
// Practically just a shallow wrapper around cuMem{Alloc,Free},
// cuMemcpy{HtoD,DtoH,..} and friends.

#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>
// nice detail: no need to include cuda.h here. All hidden in memory.cpp

namespace hops {

// untypted device memory buffer
//   * Could be typed of course, but the only two reasonable instances in hops
//     design would be 'float' and 'double' (maybe float16?). Not worth making
//     this a template just for that I think.
class DeviceBuffer
{
	void *data_ = 0;
	size_t bytes_ = 0;

  public:
	DeviceBuffer() = default;

	// allocate a buffer of 'size' bytes (uninitialized)
	explicit DeviceBuffer(size_t size);

	// pseudo-constructors copying data from host to device
	static DeviceBuffer from_host(std::span<const float> host_data)
	{
		auto buf = DeviceBuffer(host_data.size() * sizeof(float));
		buf.copy_from_host(host_data.data());
		return buf;
	}
	static DeviceBuffer from_host(std::span<const double> host_data)
	{
		auto buf = DeviceBuffer(host_data.size() * sizeof(double));
		buf.copy_from_host(host_data.data());
		return buf;
	}

	// free the device memory
	void release() noexcept;
	~DeviceBuffer() { release(); }

	// move semantics
	DeviceBuffer(DeviceBuffer &&other) noexcept
	    : data_(std::exchange(other.data_, nullptr)),
	      bytes_(std::exchange(other.bytes_, 0))
	{}
	DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
	{
		if (this != &other)
		{
			release();
			data_ = std::exchange(other.data_, nullptr);
			bytes_ = std::exchange(other.bytes_, 0);
		}
		return *this;
	}

	void *data_raw() noexcept { return data_; }
	void const *data_raw() const noexcept { return data_; }
	template <class T> T *data() noexcept
	{
		assert(bytes_ % sizeof(T) == 0);
		return static_cast<T *>(data_);
	}
	template <class T> T const *data() const noexcept
	{
		assert(bytes_ % sizeof(T) == 0);
		return static_cast<T const *>(data_);
	}

	size_t bytes() const noexcept { return bytes_; }
	template <class T> size_t size() const noexcept
	{
		assert(bytes_ % sizeof(T) == 0);
		return bytes_ / sizeof(T);
	}

	// move data between host and device.
	// NOTE: thanks to unified adress space, CUDA can figure out which side
	// any given pointer resides on, but I still like to be explicit about it.
	void copy_from_host(void const *host_data);
	void copy_to_host(void *host_data) const;
	template <class T> std::vector<T> copy_to_host() const
	{
		assert(bytes_ % sizeof(T) == 0);
		std::vector<T> host_data(bytes_ / sizeof(T));
		copy_to_host(host_data.data());
		return host_data;
	}
};

// missing: device-to-device copies (including async and 2D versions)
// also missing: some kind of 'zero' function.
} // namespace hops