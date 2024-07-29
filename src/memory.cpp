#include "hops/memory.h"

#include "hops/error.h"
#include <cassert>
#include <cuda.h>

// CUDAs driver API uses 'CUdeviceptr', which is a pointer-sized integer type.
// We do 'void*' instead. Ugly reinterpret_casts, but should be valid on all (64
// bit) platforms.
static_assert(sizeof(CUdeviceptr) == sizeof(void *));

hops::DeviceBuffer::DeviceBuffer(size_t size) : bytes_(size)
{
	check(cuMemAlloc(reinterpret_cast<CUdeviceptr *>(&data_), size));
	assert(data_);
}

void hops::DeviceBuffer::release() noexcept
{
	check(cuMemFree(reinterpret_cast<CUdeviceptr>(data_)));
}

void hops::DeviceBuffer::copy_to_host(void *host_data) const
{
	check(
	    cuMemcpyDtoH(host_data, reinterpret_cast<CUdeviceptr>(data_), bytes_));
}

void hops::DeviceBuffer::copy_from_host(void const *host_data)
{
	check(
	    cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(data_), host_data, bytes_));
}