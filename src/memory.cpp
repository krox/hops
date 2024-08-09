#include "hops/memory.h"

#include "hops/error.h"
#include <cassert>
#include <cuda.h>

// CUDAs driver API uses 'CUdeviceptr' instead of 'void*'. In modern CUDA, that
// is a mostly pointless distinction, so we reinterpret_cast it away. For more
// info (including historic reasons for this distinction), see:
//   https://www.cudahandbook.com/2013/08/why-does-cuda-cudeviceptr-use-unsigned-int-instead-of-void/
static_assert(sizeof(CUdeviceptr) == sizeof(void *));
static_assert(alignof(CUdeviceptr) == alignof(void *));
static_assert(sizeof(CUdeviceptr) == 8);

void *hops::device_malloc(size_t n)
{
	if (n == 0)
		return nullptr;
	void *ptr;
	check(cuMemAlloc((CUdeviceptr *)&ptr, n));
	assert(ptr != nullptr);
	return ptr;
}

void hops::device_free(void *ptr)
{
	if (ptr)
		check(cuMemFree((CUdeviceptr)ptr));
}

void hops::device_memcpy(void *dst, void const *src, size_t n)
{
	if (n == 0)
		return;
	check(cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, n));
}

void hops::device_memcpy_2d(void *dst, size_t dpitch, void const *src,
                            size_t spitch, size_t width, size_t height)
{
	if (width == 0 || height == 0)
		return;

	CUDA_MEMCPY2D cfg = {};
	cfg.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	cfg.srcDevice = (CUdeviceptr)src;
	cfg.srcPitch = spitch;
	cfg.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	cfg.dstDevice = (CUdeviceptr)dst;
	cfg.dstPitch = dpitch;
	cfg.WidthInBytes = width;
	cfg.Height = height;
	cfg.srcXInBytes = cfg.srcY = cfg.dstXInBytes = cfg.dstY = 0;

	check(cuMemcpy2D(&cfg));
}

void hops::device_memcpy_to_host(void *dst, void const *src, size_t n)
{
	if (n == 0)
		return;
	check(cuMemcpyDtoH(dst, (CUdeviceptr)src, n));
}

void hops::device_memcpy_from_host(void *dst, void const *src, size_t n)
{
	if (n == 0)
		return;
	check(cuMemcpyHtoD((CUdeviceptr)dst, src, n));
}

void hops::device_memclear(void *ptr, size_t n)
{
	if (n == 0)
		return;
	check(cuMemsetD8((CUdeviceptr)ptr, 0, n));
}