#include "hops/hops.h"

#include <cassert>

namespace {
CUcontext cuda_context = nullptr;
}

void hops::init(int device_id, int verbose)
{
	// TODO: is it a problem if we call this multiple times?
	check(cuInit(0));

	int driverVersion;
	cuDriverGetVersion(&driverVersion);
	if (verbose)
		std::cout << "CUDA driver version: " << driverVersion << '\n';

	int device_count;
	check(cuDeviceGetCount(&device_count));
	if (device_count == 0)
		throw Error("No CUDA devices found");

	if (device_id >= device_count || device_id < 0)
		throw Error("Specified CUDA device not found");

	CUdevice device;
	check(cuDeviceGet(&device, device_id));
	check(cuCtxCreate(&cuda_context, 0, device));

	// TODO: use cuDevicePrimaryCtxRetain

	char name[256];
	check(cuDeviceGetName(name, 256, 0));
	if (verbose)
		std::cout << "Using device: " << name << '\n';

	size_t totalMem;
	int unifiedAddressing = -1;
	check(cuDeviceTotalMem(&totalMem, device));
	check(cuDeviceGetAttribute(&unifiedAddressing,
	                           CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));

	// sanity check. Unified adressing should be enabled on all 64 bit
	// platforms. (Not to be confused with unified memory)
	assert(unifiedAddressing);

	if (verbose)
	{
		std::cout << "Total device memory: " << (totalMem >> 20) << " MiB"
		          << std::endl;
	}
}

void hops::finalize() noexcept
{
	if (cuda_context)
	{
		check(cuCtxDestroy(cuda_context));
		cuda_context = nullptr;
	}
}

void hops::sync() { check(cuCtxSynchronize()); }