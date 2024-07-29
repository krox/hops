// this example is adapted from CUDA's documentation
// (https://docs.nvidia.com/cuda/nvrtc/index.html#example-saxpy),
// using some (quite shallow) abstractions of hops over using the CUDA runtime
// API directly. Also cleaned up to feel more like proper C++.

#include "hops/hops.h"
#include <cassert>
#include <vector>

const char *saxpy_source = R"raw(
extern "C"  __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    out[tid] = a * x[tid] + y[tid];
  }
}
)raw";

int main()
{
	// init cuda, select device, create context
	hops::init();
	atexit(hops::finalize);

	// compile and load the kernel
	auto mod = hops::CudaModule(saxpy_source, "saxpy.cu");
	auto kernel = mod.get_function("saxpy");

	// generate some input data
	size_t nThreads = 128;
	size_t nBlocks = 32;
	size_t n = nThreads * nBlocks;
	float a = 5.1f;
	auto hX = std::vector<float>(n);
	auto hY = std::vector<float>(n);
	for (size_t i = 0; i < n; ++i)
	{
		hX[i] = static_cast<float>(i);
		hY[i] = static_cast<float>(i * 2);
	}

	// copy data to device
	auto dX = hops::DeviceBuffer::from_host(hX);
	auto dY = hops::DeviceBuffer::from_host(hY);
	auto dOut = hops::DeviceBuffer(n * sizeof(float));

	// execute the SAXPY kernel
	void *args[] = {&a, &dX, &dY, &dOut, &n};
	hops::check(cuLaunchKernel(kernel, nBlocks, 1, 1, // grid dim
	                           nThreads, 1, 1,        // block dim
	                           0, nullptr,            // shared mem and stream
	                           args, 0));             // arguments
	hops::sync(); // maybe not needed (does copy_to_host block?)

	// retrieve and print output
	auto hOut = dOut.copy_to_host<float>();
	assert(hOut.size() == n);
	for (size_t i = 0; i < 2; ++i)
	{
		std::cout << a << " * " << hX[i] << " + " << hY[i] << " = " << hOut[i]
		          << '\n';
	}
	std::cout << "...\n";
	for (size_t i = n - 2; i < n; ++i)
	{
		std::cout << a << " * " << hX[i] << " + " << hY[i] << " = " << hOut[i]
		          << '\n';
	}
}