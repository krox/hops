// this example is adapted from CUDA's documentation
// (https://docs.nvidia.com/cuda/nvrtc/index.html#example-saxpy),
// using some (quite shallow) abstractions of hops over using the CUDA runtime
// API directly. Also cleaned up to feel more like proper C++.

#include "hops/hops.h"
#include <cassert>
#include <string>
#include <vector>

using namespace std::string_literals;

int main()
{
	// init cuda, select device, create context
	hops::init();
	atexit(hops::finalize);

	auto kernel = hops::RawKernel(R"raw(
template<class T>
__global__ void axpy(T a, T *x, T *y, T *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    out[tid] = a * x[tid] + y[tid];
  }
}
)raw",
	                              "saxpy.cu", "axpy<float>");

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
	auto dX = hops::device_buffer<float>::from_host(hX);
	auto dY = hops::device_buffer<float>::from_host(hY);
	auto dOut = hops::device_buffer<float>(n);

	// execute the SAXPY kernel
	kernel.launch(nBlocks, nThreads, a, dX.data(), dY.data(), dOut.data(), n);
	hops::sync(); // maybe not needed (does copy_to_host block?)

	// retrieve and print output
	auto hOut = dOut.to_host();
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