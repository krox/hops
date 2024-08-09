// same as examples/saxpy.cpp, but based on the 'ParallelKernel' class

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

	// generate some input data
	size_t n = 4096;
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
	// NOTE: first call to '.get_function' compiles/loads the cuda module
	auto kernel = hops::ParallelKernel<float *, float, float *, float *>(
	    "out[x] = alpha * a[x] + b[x];",
	    std::vector<std::string>{"out", "alpha", "a", "b"});
	kernel.launch(n, dOut.data(), a, dX.data(), dY.data());

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