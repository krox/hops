#pragma once

#include "hops/base.h"
#include "hops/error.h"
#include "hops/signature.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <nvrtc.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hops {

using namespace std::string_literals;

// This class combines compilation of cuda code (via the NVRTC library) and
// loading of the resulting kernels to the GPU.
//   * CUDA supports multiple CUfunction's in a single CUlibrary, but we dont
//     expose this, simplifying the interface.
class Kernel
{
	CUlibrary lib_ = {};
	CUkernel f_ = {};

  public:
	Kernel() = default;

	// create/compile a cuda "program" from a single source code "file"
	//   * this does not require an active CUDA context, but 'cuInit()' must
	//     have been called before
	//   * the "filename" is only used for error messages
	//   * 'kernel_name' is a full C++ name of the kernel function, including
	//     template arguments
	explicit Kernel(std::string const &source,
	                std::string const &filename = "<anonymous>",
	                std::string const &kernel_name = "kernel");

	// Launch the kernel
	//   * there is no way to check the types of the arguments at this level, so
	//     make sure they match the kernel signature.
	template <class... Args> void launch(dim3 grid, dim3 block, Args... args)
	{
		assert(f_);
		void *arg_ptrs[] = {&args...};
		check(cuLaunchKernel((CUfunction)f_, grid.x, grid.y, grid.z, block.x,
		                     block.y, block.z, 0, nullptr, arg_ptrs, 0));
	}

	void unload()
	{
		if (lib_)
			check(cuLibraryUnload(lib_));
		lib_ = nullptr;
		f_ = nullptr;
	}

	// move-only
	Kernel(Kernel const &) = delete;
	Kernel &operator=(Kernel const &) = delete;
	Kernel(Kernel &&other) noexcept
	    : lib_(std::exchange(other.lib_, nullptr)),
	      f_(std::exchange(other.f_, nullptr))
	{}
	Kernel &operator=(Kernel &&other) noexcept
	{
		if (this != &other)
		{
			unload();
			lib_ = std::exchange(other.lib_, nullptr);
			f_ = std::exchange(other.f_, nullptr);
		}
		return *this;
	}
	~Kernel() noexcept { unload(); }
};

// Category of kernels that simply execute code on each GPU thread in 3D
// grid in parallel.
//   * automates some boilerplate in the cuda code
//   * effectively type-safe, as the kernel signature is generated
//   automatically
//   * gridSize/blockSize are automatic, '.launch(...)' just takes a total
//   size
//   * TODO:
//       * more sophisticated block size selection
//       * multiple variants of a kernel in a single class
//         (e.g. float/double, 1-3 dimensions, fixed strides)
class ParallelKernel
{
	Kernel instance_;
	Signature signature_;

  public:
	ParallelKernel(Signature const &signature,
	               std::string const &source_fragment);

	template <class... Args> void launch(dim3 size, Args... args)
	{
		// TODO: check argument types...

		// automatic block size should be a lot more sophisticated...
		auto block = dim3(256);

		dim3 grid;
		grid.x = (size.x + block.x - 1) / block.x;
		grid.y = (size.y + block.y - 1) / block.y;
		grid.z = (size.z + block.z - 1) / block.z;

		instance_.launch<dim3, Args...>(grid, block, size, args...);
	}
};

} // namespace hops