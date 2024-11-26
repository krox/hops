#pragma once

#include "hops/base.h"
#include "hops/error.h"
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

// just for passing to
union KernelArgument
{
	void *ptr;
	float f32;
	double f64;
};

// This class combines compilation of cuda code (via the NVRTC library) and
// loading of the resulting kernels to the GPU.
//   * CUDA supports multiple CUfunction's in a single CUlibrary, but we dont
//     expose this, simplifying the interface.
class RawKernel
{
	CUlibrary lib_ = {};
	CUkernel f_ = {};

  public:
	RawKernel() = default;

	// create/compile a cuda "program" from a single source code "file"
	//   * this does not require an active CUDA context, but 'cuInit()' must
	//     have been called before
	//   * the "filename" is only used for error messages
	//   * 'kernel_name' is a full C++ name of the kernel function, including
	//     template arguments
	explicit RawKernel(std::string const &source,
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

	void launch_raw(dim3 grid, dim3 block, void **args)
	{
		assert(f_);
		assert(args); // yes, super weak validation check. Sadly, Cuda does not
		              // seem to offer introspection to check the types (or even
		              // the number) of the arguments of a compiled kernel.
		check(cuLaunchKernel((CUfunction)f_, grid.x, grid.y, grid.z, block.x,
		                     block.y, block.z, 0, nullptr, args, 0));
	}

	void unload()
	{
		if (lib_)
			check(cuLibraryUnload(lib_));
		lib_ = nullptr;
		f_ = nullptr;
	}

	// move-only
	RawKernel(RawKernel const &) = delete;
	RawKernel &operator=(RawKernel const &) = delete;
	RawKernel(RawKernel &&other) noexcept
	    : lib_(std::exchange(other.lib_, nullptr)),
	      f_(std::exchange(other.f_, nullptr))
	{}
	RawKernel &operator=(RawKernel &&other) noexcept
	{
		if (this != &other)
		{
			unload();
			lib_ = std::exchange(other.lib_, nullptr);
			f_ = std::exchange(other.f_, nullptr);
		}
		return *this;
	}
	~RawKernel() noexcept { unload(); }
};

} // namespace hops