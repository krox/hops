#pragma once

#include "hops/error.h"
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <string_view>
#include <utility>

namespace hops {

// a CUDA module is a collectection of (compiled and loaded) kernels
class CudaModule
{
	CUmodule module_;

  public:
	// create a cuda module from a single source code "file"
	//   * the "filename" is only used for error messages
	//   * the program can contain arbitrary number of kernels
	//   * in cuda, this is three steps: create + compile + load,
	//     dont think there is a reason to split it up here.
	//   * future: transparently caching compilation results (PTX or CUBIN)
	//     would be cool here. Maybe implement together with caching of tuning
	//     results.
	explicit CudaModule(
	    std::string const &source,
	    std::string const &filename = "<anonymous>" /*, compile parameters*/);

	// get a kernel/function from the module by name
	//   * throws if the function is not found
	//   * 'CUfunction' should be considered a non-owning reference. I.e. it
	//     does not need a destructor but will become dangling if the module is
	//     destroyed or unloaded.
	CUfunction get_function(std::string const &name);

	CudaModule(CudaModule const &) = delete;
	CudaModule &operator=(CudaModule const &) = delete;
	CudaModule(CudaModule &&other) noexcept
	    : module_(std::exchange(other.module_, nullptr))
	{}
	CudaModule &operator=(CudaModule &&other) noexcept
	{
		if (this != &other)
		{
			if (module_)
				check(cuModuleUnload(module_));
			module_ = std::exchange(other.module_, nullptr);
		}
		return *this;
	}
	~CudaModule()
	{
		if (module_)
			check(cuModuleUnload(module_));
	}
};
} // namespace hops