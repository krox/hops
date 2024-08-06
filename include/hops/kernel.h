#pragma once

#include "hops/error.h"
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hops {

// this class combines compilation of cuda code (via the NVRTC library) and
// loading of the resulting kernels to the GPU.
class CudaModule
{
	nvrtcProgram prog_ = {};
	CUmodule module_ = {}; // remains nullptr until compilation

	std::vector<std::string> compile_options_;

	// unmangled -> lowered names of kernel functions
	std::unordered_map<std::string, std::string> names_;

  public:
	// create a cuda "program" from a single source code "file"
	//   * the "filename" is only used for error messages
	//   * the program can contain arbitrary number of kernels
	//   * does not yet compile/load anything, which means this does not require
	//     an active CUDA context, and can be called in a static constructor
	//     just fine.
	explicit CudaModule(std::string const &source,
	                    std::string const &filename = "<anonymous>",
	                    std::span<const std::string> compile_options = {},
	                    std::span<const std::string> kernel_names = {});

	// add a "name expression". useful for dealing with name mangling and
	// required to force template instantiation.
	//   * must be called before compilation
	//   * understands some bash-style expansions. e.g. "foo<{float,double}>"
	//     will add two names. Mostly useful for templated functions.
	void add_name(std::string const &name);

	// compile and load the module
	//   * this requires an active CUDA context
	//   * no need to call this explicitly, as the first call to 'get_function'
	//     will trigger compilation automatically.
	void compile();

	// get a kernel/function from the module by name
	//   * throws if the function is not found
	//   * 'name' can be either
	//       * a properly mangled name of a function (easy for 'extern "C"'),
	//       * or a proper C++ name that was included in the list of
	//        'kernel_names' at compile time (this is required for templated
	//        functions in order to force instantiation)
	//   * 'CUfunction' should be considered a non-owning reference. I.e. it
	//     does not need a destructor but will become dangling if the module is
	//     destroyed or unloaded.
	CUfunction get_function(std::string const &name);

	void release() noexcept {}

	// not copyable.
	// NOTE: also not movable for now. In a possible future, a fancy
	// kernel-handle could contain a pointer back to the module it resides in.
	CudaModule(CudaModule const &) = delete;
	CudaModule &operator=(CudaModule const &) = delete;

	~CudaModule() noexcept
	{
		// TODO: currently we rely on context destruction to clean up the
		// module, which is not ideal. But it's not clear how to do it better
		// given that 'hops::{init,finalize}' are called explicitly
		// if (module_)
		//	check(cuModuleUnload(module_));
		if (prog_)
			check(nvrtcDestroyProgram(&prog_));
	}
};
} // namespace hops