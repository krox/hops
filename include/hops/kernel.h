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

// tiny wrapper around 'cuLaunchKernel'
//   * there is no way to check the types of the arguments, so make sure they
//     match the kernel signature.
template <class... Args>
void launch(CUkernel f, dim3 grid, dim3 block, Args... args)
{
	void *arg_ptrs[] = {&args...};
	check(cuLaunchKernel((CUfunction)f, grid.x, grid.y, grid.z, block.x,
	                     block.y, block.z, 0, nullptr, arg_ptrs, 0));
}

// this class combines compilation of cuda code (via the NVRTC library) and
// loading of the resulting kernels to the GPU.
class CudaLibrary
{
	nvrtcProgram prog_ = {};
	CUlibrary lib_ = {}; // remains nullptr until compilation

	// unmangled -> lowered names of kernel functions
	std::unordered_map<std::string, std::string> names_;

  public:
	// create/compile a cuda "program" from a single source code "file"
	//   * this does not require an active CUDA context, but 'cuInit()' must
	//     have been called before
	//   * the "filename" is only used for error messages
	//   * 'kernel_names' is a list of kernels in proper C++ syntax that should
	//	    be included in the program. This is useful to deal with name
	//      mangling and required for templated kernels in order to force
	//      instantiation.
	explicit CudaLibrary(std::string const &source,
	                     std::string const &filename = "<anonymous>",
	                     std::span<const std::string> compile_options = {},
	                     std::span<const std::string> kernel_names = {});

	// get a kernel from the library by name
	//   * throws if the kernel is not found
	//   * 'name' can be either
	//       * a properly mangled name of a kernel (easy for 'extern "C"'),
	//       * a name that was included in the list of 'kernel_names' in the
	//         library constructor
	//   * 'CUkernel' is a non-owning reference. I.e. it does not require a
	//     destructor but will become dangling if the library is unloaded.
	CUkernel get_kernel(std::string const &name);

	// not copyable.
	// NOTE: also not movable for now. In a possible future, a fancy
	// kernel-handle could contain a pointer back to the library it resides in.
	CudaLibrary(CudaLibrary const &) = delete;
	CudaLibrary &operator=(CudaLibrary const &) = delete;

	~CudaLibrary() noexcept
	{
		if (lib_)
			check(cuLibraryUnload(lib_));
		if (prog_)
			check(nvrtcDestroyProgram(&prog_));
	}
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
	std::unique_ptr<CudaLibrary> lib_;
	Signature signature_;

  public:
	ParallelKernel(Signature const &signature,
	               std::string const &source_fragment);

	template <class... Args> void launch(dim3 size, Args... args)
	{
		auto f = lib_->get_kernel("kernel");

		// TODO: check argument types...

		// automatic block size should be a lot more sophisticated...
		auto block = dim3(256);

		dim3 grid;
		grid.x = (size.x + block.x - 1) / block.x;
		grid.y = (size.y + block.y - 1) / block.y;
		grid.z = (size.z + block.z - 1) / block.z;

		hops::launch<dim3, Args...>(f, grid, block, size, args...);
	}
};

} // namespace hops