#pragma once

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

// compatible with CUDA's 'dim3' type
struct dim3
{
	uint32_t x = 1, y = 1, z = 1;

	constexpr dim3() noexcept = default;
	constexpr dim3(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept
	    : x(x), y(y), z(z)
	{}
};

// tiny wrapper around 'cuLaunchKernel'
//   * there is no way to check the types of the arguments, so make sure they
//     match the kernel signature.
template <class... Args>
void launch(CUfunction f, dim3 grid, dim3 block, Args... args)
{
	void *arg_ptrs[] = {&args...};
	check(cuLaunchKernel(f, grid.x, grid.y, grid.z, block.x, block.y, block.z,
	                     0, nullptr, arg_ptrs, 0));
}

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
template <class T> struct typestr;
template <> struct typestr<float>
{
	static constexpr std::string value = "float"s;
};
template <> struct typestr<double>
{
	static constexpr std::string value = "double"s;
};
template <> struct typestr<float *>
{
	static constexpr std::string value = "float*"s;
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
//       * multiple variants of a kernel (e.g. float/double, 1-3 dimensions)
template <class... Args> class ParallelKernel
{
	std::unique_ptr<CudaModule> module_;

  public:
	ParallelKernel(std::string const &source_fragment,
	               std::span<const std::string> arg_names)
	{
		assert(arg_names.size() == sizeof...(Args));
		std::string param_list = "dim3 totalDim";
		auto arg_types = std::array{typestr<Args>::value...};
		for (size_t i = 0; i < arg_names.size(); ++i)
			param_list += ", " + arg_types[i] + " " + arg_names[i];

		auto source = std::format(R"raw(
extern "C" __global__ void kernel({})
{{
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;
  auto z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x < totalDim.x && y < totalDim.y && z < totalDim.z)
  {{
    {}
  }}
}}
			)raw",
		                          param_list, source_fragment);
		module_ = std::make_unique<CudaModule>(source, "parallel_kernel.cu");
	}

	void launch(dim3 size, Args... args)
	{
		auto f = module_->get_function("kernel");

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