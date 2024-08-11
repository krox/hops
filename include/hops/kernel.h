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

// parameter type for kernels processing arrays in parallel
template <class T> class parallel
{
	T *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
	parallel(T *data, ptrdiff_t stride_x, ptrdiff_t stride_y,
	         ptrdiff_t stride_z)
	    : data_(data), stride_x_(stride_x), stride_y_(stride_y),
	      stride_z_(stride_z)
	{}
};

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
template <class T> struct typestr;
#define TYPESTR(T)                                                             \
	template <> struct typestr<T>                                              \
	{                                                                          \
		static constexpr std::string_view value = #T;                          \
	}
TYPESTR(float);
TYPESTR(double);
TYPESTR(float *);
TYPESTR(double *);
TYPESTR(float const *);
TYPESTR(double const *);
TYPESTR(int);
TYPESTR(unsigned int);
TYPESTR(long);
TYPESTR(unsigned long);
TYPESTR(long long);
TYPESTR(parallel<float>);
TYPESTR(parallel<double>);
TYPESTR(parallel<float const>);
TYPESTR(parallel<double const>);
#undef TYPESTR

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
//       * more convenience to offset pointers with strides, so that the
//         "user"-code does not have to do
//              arr[x*stride.x + y*stride.y + z*stride.z + <other_offsets>]
//         but rather something like
//              arr_local[<other_offset>]
//         Also, maybe this fits neatly together with generating optimized
//         kernel for stride=1 or 1D/2D cases.
template <class... Args> class ParallelKernel
{
	std::unique_ptr<CudaLibrary> lib_;

  public:
	ParallelKernel(std::string const &source_fragment,
	               std::span<const std::string> arg_names)
	{
		if (arg_names.size() != sizeof...(Args))
			throw std::runtime_error(std::format(
			    "ParallelKernel: expected {} argument names, got {}",
			    sizeof...(Args), arg_names.size()));
		std::string param_list = "dim3 totalDim";
		auto arg_types = std::array{std::string(typestr<Args>::value)...};
		for (size_t i = 0; i < arg_names.size(); ++i)
			param_list += ", "s + arg_types[i] + " " + arg_names[i];

		auto source = std::format(R"raw(


template <class T> class parallel
{{
	T *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
	parallel(T *data, ptrdiff_t stride_x, ptrdiff_t stride_y, ptrdiff_t stride_z)
	    : data_(data), stride_x_(stride_x), stride_y_(stride_y),
	      stride_z_(stride_z)
	{{}}


#ifdef __CUDA_ARCH__
    // access the part of the array that belongs to the current thread
	T &operator*() const
	{{
		// counting on the cuda compiler to merge calculations of x, y, z across
		// multiple parallel arrays. Should check some PTX output to be sure...
		auto x = blockIdx.x * blockDim.x + threadIdx.x;
		auto y = blockIdx.y * blockDim.y + threadIdx.y;
		auto z = blockIdx.z * blockDim.z + threadIdx.z;
		return data_[x * stride_x_ + y * stride_y_ + z * stride_z_];
	}}
#endif
}};

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
		lib_ = std::make_unique<CudaLibrary>(source, "parallel_kernel.cu");
	}

	void launch(dim3 size, Args... args)
	{
		auto f = lib_->get_kernel("kernel");

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