#include "hops/kernel.h"

#include <cassert>
#include <format>

namespace {
using namespace hops;
// RAII wrapper and some conveneince functions for nvrtcProgram
//   * with a bit of cleanup, could make this part hops' public API, but wasnt
//     needed so far.
class NvrtcProgram
{
	nvrtcProgram prog_ = {};

  public:
	NvrtcProgram(std::string const &source, std::string const &filename)
	{
		check(nvrtcCreateProgram(&prog_, source.c_str(), filename.c_str(), 0,
		                         nullptr, nullptr));
	}

	~NvrtcProgram() { nvrtcDestroyProgram(&prog_); }
	NvrtcProgram(NvrtcProgram const &) = delete;
	NvrtcProgram &operator=(NvrtcProgram const &) = delete;
	nvrtcProgram get() const { return prog_; }

	void compile()
	{
		char const *ops[] = {"-default-device"};
		auto r = nvrtcCompileProgram(prog_, 1, ops);
		if (r == NVRTC_SUCCESS)
			return;
		size_t logSize;

		check(nvrtcGetProgramLogSize(prog_, &logSize));
		std::string log;
		log.resize(logSize);
		check(nvrtcGetProgramLog(prog_, log.data()));
		throw Error("Failed to compile CUDA code: " + log);
	}

	std::string get_ptx()
	{
		size_t ptxSize;
		check(nvrtcGetPTXSize(prog_, &ptxSize));
		std::string ptx;
		ptx.resize(ptxSize);
		check(nvrtcGetPTX(prog_, ptx.data()));
		return ptx;
	}
};

} // namespace

hops::Kernel::Kernel(std::string const &source, std::string const &filename,
                     std::string const &kernel_name)
{
	// create/compile
	auto prog = NvrtcProgram(source, filename);
	check(nvrtcAddNameExpression(prog.get(), std::string(kernel_name).c_str()));
	prog.compile();
	char const *lowered_kernel_name_;
	check(nvrtcGetLoweredName(prog.get(), kernel_name.c_str(),
	                          &lowered_kernel_name_));

	// load the program (depending on CUDA settings, actual loading to GPU might
	// be deferred further, but thats transparent to us)
	auto ptx = prog.get_ptx();
	check(cuLibraryLoadData(&lib_, ptx.c_str(), nullptr, nullptr, 0, nullptr,
	                        nullptr, 0));

	// get the kernel
	check(cuLibraryGetKernel(&f_, lib_, lowered_kernel_name_));
}

namespace {
using namespace hops;

std::string make_param_list(Signature const &signature)
{
	std::string param_list = "dim3 hops_total_dim";
	for (auto const &param : signature.params())
	{
		param_list += ", ";

		switch (param.kind)
		{
		case ParameterKind::raw:
			param_list += param.type + " " + param.name;
			break;
		case ParameterKind::in:
			param_list += "parallel<const " + param.type + "> " + param.name;
			break;
		case ParameterKind::out:
		case ParameterKind::inout:
			param_list += "parallel<" + param.type + "> " + param.name;
			break;
		default:
			assert(false);
		}
	}

	return param_list;
}
} // namespace

hops::ParallelKernel::ParallelKernel(hops::Signature const &signature,
                                     std::string const &source_fragment)
    : signature_(signature)
{
	auto param_list = make_param_list(signature);

	auto source = std::format(R"raw(


template <class T> class parallel
{{
	T *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
    // access the part of the array that belongs to the current thread
	T &operator()() const
	{{
		// counting on the cuda compiler to merge calculations of x, y, z across
		// multiple parallel arrays. Should check some PTX output to be sure...
		auto x = blockIdx.x * blockDim.x + threadIdx.x;
		auto y = blockIdx.y * blockDim.y + threadIdx.y;
		auto z = blockIdx.z * blockDim.z + threadIdx.z;
		return data_[x * stride_x_ + y * stride_y_ + z * stride_z_];
	}}
}};

__global__ void kernel({})
{{
  dim3 hops_tid;
  hops_tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  hops_tid.y = blockIdx.y * blockDim.y + threadIdx.y;
  hops_tid.z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (hops_tid.x < hops_total_dim.x && hops_tid.y < hops_total_dim.y && hops_tid.z < hops_total_dim.z)
  {{
    {}
  }}
}}
			)raw",
	                          param_list, source_fragment);
	instance_ = Kernel(source, "parallel_kernel.cu");
}