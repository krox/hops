#include "hops/kernel.h"

#include <cassert>
#include <format>

hops::CudaLibrary::CudaLibrary(std::string const &source,
                               std::string const &filename,
                               std::span<const std::string> compile_options,
                               std::span<const std::string> kernel_names)
{
	check(nvrtcCreateProgram(&prog_, source.c_str(), filename.c_str(), 0,
	                         nullptr, nullptr));
	for (auto const &name : kernel_names)
		check(nvrtcAddNameExpression(prog_, name.c_str()));

	// compile it
	std::vector<char const *> options;
	for (auto const &opt : compile_options)
		options.push_back(opt.c_str());
	options.push_back("-default-device");
	auto r = nvrtcCompileProgram(prog_, options.size(), options.data());
	if (r != NVRTC_SUCCESS)
	{
		size_t logSize;
		check(nvrtcGetProgramLogSize(prog_, &logSize));
		std::string log;
		log.resize(logSize);
		check(nvrtcGetProgramLog(prog_, log.data()));
		std::cout << log << '\n';
		check(r); // will throw
	}

	// get lowered names of explicit kernel names
	for (auto const &name : kernel_names)
	{
		char const *buf;
		check(nvrtcGetLoweredName(prog_, name.c_str(), &buf));
		names_[name] = std::string(buf);
	}

	// step 3: load it into a library
	size_t ptxSize;
	check(nvrtcGetPTXSize(prog_, &ptxSize));
	std::string ptx;
	ptx.resize(ptxSize);
	check(nvrtcGetPTX(prog_, ptx.data()));
	check(cuLibraryLoadData(&lib_, ptx.c_str(), nullptr, nullptr, 0, nullptr,
	                        nullptr, 0));
}

CUkernel hops::CudaLibrary::get_kernel(std::string const &name)
{
	assert(lib_);
	// if (!lib_)
	//	compile();

	char const *lowered_name;
	if (auto it = names_.find(name); it != names_.end())
		lowered_name = it->second.c_str();
	else
		lowered_name = name.c_str();

	CUkernel kernel;
	auto r = cuLibraryGetKernel(&kernel, lib_, lowered_name);
	if (r == CUDA_ERROR_NOT_FOUND)
		throw std::runtime_error("CudaLibrary::get_kernel: kernel not found: " +
		                         name);
	check(r);
	return kernel;
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

extern "C" __global__ void kernel({})
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
	lib_ = std::make_unique<CudaLibrary>(source, "parallel_kernel.cu");
}