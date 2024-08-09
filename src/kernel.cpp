#include "hops/kernel.h"

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