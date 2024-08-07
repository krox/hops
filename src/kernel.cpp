#include "hops/kernel.h"

hops::CudaModule::CudaModule(std::string const &source,
                             std::string const &filename,
                             std::span<const std::string> compile_options,
                             std::span<const std::string> kernel_names)
    : compile_options_(compile_options.begin(), compile_options.end())
{
	compile_options_.push_back("-default-device");
	check(nvrtcCreateProgram(&prog_, source.c_str(), filename.c_str(), 0,
	                         nullptr, nullptr));
	for (auto const &name : kernel_names)
		add_name(name);
}

void hops::CudaModule::compile()
{
	if (module_)
		throw std::logic_error("CudaModule::compile: already compiled");

	// compile it
	std::vector<char const *> options;
	for (auto const &opt : compile_options_)
		options.push_back(opt.c_str());
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
	for (auto &[name, lowered] : names_)
	{
		char const *buf;
		check(nvrtcGetLoweredName(prog_, name.c_str(), &buf));
		lowered = std::string(buf);
		// std::cout << "Lowered name: " << name << " -> " << lowered << '\n';
	}

	// step 3: load it into a module
	size_t ptxSize;
	check(nvrtcGetPTXSize(prog_, &ptxSize));
	std::string ptx;
	ptx.resize(ptxSize);
	check(nvrtcGetPTX(prog_, ptx.data()));
	check(cuModuleLoadDataEx(&module_, ptx.c_str(), 0, 0, 0));
}

void hops::CudaModule::add_name(std::string const &name)
{
	if (size_t i = name.find_first_of('{'); i != std::string::npos)
		if (size_t j = name.find_first_of('}', i); j != std::string::npos)
		{
			auto prefix = name.substr(0, i);
			auto suffix = name.substr(j + 1);
			// split options by ','
			auto options = name.substr(i + 1, j - i - 1);
			// add each option
			for (size_t i = 0, j = 0; i < options.size(); i = j + 1)
			{
				j = options.find(',', i);
				if (j == std::string::npos)
					j = options.size();
				add_name(prefix + options.substr(i, j - i) + suffix);
			}
			return;
		}
	// std::cout << "Adding name: '" << name << "'\n";
	names_[name] = "";
	check(nvrtcAddNameExpression(prog_, name.c_str()));
}

CUfunction hops::CudaModule::get_function(std::string const &name)
{
	if (!module_)
		compile();

	char const *lowered_name;
	if (auto it = names_.find(name); it != names_.end())
		lowered_name = it->second.c_str();
	else
		lowered_name = name.c_str();

	CUfunction func;
	auto r = cuModuleGetFunction(&func, module_, lowered_name);
	if (r == CUDA_ERROR_NOT_FOUND)
		throw std::runtime_error(
		    "CudaModule::get_function: function not found: " + name);
	check(r);
	return func;
}