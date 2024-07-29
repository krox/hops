#include "hops/kernel.h"

namespace {
// shouldnt (a vastly more advanced version of) this be in the standard library?
template <class Callable> struct ScopeGuard
{
	Callable c;
	~ScopeGuard() { c(); }
	ScopeGuard(Callable c) : c(c) {}
};
} // namespace

hops::CudaModule::CudaModule(std::string const &source,
                             std::string const &filename)
{
	// step 1: create the program
	nvrtcProgram prog = {};
	check(nvrtcCreateProgram(&prog, source.c_str(), filename.c_str(), 0,
	                         nullptr, nullptr));
	auto guard = ScopeGuard{[&] { check(nvrtcDestroyProgram(&prog)); }};

	// step 2: compile it
	const char *opts[] = {"--device-as-default-execution-space"};
	auto r = nvrtcCompileProgram(prog, 1, opts);
	if (r != NVRTC_SUCCESS)
	{
		size_t logSize;
		check(nvrtcGetProgramLogSize(prog, &logSize));
		std::string log;
		log.resize(logSize);
		check(nvrtcGetProgramLog(prog, log.data()));
		std::cout << log << '\n';
		check(r); // will throw
	}

	// step 3: load it into a module
	size_t ptxSize;
	check(nvrtcGetPTXSize(prog, &ptxSize));
	std::string ptx;
	ptx.resize(ptxSize);
	check(nvrtcGetPTX(prog, ptx.data()));
	check(cuModuleLoadDataEx(&module_, ptx.c_str(), 0, 0, 0));
}

CUfunction hops::CudaModule::get_function(std::string const &name)
{
	CUfunction func;
	check(cuModuleGetFunction(&func, module_, name.c_str()));
	return func;
}