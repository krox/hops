#pragma once

#include <cuda.h>
#include <format>
#include <nvrtc.h>
#include <source_location>
#include <stdexcept>
#include <string>

namespace hops {

struct Error : std::runtime_error
{
	Error(std::string const &message) : std::runtime_error(message) {}
};

// throw an exception if the result is not CUDA_SUCCESS
inline void
check(CUresult r,
      std::source_location const location = std::source_location::current())
{
	if (r == CUDA_SUCCESS) [[likely]]
		return;
	const char *msg;
	cuGetErrorName(r, &msg);
	throw Error(std::format("{}({}): CUDA error: {}", location.file_name(),
	                        location.line(), msg));
}

// throw an exception if the result is not NVRTC_SUCCESS
inline void
check(nvrtcResult r,
      std::source_location const location = std::source_location::current())
{
	if (r == NVRTC_SUCCESS) [[likely]]
		return;
	throw Error(std::format("{}({}): nvrtc error: {}", location.file_name(),
	                        location.line(), nvrtcGetErrorString(r)));
}
} // namespace hops