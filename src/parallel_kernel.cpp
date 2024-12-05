#include "hops/parallel_kernel.h"

#include "fmt/format.h"
#include "fmt/ranges.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

hops::RawKernel
hops::make_parallel_kernel(Signature const &signature,
                           std::string_view source_fragment,
                           std::span<const std::string> type_list)
{
	// NOTE: the `strided_*` structs may only contain a single pointer member.
	// On the C++ side, these classes dont exist and are passed into the kernel
	// as raw pointers.
	auto source = std::format(R"raw(
#include "hops_cuda.h"
using namespace hops;

__device__ void hops_func({})
{{
  {}
}}

template<class Arg, class... Args>
__global__ void hops_kernel(dim3 hops_total_dim, Arg arg, Args... args)
{{
  dim3 hops_tid;
  hops_tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  hops_tid.y = blockIdx.y * blockDim.y + threadIdx.y;
  hops_tid.z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (hops_tid.x < hops_total_dim.x && hops_tid.y < hops_total_dim.y && hops_tid.z < hops_total_dim.z)
  {{
    auto out = arg.read(hops_tid);
    hops_func(out, args.read(hops_tid)...);
    arg.write(hops_tid, out);
  }}
}}
			)raw",
	                          signature.cuda(), source_fragment);

	std::string kernel_name =
	    fmt::format("hops_kernel<{}>", fmt::join(type_list, ", "));
	// fmt::print("full kernel name: {}\n", kernel_name);
	return RawKernel(source, "parallel_kernel.cu", kernel_name);
}
