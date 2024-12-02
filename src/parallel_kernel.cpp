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
hops::make_parallel_kernel(std::string_view source_fragment,
                           std::string_view func_name,
                           std::span<const std::string> type_list)
{
	// NOTE: the `strided_*` structs may only contain a single pointer member.
	// On the C++ side, these classes dont exist and are passed into the kernel
	// as raw pointers.
	auto source = std::format(R"raw(
#include "hops_cuda.h"
using namespace hops;

{}

template<class Arg, class... Args>
__global__ void hops_kernel(dim3 hops_total_dim, Arg arg, Args... args)
{{
  dim3 hops_tid;
  hops_tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  hops_tid.y = blockIdx.y * blockDim.y + threadIdx.y;
  hops_tid.z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (hops_tid.x < hops_total_dim.x && hops_tid.y < hops_total_dim.y && hops_tid.z < hops_total_dim.z)
  {{
    auto out = read(arg, hops_tid);
    {}(out, read(args, hops_tid)...);
    write(arg, hops_tid, out);
  }}
}}
			)raw",
	                          source_fragment, func_name);

	std::string kernel_name =
	    fmt::format("hops_kernel<{}>", fmt::join(type_list, ", "));
	// fmt::print("full kernel name: {}\n", kernel_name);
	return RawKernel(source, "parallel_kernel.cu", kernel_name);
}

std::string hops::make_cuda_type(Layout const &layout)
{
	assert(layout.ndim() <= 3);
	assert(layout.type().height() == 1 && layout.type().width() == 1);

	// note: `.complex_stride()` is 0 for real data.
	if (layout.complexity() == Complexity::real)
		return fmt::format("strided<{},{},{},{},0,0,0>",
		                   cuda(layout.precision()), layout.stride(0),
		                   layout.stride(1), layout.stride(2));
	if (layout.complexity() == Complexity::complex)
		return fmt::format("strided<hops::complex<{}>,{},{},{},{},0,0>",
		                   cuda(layout.precision()), layout.stride(0),
		                   layout.stride(1), layout.stride(2),
		                   layout.complex_stride());
	assert(false);
}
