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
	auto source = std::format(R"raw(
template <class T, int stride_x, int stride_y, int stride_z> struct strided
{{
    // important: must only contain a single pointer, nothing more. When launching a kernel, parameters of type strided<...> are filled by simple 'void*' arguments
	T *data_ = nullptr;

    // access the part of the array that belongs to the current thread
	T &operator()(dim3 tid) const
	{{
		return data_[tid.x * stride_x + tid.y * stride_y + tid.z * stride_z];
	}}
}};

{}

template<class T, int x, int y, int z>
T& get_elem(strided<T,x,y,z> p, dim3 tid)
{{
  return p(tid);
}}

template<class T>
T get_elem(T scalar, dim3)
{{
  return scalar;
}}

template<class... Args>
__global__ void hops_kernel(dim3 hops_total_dim, Args... args)
{{
  dim3 hops_tid;
  hops_tid.x = blockIdx.x * blockDim.x + threadIdx.x;
  hops_tid.y = blockIdx.y * blockDim.y + threadIdx.y;
  hops_tid.z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (hops_tid.x < hops_total_dim.x && hops_tid.y < hops_total_dim.y && hops_tid.z < hops_total_dim.z)
  {{
    {}(get_elem(args, hops_tid)...);
  }}
}}
			)raw",
	                          source_fragment, func_name);

	std::string kernel_name =
	    fmt::format("hops_kernel<{}>", fmt::join(type_list, ", "));
	return RawKernel(source, "parallel_kernel.cu", kernel_name);
}
