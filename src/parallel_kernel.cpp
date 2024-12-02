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
template <class T, int stride_x, int stride_y, int stride_z> struct strided
{{
  T *data_;
}};

template <class T, int stride_x, int stride_y, int stride_z, int stride_complex> struct strided_complex
{{
  T* data_;
}};

template<class T>
T read(T a, dim3)
{{
  return a;
}}

template<class T, int x, int y, int z>
T read(strided<T,x,y,z> a, dim3 tid)
{{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  return ptr[0];
}}

template<class T, int x, int y, int z>
void write(strided<T,x,y,z> a, dim3 tid, T value)
{{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  ptr[0] = value;
}}

/*
template<class T, int x, int y, int z, int c>
std::complex<T> read(stride_complex<T,x,y,z> a, dim3 tid)
{{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  return std::complex<T>(ptr[0], ptr[c]);
}}

template<class T, int x, int y, int z, int c>
void write(stride_complex<T,x,y,z> a, dim3 tid, std::complex<T> value)
{{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  ptr[0] = value.real();
  ptr[c] = value.imag();
}}
*/

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
	return RawKernel(source, "parallel_kernel.cu", kernel_name);
}
