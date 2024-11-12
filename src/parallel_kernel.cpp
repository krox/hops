#include "hops/parallel_kernel.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

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
	instance_ = RawKernel(source, "parallel_kernel.cu");
}
