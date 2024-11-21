#include "hops/parallel_kernel.h"

#include "fmt/format.h"
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
	std::string param_list;
	bool first = true;
	for (auto const &param : signature.params())
	{
		if (!first)
			param_list += ", ";
		first = false;

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

std::string make_type_list(Signature const &signature)
{
	std::string param_list;
	bool first = true;
	for (auto const &param : signature.params())
	{
		if (!first)
			param_list += ", ";
		first = false;

		switch (param.kind)
		{
		case ParameterKind::raw:
			param_list += param.type;
			break;
		case ParameterKind::in:
			param_list += "parallel<const " + param.type + "> ";
			break;
		case ParameterKind::out:
		case ParameterKind::inout:
			param_list += "parallel<" + param.type + "> ";
			break;
		default:
			assert(false);
		}
	}

	return param_list;
}
} // namespace

hops::ParallelKernel::ParallelKernel(hops::Signature const &signature,
                                     std::string_view source_fragment,
                                     std::string_view func_name)
    : signature_(signature)
{
	auto type_list = make_type_list(signature);

	auto source = std::format(R"raw(


template <class T> class parallel
{{
	T *data_ = nullptr;
	ptrdiff_t stride_x_ = 0, stride_y_ = 0, stride_z_ = 0;

  public:
    // access the part of the array that belongs to the current thread
	T &operator()(dim3 tid) const
	{{
		return data_[tid.x * stride_x_ + tid.y * stride_y_ + tid.z * stride_z_];
	}}
}};

{}

template<class T>
T& get_elem(parallel<T> p, dim3 tid)
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

	std::string kernel_name = fmt::format("hops_kernel<{}>", type_list);
	instance_ = RawKernel(source, "parallel_kernel.cu", kernel_name);
}
