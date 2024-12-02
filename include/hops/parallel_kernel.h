#pragma once

#include "fmt/format.h"
#include "hops/base.h"
#include "hops/raw_kernel.h"
#include "hops/view.h"
#include <cassert>
#include <format>
#include <string>
#include <string_view>

namespace hops {

// only checks and normalizes the 'shape' part, does not look at types
// (complexity, precision, tensor shape)
inline dim3 unify_shapes(std::vector<Layout *> const &args)
{
	assert(args.size() > 0);
	auto shape = args[0]->shape();
	for (int i = 0; i < args.size(); ++i)
		assert(args[i]->shape() == shape);
	assert(shape.ndim() <= 3);
	auto dim = dim3(1, 1, 1);
	if (shape.ndim() >= 1)
		dim.x = shape[0];
	if (shape.ndim() >= 2)
		dim.y = shape[1];
	if (shape.ndim() >= 3)
		dim.z = shape[2];
	return dim;
}

template <class T> struct is_view : std::false_type
{};
template <> struct is_view<View> : std::true_type
{};
template <> struct is_view<ConstView> : std::true_type
{};

namespace {
template <class... Args>
std::vector<Layout *> collect_parallel_args(Args... args)
{
	std::vector<Layout *> r;

	// sensible code:
	// for(arg : args)
	//   static_if(is_view<arg>::value)
	//	   r.push_back(&arg);

	// actually working C++ code:
	(
	    [&]() {
		    if constexpr (is_view<Args>::value)
			    r.push_back(&args);
	    }(),
	    ...);
	return r;
}

} // namespace

RawKernel make_parallel_kernel(std::string_view source_fragment,
                               std::string_view func_name,
                               std::span<const std::string> type_list);

// returns things like `strided<float, 1,0,0>`
std::string make_cuda_type(Layout const &layout);

// Category of kernels that simply execute code on each GPU thread in 3D
// grid in parallel.
//   * automates some boilerplate in the cuda code
//   * effectively type-safe, as the kernel signature is generated
//   automatically
//   * gridSize/blockSize are automatic, '.launch(...)' just takes a total
//   size
//   * TODO:
//       * more sophisticated block size selection
//       * multiple variants of a kernel in a single class
//         (e.g. float/double, 1-3 dimensions, fixed strides)
class ParallelKernel
{
	std::string source_fragment_;
	std::string func_name_;

	template <class... Args> void launch_impl(dim3 size, Args... args)
	{
		std::vector<std::string> arg_types;
		std::vector<KernelArgument> arg_values;
		(
		    [&]() {
			    if constexpr (std::is_same_v<Args, View> ||
			                  std::is_same_v<Args, ConstView>)
			    {
				    arg_types.push_back(make_cuda_type(args));
				    arg_values.push_back(
				        {.ptr = const_cast<void *>(args.data())});
			    }
			    else if constexpr (std::is_same_v<Args, float>)
			    {
				    arg_types.push_back("float");
				    arg_values.push_back({.f32 = args});
			    }
			    else if constexpr (std::is_same_v<Args, double>)
			    {
				    arg_types.push_back("double");
				    arg_values.push_back({.f64 = args});
			    }
			    else
			    {
				    assert(false);
			    }
		    }(),
		    ...);

		auto instance =
		    make_parallel_kernel(source_fragment_, func_name_, arg_types);

		// automatic block size should be a lot more sophisticated...
		auto block = dim3(256);

		dim3 grid;
		grid.x = (size.x + block.x - 1) / block.x;
		grid.y = (size.y + block.y - 1) / block.y;
		grid.z = (size.z + block.z - 1) / block.z;

		std::vector<void *> arg_ptrs;
		arg_ptrs.push_back(&size);
		for (auto &arg : arg_values)
			arg_ptrs.push_back(&arg);

		instance.launch_raw(grid, block, arg_ptrs.data());
	}

  public:
	explicit ParallelKernel(std::string_view source_fragment,
	                        std::string_view func_name)
	    : source_fragment_(source_fragment), func_name_(func_name)
	{}

	template <class... Args> void launch(Args... args)
	{
		std::vector<Layout *> parallel_args = collect_parallel_args(args...);
		auto dim = unify_shapes(parallel_args);
		launch_impl(dim, args...);
	}
};

} // namespace hops