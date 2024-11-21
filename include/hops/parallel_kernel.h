#pragma once

#include "hops/base.h"
#include "hops/raw_kernel.h"
#include "hops/view.h"
#include <cassert>
#include <format>
#include <string>
#include <string_view>

namespace hops {

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
	RawKernel instance_;
	Signature signature_;

  public:
	ParallelKernel(Signature const &signature, std::string_view source_fragment,
	               std::string_view func_name);

	template <class... Args> void launch(dim3 size, Args... args)
	{
		// TODO: check argument types...

		// automatic block size should be a lot more sophisticated...
		auto block = dim3(256);

		dim3 grid;
		grid.x = (size.x + block.x - 1) / block.x;
		grid.y = (size.y + block.y - 1) / block.y;
		grid.z = (size.z + block.z - 1) / block.z;

		instance_.launch<dim3, Args...>(grid, block, size, args...);
	}
};

} // namespace hops