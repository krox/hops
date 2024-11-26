#include "hops/arithmetic.h"

#include "hops/parallel_kernel.h"

void hops::mul(View out, double alpha, ConstView a, ConstView b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.precision() == a.precision());
	assert(out.precision() == b.precision());

	auto source = "void func(auto& out, auto alpha, auto a, auto b)"
	              "{ out = alpha * a * b; }";
	static auto kernel = hops::ParallelKernel(source, "func");

	kernel.launch(out, alpha, a, b);
}
void hops::add_mul(View out, double alpha, ConstView a, ConstView b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.precision() == a.precision());
	assert(out.precision() == b.precision());

	auto source = "void func(auto& out, auto alpha, auto a, auto b)"
	              "{ out += alpha * a * b; }";
	static auto kernel = hops::ParallelKernel(source, "func");

	kernel.launch(out, alpha, a, b);
}
