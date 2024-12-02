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

	// note: on the cuda side, mixed precision is generally not allowed, which
	// is on purpose.
	if (out.precision() == Precision::float32)
		kernel.launch(out, static_cast<float>(alpha), a, b);
	else
		kernel.launch(out, alpha, a, b);
}

void hops::add_mul(View out, double alpha, ConstView a, ConstView b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.precision() == a.precision());
	assert(out.precision() == b.precision());

	auto source = "void func(auto& out, float alpha, auto a, auto b)"
	              "{ out += alpha * a * b; }";
	static auto kernel = hops::ParallelKernel(source, "func");

	if (out.precision() == Precision::float32)
		kernel.launch(out, static_cast<float>(alpha), a, b);
	else
		kernel.launch(out, alpha, a, b);
}
