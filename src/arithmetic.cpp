#include "hops/arithmetic.h"

#include "hops/kernel.h"

namespace {
std::string type_string(float) { return "float"; }
std::string type_string(double) { return "double"; }
} // namespace

template <class T>
void hops::mul(View<T> out, T alpha, View<const T> a, View<const T> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.shape() == a.shape());
	assert(out.shape() == b.shape());

	static auto kernel = [&]() {
		auto type = type_string(T{});
		auto signature = hops::Signature()
		                     .out(type, "out")
		                     .raw(type, "alpha")
		                     .in(type, "a")
		                     .in(type, "b");
		auto source = "out() = alpha * a() * b();";
		return hops::ParallelKernel(signature, source);
	}();

	assert(out.rank() == 1);
	kernel.launch(out.size(), out.ewise(), alpha, a.ewise(), b.ewise());
}
template <class T>
void hops::add_mul(View<T> out, T alpha, View<const T> a, View<const T> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.shape() == a.shape());
	assert(out.shape() == b.shape());

	static auto kernel = [&]() {
		auto type = type_string(T{});
		auto signature = hops::Signature()
		                     .inout(type, "out")
		                     .raw(type, "alpha")
		                     .in(type, "a")
		                     .in(type, "b");
		auto source = "out() += alpha * a() * b();";
		return hops::ParallelKernel(signature, source);
	}();

	assert(out.rank() == 1);
	kernel.launch(out.size(), out.ewise(), alpha, a.ewise(), b.ewise());
}

// instantiate the template functions
template void hops::mul<float>(View<float>, float, View<const float>,
                               View<const float>);
template void hops::mul<double>(View<double>, double, View<const double>,
                                View<const double>);
template void hops::add_mul<float>(View<float>, float, View<const float>,
                                   View<const float>);
template void hops::add_mul<double>(View<double>, double, View<const double>,
                                    View<const double>);