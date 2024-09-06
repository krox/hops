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

	if (out.rank() != 1)
		throw std::runtime_error("Not implemented yet");
	auto type = type_string(T{});

	static auto kernel = [&]() {
		auto signature = hops::Signature()
		                     .out(type, "out")
		                     .raw(type, "alpha")
		                     .in(type, "a")
		                     .in(type, "b");
		auto source = "*out = alpha * *a * *b;";
		return hops::ParallelKernel(signature, source);
	}();

	kernel.launch(out.size(), parallel<T>{out.data(), out.stride(0), 0, 0},
	              alpha, parallel<const T>{a.data(), a.stride(0), 0, 0},
	              parallel<const T>{b.data(), b.stride(0), 0, 0});
}
template <class T>
void hops::add_mul(View<T> out, T alpha, View<const T> a, View<const T> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.shape() == a.shape());
	assert(out.shape() == b.shape());

	if (out.rank() != 1)
		throw std::runtime_error("Not implemented yet");
	auto type = type_string(T{});

	static auto kernel = [&]() {
		auto signature = hops::Signature()
		                     .inout(type, "out")
		                     .raw(type, "alpha")
		                     .in(type, "a")
		                     .in(type, "b");
		auto source = "*out += alpha * *a * *b;";
		return hops::ParallelKernel(signature, source);
	}();

	kernel.launch(out.size(), parallel<T>{out.data(), out.stride(0), 0, 0},
	              alpha, parallel<const T>{a.data(), a.stride(0), 0, 0},
	              parallel<const T>{b.data(), b.stride(0), 0, 0});
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