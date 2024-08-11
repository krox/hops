#include "hops/arithmetic.h"

#include "hops/kernel.h"

template <class T>
void hops::mul(View<T> out, T alpha, View<const T> a, View<const T> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.shape() == a.shape());
	assert(out.shape() == b.shape());

	if (out.rank() == 1)
	{
		static auto kernel =
		    hops::ParallelKernel<parallel<T>, T, parallel<const T>,
		                         parallel<const T>>(
		        "*out = alpha * *a * *b;",
		        std::vector<std::string>{"out", "alpha", "a", "b"});
		kernel.launch(out.size(), {out.data(), out.stride(0), 0, 0}, alpha,
		              {a.data(), a.stride(0), 0, 0},
		              {b.data(), b.stride(0), 0, 0});
	}
	else
		throw std::runtime_error("Not implemented yet");
}

template <class T>
void hops::add_mul(View<T> out, T alpha, View<const T> a, View<const T> b)
{
	// TODO: a lot of broadcasting logic, and then axes-normalization to finally
	// dispatch to a proper kernel.
	assert(out.shape() == a.shape());
	assert(out.shape() == b.shape());

	if (out.rank() == 1)
	{
		static auto kernel =
		    hops::ParallelKernel<parallel<T>, T, parallel<const T>,
		                         parallel<const T>>(
		        "*out += alpha * *a * *b;",
		        std::vector<std::string>{"out", "alpha", "a", "b"});
		kernel.launch(out.size(), {out.data(), out.stride(0), 0, 0}, alpha,
		              {a.data(), a.stride(0), 0, 0},
		              {b.data(), b.stride(0), 0, 0});
	}
	else
		throw std::runtime_error("Not implemented yet");
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