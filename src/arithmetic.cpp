#include "hops/arithmetic.h"

#include "hops/kernel.h"

namespace {
auto mod = hops::CudaModule(
    R"raw(template <class T> struct ptr
{
	T *ptr = nullptr;
	ptrdiff_t stride = 1;

    T& operator()(size_t i) const
    {
        return ptr[i * stride];
    }
};

template<bool accumulate, class T>
__global__
void mul(ptr<T> out, size_t n, T alpha, ptr<const T> a, ptr<const T> b)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n)
    {
        if constexpr(accumulate)
            out(x) += alpha * a(x) * b(x);
        else
            out(x) = alpha * a(x) * b(x);
    }
}
)raw",
    "arithmetic.cu", {},
    std::vector<std::string>{"mul<{true,false},{float,double}>"});
}

namespace {
std::string typestr(float const &) { return "float"; }
std::string typestr(double const &) { return "double"; }
} // namespace

// out = alpha * a * b
template <bool accumulate, class T>
void hops::mul(DevicePtr<T> out, size_t n, T alpha, DevicePtr<const T> a,
               DevicePtr<const T> b)
{

	auto name = std::format("mul<{},{}>", accumulate ? "true" : "false",
	                        typestr(alpha));
	static CUfunction f = mod.get_function(name);
	size_t nThreads = 128;
	size_t nBlocks = (n + nThreads - 1) / nThreads;
	void *args[] = {&out, &n, &alpha, &a, &b};

	check(cuLaunchKernel(f, nBlocks, 1, 1, // grid dim
	                     nThreads, 1, 1,   // block dim
	                     0, nullptr,       // shared mem, stream
	                     args, 0));        // arguments
}

// instantiate the template functions
template void hops::mul<true, float>(DevicePtr<float>, size_t, float,
                                     DevicePtr<const float>,
                                     DevicePtr<const float>);
template void hops::mul<true, double>(DevicePtr<double>, size_t, double,
                                      DevicePtr<const double>,
                                      DevicePtr<const double>);
template void hops::mul<false, float>(DevicePtr<float>, size_t, float,
                                      DevicePtr<const float>,
                                      DevicePtr<const float>);
template void hops::mul<false, double>(DevicePtr<double>, size_t, double,
                                       DevicePtr<const double>,
                                       DevicePtr<const double>);