#include "hops/raw_kernel.h"

#include "fmt/format.h"
#include <cassert>
#include <format>

namespace {
using namespace hops;
// RAII wrapper and some conveneince functions for nvrtcProgram
//   * with a bit of cleanup, could make this part hops' public API, but wasnt
//     needed so far.
class NvrtcProgram
{
	nvrtcProgram prog_ = {};
	std::string source_;

  public:
	NvrtcProgram(std::string_view source, std::string const &filename)
	    : source_(source)
	{
		int include_count = 1;
		char const *include_names[] = {"hops_cuda.h"};
		char const *include_sources[] = {internal::cuda_library_source};
		check(nvrtcCreateProgram(&prog_, source_.c_str(), filename.c_str(),
		                         include_count, include_sources,
		                         include_names));
	}

	~NvrtcProgram() { nvrtcDestroyProgram(&prog_); }
	NvrtcProgram(NvrtcProgram const &) = delete;
	NvrtcProgram &operator=(NvrtcProgram const &) = delete;
	nvrtcProgram get() const { return prog_; }

	void compile()
	{
		// NOTE:
		//   * `-default-device` removes the need to write `__device__` in front
		//     of every function.
		//   * without `--no-source-include`, NVRTC includes the working
		//     directory of the hops executable as an include path. Quite sure
		//     we dont want that. Btw: this is not specific to NVRTC alone. also
		//     NVCC by default includes the path of any source file as an
		//     include path, which is kinda quirky.
		char const *ops[] = {"-default-device", "-std=c++20",
		                     "--no-source-include"};
		auto r = nvrtcCompileProgram(prog_, 2, ops);
		if (r == NVRTC_SUCCESS)
			return;
		size_t logSize;

		check(nvrtcGetProgramLogSize(prog_, &logSize));
		std::string log;
		log.resize(logSize);
		check(nvrtcGetProgramLog(prog_, log.data()));
		throw Error("Failed to compile CUDA code. Kernel code:\n" + source_ +
		            "\n\nError:\n" + log);
	}

	std::string get_ptx()
	{
		size_t ptxSize;
		check(nvrtcGetPTXSize(prog_, &ptxSize));
		std::string ptx;
		ptx.resize(ptxSize);
		check(nvrtcGetPTX(prog_, ptx.data()));
		return ptx;
	}
};

} // namespace

hops::RawKernel::RawKernel(std::string const &source,
                           std::string const &filename,
                           std::string const &kernel_name)
{
	// create/compile
	// fmt::print("source:\n{}\n", source);
	auto prog = NvrtcProgram(source, filename);
	check(nvrtcAddNameExpression(prog.get(), std::string(kernel_name).c_str()));
	prog.compile();
	char const *lowered_kernel_name_;
	check(nvrtcGetLoweredName(prog.get(), kernel_name.c_str(),
	                          &lowered_kernel_name_));

	// load the program (depending on CUDA settings, actual loading to GPU might
	// be deferred further, but thats transparent to us)
	auto ptx = prog.get_ptx();
	// fmt::print("PTX:\n{}\n", ptx);
	check(cuLibraryLoadData(&lib_, ptx.c_str(), nullptr, nullptr, 0, nullptr,
	                        nullptr, 0));

	// get the kernel
	check(cuLibraryGetKernel(&f_, lib_, lowered_kernel_name_));
}

// hops header code included in every CUDA kernel.
//   * ideally, this would be in a separate file ("hops_cuda.h" or something),
//     but neither C++ nor CMake make it really convenient to include a file as
//     binary resource, so we just put it into a string literal here.
char const *hops::internal::cuda_library_source = R"raw(

namespace hops
{

template<class T>
struct complex
{
  T re;
  T im;

  complex() = default;
  complex(T re, T im) : re(re), im(im) {}

  T real() const { return re; }
  T imag() const { return im; }
};

template<class T> complex<T>  operator- (complex<T> a)               { return {-a.re, -a.im}; }
template<class T> complex<T>  conj      (complex<T> a)               { return {a.re, -a.im}; }
template<class T> complex<T>  adj       (complex<T> a)               { return {a.re, -a.im}; }
template<class T> T           norm2     (complex<T> a)               { return a.re * a.re + a.im * a.im; }

template<class T> complex<T>  operator+ (complex<T> a, complex<T> b) { return {a.re + b.re, a.im + b.im}; }
template<class T> complex<T>  operator- (complex<T> a, complex<T> b) { return {a.re - b.re, a.im - b.im}; }
template<class T> complex<T>  operator* (complex<T> a, complex<T> b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
template<class T> complex<T>  operator/ (complex<T> a, complex<T> b) { return a * conj(b) / norm2(b); }

template<class T> complex<T>& operator+=(complex<T>& a, complex<T> b) { a.re += b.re; a.im += b.im; return a; }
template<class T> complex<T>& operator-=(complex<T>& a, complex<T> b) { a.re -= b.re; a.im -= b.im; return a; }
template<class T> complex<T>& operator*=(complex<T>& a, complex<T> b) { a = a * b; return a; }
template<class T> complex<T>& operator/=(complex<T>& a, complex<T> b) { a = a / b; return a; }

template<class T> complex<T>  operator+ (complex<T> a, T b)          { return {a.re + b, a.im}; }
template<class T> complex<T>  operator+ (T a, complex<T> b)          { return {a + b.re, b.im}; }
template<class T> complex<T>  operator- (complex<T> a, T b)          { return {a.re - b, a.im}; }
template<class T> complex<T>  operator- (T a, complex<T> b)          { return {a - b.re, -b.im}; }
template<class T> complex<T>  operator* (complex<T> a, T b)          { return {a.re * b, a.im * b}; }
template<class T> complex<T>  operator* (T a, complex<T> b)          { return {a * b.re, a * b.im}; }
template<class T> complex<T>  operator/ (complex<T> a, T b)          { return {a.re / b, a.im / b}; }
template<class T> complex<T>  operator/ (T a, complex<T> b)          { return a * conj(b) / norm2(b); }

template<class T> complex<T>& operator+=(complex<T>& a, T b)         { a.re += b; return a; }
template<class T> complex<T>& operator-=(complex<T>& a, T b)         { a.re -= b; return a; }
template<class T> complex<T>& operator*=(complex<T>& a, T b)         { a.re *= b; a.im *= b; return a; }
template<class T> complex<T>& operator/=(complex<T>& a, T b)         { a.re /= b; a.im /= b; return a; }



// x,y,z = parallel strides
// c = complex stride (0 for real data)
// h,w = matrix strides (0 for scalar data)
template<class,int,int,int,int,int,int> struct strided;

template <int x, int y, int z> struct strided<float, x, y, z, 0, 0, 0> { float *data_; };
template <int x, int y, int z> struct strided<double, x, y, z, 0, 0, 0> { double *data_; };
template <int x, int y, int z, int c> struct strided<complex<float>, x, y, z, c, 0, 0> { float *data_; };
template <int x, int y, int z, int c> struct strided<complex<double>, x, y, z, c, 0, 0> { double *data_; };

// single value
template<class T>
T read(T a, dim3)
{
  return a;
}

// scalar real
template<class T, int x, int y, int z>
T read(strided<T,x,y,z,0,0,0> a, dim3 tid)
{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  return ptr[0];
}
template<class T, int x, int y, int z>
void write(strided<T,x,y,z,0,0,0> a, dim3 tid, T value)
{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  ptr[0] = value;
}

// scalar complex
template<class T, int x, int y, int z, int c>
complex<T> read(strided<complex<T>,x,y,z,c,0,0> a, dim3 tid)
{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  return {ptr[0], ptr[c]};
}
template<class T, int x, int y, int z, int c>
void write(strided<complex<T>,x,y,z,c,0,0> a, dim3 tid, complex<T> value)
{
  T* ptr = a.data_ + tid.x * x + tid.y * y + tid.z * z;
  ptr[0] = value.real();
  ptr[c] = value.imag();
}

} // namespace hops
)raw";
