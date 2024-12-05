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



// first 3 axes are parallel along CUDA's grid/block dimensions. Rest is 'inner' indices, depending on the type 'T'.
// IMPORTANT: the various 'strided' classes may only contain a single pointer member, as they are passed to the kernel as raw pointers from C++.
// special case: the no-strides case is in fact a scalar float/double.
template<class,int...> struct strided;

template<class T> struct strided<T>
{
  T data_;
  
  T read(dim3) const { return data_; }
  // note: no 'write' function. Scalar output from parallel kernel is not supported.
};

template <class T, int sx, int sy, int sz> struct strided<T, sx, sy, sz>
{
	T* data_;

	T read(dim3 tid) const
	{
		return data_[tid.x * sx + tid.y * sy + tid.z * sz];
	}

	void write(dim3 tid, T value)
	{
		data_[tid.x * sx + tid.y * sy + tid.z * sz] = value;
	}
};

template <class T, int sx, int sy, int sz, int sc> struct strided<complex<T>, sx, sy, sz, sc>
{
	T* data_;

	complex<T> read(dim3 tid) const
	{
		T* ptr = data_ + tid.x * sx + tid.y * sy + tid.z * sz;
		return {ptr[0], ptr[sc]};
	}

	void write(dim3 tid, complex<T> value)
	{
		T* ptr = data_ + tid.x * sx + tid.y * sy + tid.z * sz;
		ptr[0] = value.real();
		ptr[sc] = value.imag();
	}
};

} // namespace hops
)raw";
