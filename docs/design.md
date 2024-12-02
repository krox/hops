# *hops* at different levels of abstraction

In the following sections we write a simple "axpy" program, using the hops library at different levels of abstraction, from low- to high-level. While this might not be correct order for a novice to learn GPU programming, it reflects closeley the design process of hops itself: start with using cuda directly, and add layers of abstractions and convenience functions step-by-step to make a users life easier and increasing expressive power.

## As a basic CUDA wrapper

Hops can be used as a simple C++ wrapper around CUDA's driver API. I.e. destructors release resources, and errors result in exceptions being thrown.

* Device memory management:
```C++
// allocate (uninitialized) device memory
int n = 10000;
auto in = hops::device_buffer<float>(n);
auto out = hops::device_buffer<float>(n);
// move data from CPU to GPU memory
in.copy_from_host(...);
```

* Create a CUDA kernel
```C++
auto source = R"raw(
extern "C" __global__
void saxpy(size_t n, float* out, float alpha, float const* in)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
    out[tid] = alpha * in[tid];
}
)raw";
auto lib = hops::CudaLibrary(source);
auto kernel = lib.get_kernel("saxpy");
```
* launch the kernel
```C++
int blockSize = 128;
int gridSize = (n + blockSize - 1) / blockSize;
hops::launch<size_t, float*, float, float const*>(kernel, gridSize, blockSize, n, out.data(), 4.2, in.data());
```

* rough mapping of `hops` functions to the CUDA driver API

| CUDA                                                                                                                               | hops                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `cuMemAlloc`                                                                                                                       | `hops::device_buffer<T>(...)`                                                                |
| `cuMemcpy{DtoH,HtoD}`                                                                                                              | `buf.copy_{from,to}_host(...)`                                                               |
| `nvrtcCreateProgram(source)` + `nvrtcCompileProgram` + `nvrtcGetPTX` + `cuLibraryLoadData` <br> +optional:`nvrtcAddNameExpression` | `hops::CudaLibrary(source)`                                                                  |
| `cuLaunchKernel[Ex](f, dims, args)`                                                                                                | either `hops::launch(f, args)` or `f.launch(args)`                                           |
| `cuMemFree`,  `cuModuleDestroy`,...                                                                                                | nothing, all handled by RAII                                                                 |
| `CUresult`, `nvrtcResult`, `cuGetErrorName`, `nvrtcGetProgramLog`,...                                                              | all checked automatically, throwing `hops::error` with reasonable message if anything fails. |

## Element-wise kernel based on `hops::ParallelKernel`

In the previous example, arguably, only two lines of the kernel code contained meaningful information. One line containing the actual arithmetic, and the list of parameters to make it work. Everything else can (and should?) be hidden using the `ParallelKernel` class like this:

```C++
auto source = "void func(float& out, float a, float b)
               {
                   out = a * b;
               }";

auto kernel = hops::ParallelKernel(source, "func");
kernel.launch(out.view(), 4.2f, in.view());
```

Some explanations:
  * Note that the the definition of the kernel does not distinguish between scalar and array parameters. It cann be called with any combination is is instantiated and compiled on demand.

## multi-dimensional arrays using `hops::parallel`

  * A `hops::ParallelKernel` can be launched without any changes on an (up to) three dimensional array with arbitrary strides. Element-wise application on a 2x3x4 array can be written as:
  ```C++
  auto shape = hops::dim3(2, 3, 4);
  auto out_view = hops::parallel(out.data(), /*strides=*/1,2,6);
  auto in_view = hops::parallel(in.data(), /*strides=*/1,2,6);
  kernel.launch(shape, out_view, 4.2, in_view);
  ```
  * These three dimensions are mapped directly to the `x,y,z` dimensions in CUDA. As a matter of convention (and a rule-of-thumb for performance), the `x` dimension should be the fastest moving one. Not a hard rule of course: transposed arrays, and broadcasting axes (setting some stride to zero) are perfectly allowed.

## higher-dimensional arrays using `hops::View`

Alternative to using the low-level `hops::parallel` to launch a kernel, we can use `hops::View` like this:
  ```C++
  auto shape = hops::dim3(2, 3, 4);
  auto out_view = hops::View<float>::contiguous(out.data(), shape);
  auto in_view = hops::View<float>::contiguous(in.data(), shape);
  kernel.launch(out_view, 4.2, in_view);
  ```
* `View` can have arbitrary strides, but also offers some convenience constructors to compute them automatically, like the `::contiguous(<data>, <shape>)` function.
* `View` knows its own shape in addition to strides, so no shape needs to be passed to `.launch`. Will throw on incompatible shapes between the `View`s.
* `View` supports an arbitrary number of dimensions. The `.launch(..)` function will figure out an efficient mapping to CUDAs `x,y,z` dimension before launching the kernel. This means a user should be free to keep the axes in a form with semantic meaning (eg, `(Lorentz,X,Y,Z,T,Spin,Color)` could be used in latticeQCD), and not worry about a good mapping to threads/threadBlocks/etc at all. Implementation details:
  * size-1 dimensions are removed
  * dimensions are ordered by increasing stride
  * compatible dimensions are merged
  * If more than 3 non-trivial dimensions are left, the same kernel will be launched multiple times.
 

## using hops predefined functions

For some common procedures, hops already defines BLAS-style functions. For example

```C++
auto in = hops::View<const float>(...);
auto out = hops::View<float>(....);
hops::mul(out, 4.2, in);
```

Additionally to the previous kernel implementation this also:
* calls simplified/optimized kernels for special values like `alpha=1` or `alpha=-1`
* TODO: maybe automatic contraction when out-stride=1?
* TODO: maybe implement mixed-precision stuff at this level?

## kernels with internal indices

Lets say the element-wise function is not quite scalar, but should act on a small collection of real numbers, such as a complex. How do we want to write that?

```C++
auto source = "void func(std::complex<double>& out, std::complex<double> const& a, std::complex<double> const& b)
{
  out.real() = a.real() * b.real() - a.imag() * b.imag();
  out.imag() = a.real() * b.imag() + a.imag() * b.real();
}";

auto kernel = ParallelKernel(source, "func");
kernel.launch(buffer.view().as_complex(), ...);
```

* The `.as_complex()` method re-interprets one axis of a real view as real/imaginary components.
* Strides between real and imaginary parts can be arbitrary. The references passed as arguments to `func` are to local variables inside the kernel-wrapper function, which handles the actual memory accesses.
* By convention, the first parameter is treated as read-write, the others are only read. (Future: making this more flexible, though that would require)

### Alternative 1:
make the precise cuda signature generated:
```C++
auto kernel = ParallelKernel("Complex double& out, Complex double a, Complex double b", "out.real() = [...]");
```
* adds a bit more type-safety, which is cute.
* going back to actuallly parsing a hops-specific with "out", "inout", "raw" keywords is kinda ugly.

### Alternative 2:
Stick to actual types:
```C++
void func(std::complex<double>& out, std::complex<double> a, std::complex<double> b){...}
```

and handle the memory access in the generated wrapper function that exists anyway already. something like

```C++
__global__ void hops_kernel(parallel<double, ...>, parallel<const double, ...>, parallel<const double, ...>)
{
  std::complex<double> out;
  std::complex<double> a = arg_a(tid);
  std::complex<double> b = arg_b(tid);

  func(out, a, b);

  arg_out(tid) = out;
}
```

* nice: memory access niecly in one place, regardless of the local function itself
* When instantiating, `ParallelKernel` maps the view type to its nomimal content type in cuda
* Open problem: `ParallelKernel` needs to figure out which parameters are in/out/inout
  * Variant A: use the return value for output, all arguments are input.
    * Hm, not great. inplace/"inout" operations are kinda important.
  * Variant B: parse the cuda signature. `ParallelKernel("double& a, double b, double c", "a=b*c;");` is actually nice (and saves us from giving the local function a name).
    * problem: how to distinguish "out" from "inout" parameters? both are written as `&` in C++/Cuda.
      * maybe we dont have to? rely on the cuda compiler to elide dead reads.
      * even better: by convention treat exactly the first parameter as "inout" (eliding read if it happens to overwrite exactly), rest as "in". Then we dont have to look for `&` characters in the signature. beware: dead writes (of parameters that happen to be purely "in") are not reliably elided. Especically without `__restric__` (which would be non trivial to add, since the memory access is hidden inside `parallel<...>`. Therefore do not default all parameters to "inout"/"out".
      * long term, some attempt at parsing the cuda signature still seems warranted. Though given that that the typename can include arbitrary C++ syntax, thats hard to do in the most general case. Likely, we would support a well-defined sub-set like `{float,double,Matrix<{float,double},NUMBER>}[const][&] {identifier}` or something, maybe use proper regex actually. And a mismatch could lead to "raw parameter, passed as-is, no support for 'View', must be scalar".
      * hm, well, 'auto' isnt really sufficient, as `Matrix<auto,3>` isnt valid C++/Cuda, but I dont want to parse a `template<class T> ...` in front of a full function definition. this isnt really working out.
  * Variant C: Just by convention, the first parameter is "inout", rest is "in". dead read of a pure "in" is probably elided by cuda optimizer.
    * Note: dont default everything to "inout". eliding dead writes is more tricky, especiallly since we cannot easily sprinkle `__restrict__` everywhere.
