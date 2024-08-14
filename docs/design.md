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

## Custom kernel based on `hops::ParallelKernel`

In the previous example, arguably, only two lines of the kernel code contained meaningful information. One line containing the actual arithmetic, and the list of parameters to make it work. Everything else can (and should?) be hidden using the `ParallelKernel` class like this:

```C++
auto source = "*out = alpha * *in;";
auto kernel = ParallelKernel<hops::parallel<float>, float, hops::parallel<const float>>>(source, {"out", "alpha", "in"});

kernel.launch(n, out.data(), 4.2, in.data());
```

Some explanations:
  * In the basic usecase here, the parameter type `hops::parallel` is essentially just a pointer, which - inside a kernel function - automatically dereferences to the array element belonging to the current GPU thread. no explicit "`array[tid]`" indexing necessary.
  * `blockSize`/`gridSize` are chosen automatically, the `kernel.launch` function only takes the overall size of the parallel launch as parameter. (TODO: actually be a bit smart about it, current implementation is quite naive...)
  * Nice bonus: a little bit of templating is quite easy at this point:
  ```C++
  // works for T=float and T=double. Other types probably need more work, see later sections...
  template<class T>
  auto kernel = ParallelKernel<paralllel<T>, T, parallel<const T>>(...);
  ```

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
