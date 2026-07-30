# AMGX-MUSA

AMGX-MUSA is the Moore Threads MUSA port of NVIDIA AmgX, a GPU-accelerated
algebraic multigrid and preconditioned iterative linear solver library.

This branch is derived from upstream NVIDIA AMGX `v2.3.0`
(`32e1f44fa93af7859490a800f137e75b6513420c`) and publishes the MUSA-adapted
PH1 work (`9e62116de2ee955050c3fd4ee88e23d674a4be5e`) for Moore Threads GPUs.

## Features

AMGX-MUSA keeps the main AMGX solver model and C API while adapting the GPU
backend to the Moore Threads software stack:

* fp32, fp64 and mixed-precision solves
* Scalar and coupled block systems
* Distributed solvers using MPI
* JSON solver configuration
* Classical Ruge-Stueben and unsmoothed aggregation algebraic multigrid
* Krylov methods including CG, BiCGSTAB and GMRES
* Smoothers and preconditioners including Jacobi, Gauss-Seidel, incomplete LU
  and Chebyshev polynomial methods
* Optional eigensolver acceleration with MKL or MAGMA when available

## MUSA Port

The MUSA port replaces the original CUDA-oriented build and runtime paths with
MUSA equivalents:

* CMake configures MUSA clang from `MUSA_ROOT` and compiles device translation
  units with `-x musa --musa-path=<MUSA_ROOT>`.
* MUSA architectures are selected with `MUSA_ARCHS`, which are translated to
  `--cuda-gpu-arch=mp_<arch>` compiler options.
* CUDA runtime, Thrust, cuBLAS, cuSPARSE and cuSOLVER usage has been adapted to
  MUSA runtime, muThrust, muBLAS, muSPARSE and muSolver paths.
* MPI examples and multi-GPU code paths are adapted for the MUSA backend.
* A standalone DILU hotspot benchmark is provided under
  `perf/DILU_forward_1x1_kernel`.

## Dependencies

Required:

* CMake 3.18 or newer
* Moore Threads MUSA Toolkit, installed at `/usr/local/musa` by default
* A Linux C/C++ toolchain compatible with the MUSA clang frontend

Optional:

* MPI implementation such as Open MPI or MPICH for distributed examples
* MKL or MAGMA for optional eigensolver functionality

## Building

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUSA_ROOT=/usr/local/musa \
  -DMUSA_ARCHS="22"

cmake --build build -j16
```

Set `MUSA_ARCHS` to the architecture for your Moore Threads GPU. For a non-MPI
build, add:

```bash
-DCMAKE_NO_MPI=ON
```

The build produces static and shared AMGX libraries plus example binaries under
the build tree.

## Running Examples

Sample input matrix data is available in `examples/matrix.mtx`, and sample
solver configurations are available in `core/configs`.

Single-process example:

```bash
./build/examples/amgx_capi \
  -m examples/matrix.mtx \
  -c core/configs/FGMRES_AGGREGATION.json
```

MPI example:

```bash
mpirun -n 2 ./build/examples/amgx_mpi_capi \
  -m examples/matrix.mtx \
  -c core/configs/FGMRES_AGGREGATION.json
```

If shared libraries are not found at runtime, add the build or install library
directory to `LD_LIBRARY_PATH`.

## Performance Hotspot Benchmark

The PH1 branch includes a standalone benchmark for the
`DILU_forward_1x1_kernel` hotspot:

```bash
cd perf/DILU_forward_1x1_kernel
cmake -B build
cmake --build build -j16

./build/bin/dilu_forward_1x1_kernel
./build/bin/dilu_forward_1x1_kernel -f 1000
./build/bin/dilu_forward_1x1_kernel -wc 5 -tc 10
```

## Upstream Project

The original AMGX project is maintained by NVIDIA:

* Upstream repository: <https://github.com/NVIDIA/AMGX>
* Upstream base used for this port: <https://github.com/NVIDIA/AMGX/tree/v2.3.0>

## License

AMGX-MUSA is released under the BSD 3-Clause License. See `LICENSE` for the
complete Moore Threads copyright and license statement, including the upstream
AMGX license text.
