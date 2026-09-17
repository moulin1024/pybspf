# Project-local corrected OpenBLAS (macOS ARM64)

This build fixes OpenBLAS 0.3.32's ARM64/Clang C11-atomics detection and keeps
OpenMP enabled. It is installed under `build/openblas-atomic/install`, an ignored
local build directory. The shared conda libraries are not replaced.

## Source correction

In `c_check`, the ARM64 Clang-version check overwrites `data`, which previously
held preprocessed compiler-feature markers. The subsequent C11 check therefore
loses `HAVE_C11`. The [two-line patch](diagnostics/openblas-0.3.32-c11.patch)
uses a separate `clang_version` variable. With the same Clang 21.1.8 compiler and
`-std=gnu11`, the original script omits `HAVE_C11` and the patched script emits it.

This enables the existing atomic scratch-buffer reservation in the OpenMP
server. `NUM_PARALLEL=4` provides four independently reserved buffer sets; merely
increasing that number without fixing synchronization would not fix the race.

## Reproduce this machine's build

Run from the repository root. Requires the installed Homebrew LLVM compiler
and the existing conda LLVM OpenMP runtime. Uses the bundled C translation of
LAPACK (`NOFORTRAN=1`) to avoid adding a Fortran/OpenMP runtime dependency.

```sh
curl -L --fail https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.32.tar.gz -o /tmp/openblas-0.3.32.tar.gz
shasum -a 256 /tmp/openblas-0.3.32.tar.gz
# f8a1138e01fddca9e4c29f9684fd570ba39dedc9ca76055e1425d5d4b1a4a766
mkdir -p build/openblas-atomic
tar -xzf /tmp/openblas-0.3.32.tar.gz -C build/openblas-atomic
patch -d build/openblas-atomic/OpenBLAS-0.3.32 -p1 < docs/diagnostics/openblas-0.3.32-c11.patch
```

Use these same arguments for both `make -j8` and `make install`, from the source
directory. `PREFIX` is the absolute project-local install directory.

```sh
make -j8 \
  CC=/opt/homebrew/opt/llvm/bin/clang HOSTCC=/opt/homebrew/opt/llvm/bin/clang \
  AR=/usr/bin/ar RANLIB=/usr/bin/ranlib \
  NOFORTRAN=1 TARGET=VORTEX DYNAMIC_ARCH=0 BINARY=64 \
  USE_THREAD=1 USE_OPENMP=1 NUM_THREADS=128 NUM_PARALLEL=4 NO_AFFINITY=1 \
  CFLAGS='-O2 -std=gnu11' \
  LDFLAGS='-L/Users/moulin/miniforge3/lib -Wl,-rpath,/Users/moulin/miniforge3/lib' \
  PREFIX=/Users/moulin/Workspace/pybspf/build/openblas-atomic/install
```

The local `lib` directory also needs relative symlinks `libopenblas.0.dylib`,
`libblas.3.dylib`, `libcblas.3.dylib`, and `liblapack.3.dylib` pointing to
`libopenblas.dylib`, matching the names used by this conda NumPy/SciPy installation.
`make install` creates the OpenBLAS symlink. From the repository root, add:

```sh
ln -s libopenblas.dylib build/openblas-atomic/install/lib/libblas.3.dylib
ln -s libopenblas.dylib build/openblas-atomic/install/lib/libcblas.3.dylib
ln -s libopenblas.dylib build/openblas-atomic/install/lib/liblapack.3.dylib
```

## Launch

From the repository root, start a new Python process or Jupyter server:

```sh
OMP_NUM_THREADS=4 python docs/diagnostics/run_with_local_blas.py -m jupyterlab
```

The launcher prepends the local library directory to `DYLD_LIBRARY_PATH` before
executing Python. This is process-local and inherited by new notebook kernels;
existing kernels must be restarted through the new server. It defaults to four
OpenMP threads if `OMP_NUM_THREADS` is unset. CPU thread choice remains explicit
and the Python library does not alter global thread settings on import.

Launch ordinary Python without this wrapper to return to the original library;
that original build still requires `OMP_NUM_THREADS=1` for these concurrent solves.

## Validation

The direct native-call probe is independent of JAX and SciPy:

```sh
OMP_NUM_THREADS=4 python docs/diagnostics/run_with_local_blas.py \
  docs/diagnostics/openblas_trsm_probe.py \
  build/openblas-atomic/install/lib/libopenblas.dylib
```

For JAX/SciPy concurrency and loaded-library inspection:

```sh
OMP_NUM_THREADS=4 python docs/diagnostics/run_with_local_blas.py docs/diagnostics/validate_local_blas.py
OMP_NUM_THREADS=8 python docs/diagnostics/run_with_local_blas.py docs/diagnostics/validate_local_blas.py
```

Both thread counts passed 50/50 direct concurrent TRSM repetitions (maximum
absolute error 1.33e-15), 100/100 JAX concurrent LU repetitions, and 100/100 SciPy
concurrent LU repetitions (maximum error 4.00e-15). The dyld image list confirms
the project-local OpenBLAS and the existing conda `libomp` are loaded, with no
original conda OpenBLAS. The corrected `_exec_blas` binary contains `casalb`
atomic compare-and-swap instructions for reserving scratch-buffer sets.

At four OpenMP threads, all 57 core, endpoint, and time-integration tests pass.
The full 3D notebook also passes, including its convergence sweep:

| Quantity | Error |
|---|---:|
| Jacobian relative L2, 33 × 41 × 49 | 1.29781e-10 |
| Curl relative L2 | 1.21264e-10 |
| Divergence maximum absolute | 5.93365e-10 |
| Laplacian relative L2 | 2.56830e-9 |
| Jacobian relative L2, 65³ convergence endpoint | 8.18837e-14 |

These match the earlier correct single-thread BLAS results while leaving
OpenMP and outer JIT enabled.

These are targeted project checks, not certification of every LAPACK routine
or every hardware/platform configuration. Build logs, configuration, binaries,
and validation JSON are retained under `build/openblas-atomic/`.
