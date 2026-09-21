# Archived NumPy/CuPy pybspf 0.1

Unmaintained snapshot of commit `361a39593de3ee6656b00f52cdd0b778a58280c5`.
Do not install this package alongside the maintained JAX `pybspf` 0.2 package.
No default CI, wheel or source distribution of the maintained packages includes it.

In a separate virtual environment, from this directory:

```sh
python -m pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 python -m pytest
python generate_reference.py --out /tmp/legacy-reference
```

The original collection references an absent `examples.navier_stokes` module.
Excluding that test, the pre-migration run had 93 passed, 3 skipped and four known
Poisson solver failures (two hybrid DST contracts, boundary-corrector gauge,
and missing `solve_fft_corrected_02`). This archive does not repair those issues.
GPU behavior was not validated in the migration environment.

The earlier monolithic implementations remain one directory above this project.
The archived shim locates them for explicit legacy regression runs. Original
API documentation and installation extras are preserved in `README.original.md`
and `docs/`. Reference generation belongs only in this isolated environment.
