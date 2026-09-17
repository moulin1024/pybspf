# Architecture

The numerical model fits a constrained B-spline, subtracts it from the samples,
and applies Fourier differentiation or integration to the residual.

```text
User API:  pybspf / operators / problem-specific solvers
                 |
Operations: differentiation / interpolation / integration
                 |
Shared fitting: ops/_common.py -> kkt.py
                 |
Primitives: backend / grid / knots / basis / boundary
```

`BSPF1D` composes grid metadata, spline evaluations, endpoint stencils, and a
cached KKT factorization. Its methods bind package-owned operation functions
inside the class definition. `BSPF2D` composes two 1D operators and performs
matrix solves along each axis. Piecewise operators stitch independent segments.

`ops/_common.py` owns sample shape/device/dtype validation and the constrained
spline solve. Fitting, differentiation, and integration share it. Batched
derivatives use matrix right-hand sides and FFTs along the sample axis, avoiding
a Python loop over signals. Real FFTs handle real data; complex signals use the
full frequency vector on both backends.

`backend.py` owns optional CuPy imports and explicit device checks. Construction
may upload geometry; computation rejects mixed backends. No machine-specific
CUDA environment setup belongs in the package. CPU-only operations reject GPU
mode explicitly rather than attempting implicit NumPy conversion.

Basis derivative matrices on the fixed operator grid are cached by order.
Arbitrary evaluation grids are evaluated afresh: caching by their first
coordinate was incorrect because different grids commonly share an endpoint.
KKT factorizations remain cached by regularization parameter.

Core imports are independent of the root compatibility shim and `legacy/`.
Solver exports are lazy, and optional FEM dependencies are imported through
individual solver modules. Existing research workflows remain in place; their
further separation is described in the [assessment](assessment.md).
