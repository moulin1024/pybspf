# bspf-models

PDE models built on the JAX pybspf core. See the repository migration guide for the module map.

The fixed-ellipse, unsteady BSPF solver is available in
`bspf_models.fluids.embedded_navier_stokes`. Its default structural mode enforces
volume divergence and normal-flux continuity separately from scalar-pressure
recovery, with independent quadrature checks at initialization and every step.
JAX BE/BDF2-AB2 stepping supports checkpointable state and natural channel outflow.
The default dense reference setup is limited to 2048 velocity DOFs. Explicit
`constraint_backend="implicit_qr", linear_backend="host_sparse"` uses local polynomial-preserving
divergence elimination, sparse implicit QR, and SVD of a bounded dense range
core (512 MB core cap, plus factorization workspace); this optional large-case CPU backend is not JIT-compatible. Numerical rank
sensitivity still limits accuracy claims. The previous pressure-stabilized method
requires explicit `incompressibility="stabilized"`.
Geometry and factorization are host setup. See [the solver guide](../../docs/jax_embedded_navier_stokes.md)
for examples, validation, and current size/performance limits.

For structural obstacle no-slip, select `wall_enforcement="constraint"`. This
adds tangential wall constraints with an independent per-step acceptance audit;
`wall_slip_error(coefficients)` reports the obstacle tangential L2 mismatch.
It reduces the admissible velocity space, so compare full-field accuracy as well
as slip. The thin-airfoil runner uses this option; see the guide for the measured
accuracy tradeoff and short-time validation.

For repeated CPU steps at moderate sizes, `projector_backend="array"` caches the
local-coordinate null basis (256 MiB cap) and removes runtime QR-worker calls.
It also exposes pure FP64 JAX projection actions for a future resident GPU loop;
the complete integrator and sparse-LU preconditioner currently remain on CPU.
See the guide for paired timing, additional memory and scaling limitations.
