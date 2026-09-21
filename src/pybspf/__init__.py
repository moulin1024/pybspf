"""JAX BSPF calculus. Importing does not alter precision or device settings."""
from .fast_axis import FastAxis, plan_fast_axis, sample_aligned_knots, boundary_clustered_knots
from .basis import basis_matrix, open_knots, spline_primitive
from .plans import Plan1D, TensorPlan, plan_1d, plan_2d, plan_3d, tensor_plan, with_regularization
from .time_integration import rk4_step, integrate_rk4, integrate_linear_midpoint
from .galerkin import Galerkin1D, galerkin_1d
from .operators import endpoint_jets, Split, decompose, derivatives, differentiate, mixed_partial, gradient, divergence, curl, hessian, laplacian, tensor_decompose, noise_diagnostics
from .calculus import interpolate, interpolate_grid, integrate, integrate_box, antiderivative
__all__ = ['FastAxis', 'FourierExtension', 'FourierExtensionPlan', 'Galerkin1D', 'Plan1D', 'Split', 'TensorPlan', 'antiderivative', 'basis_matrix', 'boundary_clustered_knots', 'curl', 'decompose', 'derivatives', 'differentiate', 'divergence', 'endpoint_jets', 'galerkin_1d', 'gradient', 'hessian', 'integrate', 'integrate_box', 'integrate_linear_midpoint', 'integrate_rk4', 'interpolate', 'interpolate_grid', 'laplacian', 'mixed_partial', 'noise_diagnostics', 'open_knots', 'plan_1d', 'plan_2d', 'plan_3d', 'plan_fast_axis', 'rk4_step', 'sample_aligned_knots', 'spline_primitive', 'tensor_decompose', 'tensor_plan', 'with_regularization']

def __getattr__(name):
    if name in ("FourierExtensionPlan", "FourierExtension"):
        from . import fourier_extension
        return getattr(fourier_extension, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
