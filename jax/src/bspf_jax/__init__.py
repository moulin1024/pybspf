"""Functional B-spline + Fourier calculus for JAX, in one to three dimensions.

The package does not modify JAX precision or device configuration at import.
"""
from .basis import basis_matrix, open_knots, spline_primitive
from .plans import Plan1D, TensorPlan, plan_1d, plan_2d, plan_3d, tensor_plan, with_regularization
from .time_integration import rk4_step, integrate_rk4, integrate_linear_midpoint, integrate_schrodinger, integrate_nlse
from .galerkin import Galerkin1D, galerkin_1d
from .elasticity import elastic_modes, integrate_elastic
from .vlasov_poisson import (VlasovPoissonPlan, poisson_dirichlet, plan_vlasov_poisson,
                             vlasov_poisson_fields, integrate_vlasov_poisson)
from .parallel_kinetic import ParallelKineticPlan, plan_parallel_kinetic, integrate_parallel_kinetic
from .alfven import AlfvenPlan, plan_alfven, integrate_alfven, alfven_energy, alfven_boundary_power
from .sine_gordon import integrate_sine_gordon
from .kdv import KdVPlan, plan_kdv, integrate_kdv
from .operators import (endpoint_jets, Split, decompose, derivatives, differentiate, mixed_partial,
                        gradient, divergence, curl, hessian, laplacian, tensor_decompose,
                        noise_diagnostics)
from .calculus import interpolate, interpolate_grid, integrate, integrate_box, antiderivative

__all__ = [
    "VlasovPoissonPlan", "poisson_dirichlet", "plan_vlasov_poisson",
    "vlasov_poisson_fields", "integrate_vlasov_poisson",
    "ParallelKineticPlan", "plan_parallel_kinetic", "integrate_parallel_kinetic",
    "AlfvenPlan", "plan_alfven", "integrate_alfven", "alfven_energy", "alfven_boundary_power",
    "integrate_sine_gordon",
    "elastic_modes", "integrate_elastic",
    "KdVPlan", "plan_kdv", "integrate_kdv",
    "noise_diagnostics", "Galerkin1D", "galerkin_1d", "integrate_linear_midpoint", "integrate_schrodinger", "integrate_nlse",
    "endpoint_jets", "rk4_step", "integrate_rk4",
    "Plan1D", "TensorPlan", "Split", "plan_1d", "plan_2d", "plan_3d", "tensor_plan",
    "with_regularization", "basis_matrix", "open_knots", "spline_primitive",
    "decompose", "tensor_decompose", "differentiate", "derivatives", "mixed_partial",
    "gradient", "divergence", "curl", "hessian", "laplacian", "interpolate",
    "interpolate_grid", "integrate", "integrate_box", "antiderivative",
]
