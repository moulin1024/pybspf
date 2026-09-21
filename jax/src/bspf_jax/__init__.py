"""Functional B-spline + Fourier calculus for JAX, in one to three dimensions.

The package does not modify JAX precision or device configuration at import.
"""
from .fast_axis import FastAxis, plan_fast_axis, sample_aligned_knots, boundary_clustered_knots
from .fast_drift_kinetic import FastDriftKineticPlan
from .basis import basis_matrix, open_knots, spline_primitive
from .plans import Plan1D, TensorPlan, plan_1d, plan_2d, plan_3d, tensor_plan, with_regularization
from .time_integration import rk4_step, integrate_rk4, integrate_linear_midpoint, integrate_schrodinger, integrate_nlse
from .galerkin import Galerkin1D, galerkin_1d
from .elasticity import elastic_modes, integrate_elastic
from .vlasov_poisson import (VlasovPoissonPlan, poisson_dirichlet, plan_vlasov_poisson,
                             vlasov_poisson_fields, integrate_vlasov_poisson)
from .drift_kinetic import (DriftKineticPlan, plan_drift_kinetic, drift_kinetic_rhs,
                            drift_kinetic_moments, integrate_drift_kinetic,
                            integrate_log_drift_kinetic, log_drift_kinetic_diagnostics)
from .parallel_kinetic import ParallelKineticPlan, plan_parallel_kinetic, integrate_parallel_kinetic
from .alfven import AlfvenPlan, plan_alfven, integrate_alfven, alfven_energy, alfven_boundary_power
from .sine_gordon import integrate_sine_gordon
from .kdv import KdVPlan, plan_kdv, integrate_kdv
from .operators import (endpoint_jets, Split, decompose, derivatives, differentiate, mixed_partial,
                        gradient, divergence, curl, hessian, laplacian, tensor_decompose,
                        noise_diagnostics)
from .calculus import interpolate, interpolate_grid, integrate, integrate_box, antiderivative
from .pressure import (
    PressurePoisson2DPlan, PressurePoisson2DResult, plan_pressure_poisson2d, compress_pressure_plan,
    pressure_gradient, pressure_divergence, pressure_schur, pressure_remove_mean,
    solve_pressure_poisson2d, project_pressure2d,
)
from .pressure3d import (
    PressurePoisson3DPlan, PressurePoisson3DResult, plan_pressure_poisson3d,
    compress_pressure_plan3d, pressure_schur3d, pressure_lift3d,
    pressure_action3d, solve_pressure_poisson3d,
)
from .navier_stokes import (
    NavierStokes2DPlan, NSStageDiagnostics, plan_navier_stokes2d,
    ns_rhs, ns_raw_rhs, ns_rk4_step, ns_vorticity, kh_initial_velocity,
    ns_divergence, ns_project_velocity,
)
from .weak_navier_stokes import (
    WeakNavierStokes2DPlan, plan_weak_navier_stokes2d,
    weak_ns_project, weak_ns_divergence, weak_ns_pointwise_divergence, weak_ns_vorticity,
    weak_ns_load, weak_ns_momentum_load, weak_ns_rhs, weak_ns_rk4_step,
    weak_ns_energy, weak_ns_helmholtz, weak_kh_initial_velocity,
)
from .stream_navier_stokes import (
    StreamNavierStokes2DPlan, plan_stream_navier_stokes2d,
    stream_ns_velocity, stream_ns_vorticity, stream_ns_divergence,
    with_stream_dynamic_boundary, stream_ns_inertia_apply, stream_ns_inertia_solve,
    StreamSponge2D, plan_stream_sponge, stream_ns_sponge_load,
    stream_ns_boundary_load, stream_ns_open_velocity,
    stream_ns_load, stream_ns_rhs, stream_ns_rk4_step, stream_ns_energy,
    stream_kh_initial, stream_evaluate_line,
)
from .convex_poisson import ConvexPoissonPlan, ConvexPoissonSolution
from .convex_poisson_grid import ConvexPoissonGridPlan, ConvexPoissonGridResult
from .convex_poisson_tensor import TensorConvexPoissonPlan, PoissonIterationError
from .fourier_extension import FourierExtensionPlan, FourierExtension
from .fourier_poisson import FourierPoissonPlan, FourierPoissonSolution
from .grad_shafranov import FixedBoundaryGSPlan, FixedBoundaryGSSolution
from .gs_response import GSFixedPointResponse, GSResponseResult
from .solovev import SolovevEquilibrium, SolovevFluxDomain
from .immersed_poisson import EllipticHole, ImmersedPoissonPlan, ImmersedPoissonSolution
from .immersed_flow import ImmersedFlowPlan, ImmersedFlowStepper

__all__ = [
    "FastAxis", "plan_fast_axis", "FastDriftKineticPlan", "sample_aligned_knots", "boundary_clustered_knots",
    "FixedBoundaryGSPlan", "FixedBoundaryGSSolution",
    "GSFixedPointResponse", "GSResponseResult",
    "SolovevEquilibrium", "SolovevFluxDomain",
    "FourierExtensionPlan", "FourierExtension",
    "FourierPoissonPlan", "FourierPoissonSolution",
    "ImmersedFlowPlan", "ImmersedFlowStepper",
    "EllipticHole", "ImmersedPoissonPlan", "ImmersedPoissonSolution",
    "ConvexPoissonPlan", "ConvexPoissonSolution",
    "ConvexPoissonGridPlan", "ConvexPoissonGridResult",
    "TensorConvexPoissonPlan", "PoissonIterationError",
    "StreamNavierStokes2DPlan", "plan_stream_navier_stokes2d",
    "stream_ns_velocity", "stream_ns_vorticity", "stream_ns_divergence",
    "with_stream_dynamic_boundary", "stream_ns_inertia_apply", "stream_ns_inertia_solve",
    "StreamSponge2D", "plan_stream_sponge", "stream_ns_sponge_load",
    "stream_ns_boundary_load", "stream_ns_open_velocity",
    "stream_ns_load", "stream_ns_rhs", "stream_ns_rk4_step", "stream_ns_energy",
    "stream_kh_initial", "stream_evaluate_line",
    "WeakNavierStokes2DPlan", "plan_weak_navier_stokes2d",
    "weak_ns_project", "weak_ns_divergence", "weak_ns_pointwise_divergence", "weak_ns_vorticity",
    "weak_ns_load", "weak_ns_momentum_load", "weak_ns_rhs", "weak_ns_rk4_step",
    "weak_ns_energy", "weak_ns_helmholtz", "weak_kh_initial_velocity",
    "PressurePoisson3DPlan", "PressurePoisson3DResult", "plan_pressure_poisson3d",
    "compress_pressure_plan3d", "pressure_schur3d", "pressure_lift3d",
    "pressure_action3d", "solve_pressure_poisson3d",
    "NavierStokes2DPlan", "NSStageDiagnostics", "plan_navier_stokes2d",
    "ns_rhs", "ns_raw_rhs", "ns_rk4_step", "ns_vorticity", "kh_initial_velocity",
    "ns_divergence", "ns_project_velocity",
    "PressurePoisson2DPlan", "PressurePoisson2DResult", "plan_pressure_poisson2d", "compress_pressure_plan",
    "pressure_gradient", "pressure_divergence", "pressure_schur", "pressure_remove_mean",
    "solve_pressure_poisson2d", "project_pressure2d",
    "VlasovPoissonPlan", "poisson_dirichlet", "plan_vlasov_poisson",
    "vlasov_poisson_fields", "integrate_vlasov_poisson",
    "DriftKineticPlan", "plan_drift_kinetic", "drift_kinetic_rhs",
    "drift_kinetic_moments", "integrate_drift_kinetic",
    "integrate_log_drift_kinetic", "log_drift_kinetic_diagnostics",
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

from .gyrokinetic_slab import (
    SlabGKPlan, plan_slab_gk, slab_gk_project, slab_gk_fields,
    slab_gk_rhs, slab_gk_diagnostics, integrate_slab_gk,
)
__all__ += ["SlabGKPlan", "plan_slab_gk", "slab_gk_project", "slab_gk_fields",
            "slab_gk_rhs", "slab_gk_diagnostics", "integrate_slab_gk"]
from .gyrokinetic_slab import slab_gk_source_rates
from .gyrokinetic_mms import SlabMMS, plan_slab_mms
__all__ += ["slab_gk_source_rates", "SlabMMS", "plan_slab_mms"]

from .gyrokinetic_slab import slab_gk_solve_charge, slab_gk_rhs_with_field_hat
__all__ += ["slab_gk_solve_charge", "slab_gk_rhs_with_field_hat"]

from .open_slab_packet import (
    OpenSlabPacket, plan_open_slab_packet, packet_initial, packet_reference,
    packet_reference_moments, packet_fields, packet_rhs, packet_diagnostics,
    integrate_open_packet,
)
__all__ += ["OpenSlabPacket", "plan_open_slab_packet", "packet_initial",
            "packet_reference", "packet_reference_moments", "packet_fields",
            "packet_rhs", "packet_diagnostics", "integrate_open_packet"]

from .linear_itg import (
    ITGRadial, LinearITG, plan_itg_radial, plan_linear_itg,
    linear_itg_fields, linear_itg_rhs, linear_itg_initial,
    linear_itg_diagnostics, integrate_linear_itg,
)
__all__ += ["ITGRadial", "LinearITG", "plan_itg_radial", "plan_linear_itg",
            "linear_itg_fields", "linear_itg_rhs", "linear_itg_initial",
            "linear_itg_diagnostics", "integrate_linear_itg"]
from .nonlinear_itg import (
    NonlinearITG, plan_nonlinear_itg, nonlinear_itg_project,
    nonlinear_itg_fields, nonlinear_itg_bracket, nonlinear_itg_rhs,
    nonlinear_itg_initial, nonlinear_itg_diagnostics, integrate_nonlinear_itg,
    nonlinear_itg_drive_power, integrate_driven_itg,
)
__all__ += ["NonlinearITG", "plan_nonlinear_itg", "nonlinear_itg_project",
            "nonlinear_itg_fields", "nonlinear_itg_bracket", "nonlinear_itg_rhs",
            "nonlinear_itg_initial", "nonlinear_itg_diagnostics", "integrate_nonlinear_itg"]
__all__ += ["nonlinear_itg_drive_power", "integrate_driven_itg"]
from .collisional_itg import (
    CollisionalITG, plan_collisional_itg, collisional_itg_collision,
    collisional_itg_rhs, collisional_itg_rates, collisional_itg_transport,
    integrate_collisional_itg,
)
__all__ += ["CollisionalITG", "plan_collisional_itg", "collisional_itg_collision",
            "collisional_itg_rhs", "collisional_itg_rates", "collisional_itg_transport",
            "integrate_collisional_itg"]
