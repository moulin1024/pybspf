# Migration to pybspf 0.2

Source snapshot: `361a39593de3ee6656b00f52cdd0b778a58280c5`.
The only maintained core is JAX, now named `pybspf`. Physical models and runners
are separately installable packages in this repository. No old import aliases
are provided. NumPy/SciPy host setup remains supported; the NumPy/CuPy backend
implementation is archived.

## Installation

Install `.[host,test]`, `packages/models[precision,test]`, then
`packages/sim[air-sea,test]` from the same checkout. Root package version is
0.2.0; models and sim start at 0.1.0. The models require core >=0.2,<0.3;
sim requires models >=0.1,<0.2. Models require SciPy >=1.15, preserving the
previous rational-flow dependency minimum. Installation does not publish a package.

## Module map

| Previous module | Maintained module |
| --- | --- |
| `bspf_jax._compressed_transform` | `bspf_models._numerics._compressed_transform` |
| `bspf_jax._energy_stable` | `bspf_models._numerics._energy_stable` |
| `bspf_jax._gpu_basis` | `bspf_models._numerics._gpu_basis` |
| `bspf_jax._gpu_linalg` | `bspf_models._numerics._gpu_linalg` |
| `bspf_jax._gpu_rational` | `bspf_models.fluids._gpu_rational` |
| `bspf_jax._immersed_assembly` | `bspf_models.fluids._immersed_assembly` |
| `bspf_jax._rational_preprocess` | `bspf_models.fluids._rational_preprocess` |
| `bspf_jax._tensor_pcg` | `bspf_models._numerics._tensor_pcg` |
| `bspf_jax._weak_basis` | `bspf_models._numerics._weak_basis` |
| `bspf_jax.air_sea` | `bspf_models.air_sea.air_sea` |
| `bspf_jax.air_sea_audit` | `bspf_models.air_sea.air_sea_audit` |
| `bspf_jax.air_sea_platform` | `bspf_sim.air_sea.platform` |
| `bspf_jax.air_sea_validation` | `bspf_sim.air_sea.validation` |
| `bspf_jax.alfven` | `bspf_models.plasma.alfven` |
| `bspf_jax.basis` | `pybspf.basis` |
| `bspf_jax.calculus` | `pybspf.calculus` |
| `bspf_jax.cavity` | `bspf_models.fluids.cavity` |
| `bspf_jax.collisional_itg` | `bspf_models.kinetic.collisional_itg` |
| `bspf_jax.convex_poisson` | `bspf_models.elliptic.convex_poisson` |
| `bspf_jax.convex_poisson_grid` | `bspf_models.elliptic.convex_poisson_grid` |
| `bspf_jax.convex_poisson_iterative` | `bspf_models.elliptic.convex_poisson_iterative` |
| `bspf_jax.convex_poisson_tensor` | `bspf_models.elliptic.convex_poisson_tensor` |
| `bspf_jax.drift_kinetic` | `bspf_models.kinetic.drift_kinetic` |
| `bspf_jax.elasticity` | `bspf_models.waves.elasticity` |
| `bspf_jax.embedded_poisson` | `bspf_models.elliptic.embedded_poisson` |
| `bspf_jax.endpoints` | `pybspf.endpoints` |
| `bspf_jax.fast_axis` | `pybspf.fast_axis` |
| `bspf_jax.fast_drift_kinetic` | `bspf_models.kinetic.fast_drift_kinetic` |
| `bspf_jax.fourier_extension` | `pybspf.fourier_extension` |
| `bspf_jax.fourier_poisson` | `bspf_models.elliptic.fourier_poisson` |
| `bspf_jax.galerkin` | `pybspf.galerkin` |
| `bspf_jax.grad_shafranov` | `bspf_models.plasma.grad_shafranov` |
| `bspf_jax.gs_response` | `bspf_models.plasma.gs_response` |
| `bspf_jax.gyrokinetic_mms` | `bspf_models.kinetic.gyrokinetic_mms` |
| `bspf_jax.gyrokinetic_slab` | `bspf_models.kinetic.gyrokinetic_slab` |
| `bspf_jax.immersed_flow` | `bspf_models.fluids.immersed_flow` |
| `bspf_jax.immersed_flow_gpu` | `bspf_models.fluids.immersed_flow_gpu` |
| `bspf_jax.immersed_poisson` | `bspf_models.elliptic.immersed_poisson` |
| `bspf_jax.isw_slope` | `bspf_models.fluids.isw_slope` |
| `bspf_jax.itg_bracket_tensor` | `bspf_models.kinetic.itg_bracket_tensor` |
| `bspf_jax.itg_statistics` | `bspf_models.kinetic.itg_statistics` |
| `bspf_jax.kdv` | `bspf_models.waves.kdv` |
| `bspf_jax.linear_itg` | `bspf_models.kinetic.linear_itg` |
| `bspf_jax.linear_itg_reference` | `bspf_models.kinetic.linear_itg_reference` |
| `bspf_jax.mapped_boussinesq` | `bspf_models.fluids.mapped_boussinesq` |
| `bspf_jax.mhd_cavity` | `bspf_models.plasma.mhd_cavity` |
| `bspf_jax.multirate` | `pybspf.multirate` |
| `bspf_jax.navier_stokes` | `bspf_models.fluids.navier_stokes` |
| `bspf_jax.noise` | `pybspf.noise` |
| `bspf_jax.nonlinear_itg` | `bspf_models.kinetic.nonlinear_itg` |
| `bspf_jax.normal_continuation` | `pybspf.normal_continuation` |
| `bspf_jax.open_slab_packet` | `bspf_models.kinetic.open_slab_packet` |
| `bspf_jax.operators` | `pybspf.operators` |
| `bspf_jax.panel_poisson` | `bspf_models.elliptic.panel_poisson` |
| `bspf_jax.parallel_kinetic` | `bspf_models.kinetic.parallel_kinetic` |
| `bspf_jax.plans` | `pybspf.plans` |
| `bspf_jax.pressure` | `bspf_models.elliptic.pressure` |
| `bspf_jax.pressure3d` | `bspf_models.elliptic.pressure3d` |
| `bspf_jax.random_wave_mms` | `bspf_models.elliptic.random_wave_mms` |
| `bspf_jax.rational_stokes` | `bspf_models.fluids.rational_stokes` |
| `bspf_jax.references` | `bspf_models.waves.references` |
| `bspf_jax.regularized_normal` | `bspf_models.elliptic.regularized_normal` |
| `bspf_jax.sine_gordon` | `bspf_models.waves.sine_gordon` |
| `bspf_jax.smooth_extension` | `bspf_models.elliptic.smooth_extension` |
| `bspf_jax.solovev` | `bspf_models.plasma.solovev` |
| `bspf_jax.stream_navier_stokes` | `bspf_models.fluids.stream_navier_stokes` |
| `bspf_jax.surface_exchange` | `bspf_models.air_sea.surface_exchange` |
| `bspf_jax.time_integration` | `pybspf.time_integration` |
| `bspf_jax.tokamak_equilibrium` | `bspf_models.plasma.tokamak_equilibrium` |
| `bspf_jax.tokamak_linear` | `bspf_models.plasma.tokamak_linear` |
| `bspf_jax.tokamak_vacuum` | `bspf_models.plasma.tokamak_vacuum` |
| `bspf_jax.tokamak_vacuum_bspf` | `bspf_models.plasma.tokamak_vacuum_bspf` |
| `bspf_jax.tokamak_velocity` | `bspf_models.plasma.tokamak_velocity` |
| `bspf_jax.vlasov_poisson` | `bspf_models.kinetic.vlasov_poisson` |
| `bspf_jax.weak_navier_stokes` | `bspf_models.fluids.weak_navier_stokes` |

`_flow_kernels` tensor operations are now `pybspf.tensor`; generic RK4 stages
and IMEX midpoint are in `pybspf.time_integration`. Physical curl assembly
remains internal to the models. `plans._integer` becomes
`pybspf.validation.integer`.

Shared `PressureLine`, `_fourier`, `_line_projector`, `_make_line`, `StreamLine`,
`_stream_line`, and `stream_evaluate_line` implementations belong to model-internal
`bspf_models._numerics.trial_spaces`. These are internal contracts, not newly
promoted public solver APIs.

`integrate_schrodinger` and `integrate_nlse` move from core time integration to
`bspf_models.waves.schrodinger`. Remaining time integrators keep their signatures.

## Previous top-level JAX exports

Use the following explicit imports instead of physical symbols from the core.
Core-owned symbols remain available directly from `pybspf`.

| Previous symbol | Implementation |
| --- | --- |
| `AlfvenPlan` | `bspf_models.plasma.alfven.AlfvenPlan` |
| `CollisionalITG` | `bspf_models.kinetic.collisional_itg.CollisionalITG` |
| `ConvexPoissonGridPlan` | `bspf_models.elliptic.convex_poisson_grid.ConvexPoissonGridPlan` |
| `ConvexPoissonGridResult` | `bspf_models.elliptic.convex_poisson_grid.ConvexPoissonGridResult` |
| `ConvexPoissonPlan` | `bspf_models.elliptic.convex_poisson.ConvexPoissonPlan` |
| `ConvexPoissonSolution` | `bspf_models.elliptic.convex_poisson.ConvexPoissonSolution` |
| `DriftKineticPlan` | `bspf_models.kinetic.drift_kinetic.DriftKineticPlan` |
| `EllipticHole` | `bspf_models.elliptic.immersed_poisson.EllipticHole` |
| `FastAxis` | `pybspf.fast_axis.FastAxis` |
| `FastDriftKineticPlan` | `bspf_models.kinetic.fast_drift_kinetic.FastDriftKineticPlan` |
| `FixedBoundaryGSPlan` | `bspf_models.plasma.grad_shafranov.FixedBoundaryGSPlan` |
| `FixedBoundaryGSSolution` | `bspf_models.plasma.grad_shafranov.FixedBoundaryGSSolution` |
| `FourierExtension` | `pybspf.fourier_extension.FourierExtension` |
| `FourierExtensionPlan` | `pybspf.fourier_extension.FourierExtensionPlan` |
| `FourierPoissonPlan` | `bspf_models.elliptic.fourier_poisson.FourierPoissonPlan` |
| `FourierPoissonSolution` | `bspf_models.elliptic.fourier_poisson.FourierPoissonSolution` |
| `GSFixedPointResponse` | `bspf_models.plasma.gs_response.GSFixedPointResponse` |
| `GSResponseResult` | `bspf_models.plasma.gs_response.GSResponseResult` |
| `Galerkin1D` | `pybspf.galerkin.Galerkin1D` |
| `ITGRadial` | `bspf_models.kinetic.linear_itg.ITGRadial` |
| `ImmersedFlowPlan` | `bspf_models.fluids.immersed_flow.ImmersedFlowPlan` |
| `ImmersedFlowStepper` | `bspf_models.fluids.immersed_flow.ImmersedFlowStepper` |
| `ImmersedPoissonPlan` | `bspf_models.elliptic.immersed_poisson.ImmersedPoissonPlan` |
| `ImmersedPoissonSolution` | `bspf_models.elliptic.immersed_poisson.ImmersedPoissonSolution` |
| `KdVPlan` | `bspf_models.waves.kdv.KdVPlan` |
| `LinearITG` | `bspf_models.kinetic.linear_itg.LinearITG` |
| `NSStageDiagnostics` | `bspf_models.fluids.navier_stokes.NSStageDiagnostics` |
| `NavierStokes2DPlan` | `bspf_models.fluids.navier_stokes.NavierStokes2DPlan` |
| `NonlinearITG` | `bspf_models.kinetic.nonlinear_itg.NonlinearITG` |
| `OpenSlabPacket` | `bspf_models.kinetic.open_slab_packet.OpenSlabPacket` |
| `ParallelKineticPlan` | `bspf_models.kinetic.parallel_kinetic.ParallelKineticPlan` |
| `Plan1D` | `pybspf.plans.Plan1D` |
| `PoissonIterationError` | `bspf_models.elliptic.convex_poisson_tensor.PoissonIterationError` |
| `PressurePoisson2DPlan` | `bspf_models.elliptic.pressure.PressurePoisson2DPlan` |
| `PressurePoisson2DResult` | `bspf_models.elliptic.pressure.PressurePoisson2DResult` |
| `PressurePoisson3DPlan` | `bspf_models.elliptic.pressure3d.PressurePoisson3DPlan` |
| `PressurePoisson3DResult` | `bspf_models.elliptic.pressure3d.PressurePoisson3DResult` |
| `SlabGKPlan` | `bspf_models.kinetic.gyrokinetic_slab.SlabGKPlan` |
| `SlabMMS` | `bspf_models.kinetic.gyrokinetic_mms.SlabMMS` |
| `SolovevEquilibrium` | `bspf_models.plasma.solovev.SolovevEquilibrium` |
| `SolovevFluxDomain` | `bspf_models.plasma.solovev.SolovevFluxDomain` |
| `Split` | `pybspf.operators.Split` |
| `StreamNavierStokes2DPlan` | `bspf_models.fluids.stream_navier_stokes.StreamNavierStokes2DPlan` |
| `StreamSponge2D` | `bspf_models.fluids.stream_navier_stokes.StreamSponge2D` |
| `TensorConvexPoissonPlan` | `bspf_models.elliptic.convex_poisson_tensor.TensorConvexPoissonPlan` |
| `TensorPlan` | `pybspf.plans.TensorPlan` |
| `VlasovPoissonPlan` | `bspf_models.kinetic.vlasov_poisson.VlasovPoissonPlan` |
| `WeakNavierStokes2DPlan` | `bspf_models.fluids.weak_navier_stokes.WeakNavierStokes2DPlan` |
| `alfven_boundary_power` | `bspf_models.plasma.alfven.alfven_boundary_power` |
| `alfven_energy` | `bspf_models.plasma.alfven.alfven_energy` |
| `antiderivative` | `pybspf.calculus.antiderivative` |
| `basis_matrix` | `pybspf.basis.basis_matrix` |
| `boundary_clustered_knots` | `pybspf.fast_axis.boundary_clustered_knots` |
| `collisional_itg_collision` | `bspf_models.kinetic.collisional_itg.collisional_itg_collision` |
| `collisional_itg_rates` | `bspf_models.kinetic.collisional_itg.collisional_itg_rates` |
| `collisional_itg_rhs` | `bspf_models.kinetic.collisional_itg.collisional_itg_rhs` |
| `collisional_itg_transport` | `bspf_models.kinetic.collisional_itg.collisional_itg_transport` |
| `compress_pressure_plan` | `bspf_models.elliptic.pressure.compress_pressure_plan` |
| `compress_pressure_plan3d` | `bspf_models.elliptic.pressure3d.compress_pressure_plan3d` |
| `curl` | `pybspf.operators.curl` |
| `decompose` | `pybspf.operators.decompose` |
| `derivatives` | `pybspf.operators.derivatives` |
| `differentiate` | `pybspf.operators.differentiate` |
| `divergence` | `pybspf.operators.divergence` |
| `drift_kinetic_moments` | `bspf_models.kinetic.drift_kinetic.drift_kinetic_moments` |
| `drift_kinetic_rhs` | `bspf_models.kinetic.drift_kinetic.drift_kinetic_rhs` |
| `elastic_modes` | `bspf_models.waves.elasticity.elastic_modes` |
| `endpoint_jets` | `pybspf.operators.endpoint_jets` |
| `galerkin_1d` | `pybspf.galerkin.galerkin_1d` |
| `gradient` | `pybspf.operators.gradient` |
| `hessian` | `pybspf.operators.hessian` |
| `integrate` | `pybspf.calculus.integrate` |
| `integrate_alfven` | `bspf_models.plasma.alfven.integrate_alfven` |
| `integrate_box` | `pybspf.calculus.integrate_box` |
| `integrate_collisional_itg` | `bspf_models.kinetic.collisional_itg.integrate_collisional_itg` |
| `integrate_drift_kinetic` | `bspf_models.kinetic.drift_kinetic.integrate_drift_kinetic` |
| `integrate_driven_itg` | `bspf_models.kinetic.nonlinear_itg.integrate_driven_itg` |
| `integrate_elastic` | `bspf_models.waves.elasticity.integrate_elastic` |
| `integrate_kdv` | `bspf_models.waves.kdv.integrate_kdv` |
| `integrate_linear_itg` | `bspf_models.kinetic.linear_itg.integrate_linear_itg` |
| `integrate_linear_midpoint` | `pybspf.time_integration.integrate_linear_midpoint` |
| `integrate_log_drift_kinetic` | `bspf_models.kinetic.drift_kinetic.integrate_log_drift_kinetic` |
| `integrate_nlse` | `bspf_models.waves.schrodinger.integrate_nlse` |
| `integrate_nonlinear_itg` | `bspf_models.kinetic.nonlinear_itg.integrate_nonlinear_itg` |
| `integrate_open_packet` | `bspf_models.kinetic.open_slab_packet.integrate_open_packet` |
| `integrate_parallel_kinetic` | `bspf_models.kinetic.parallel_kinetic.integrate_parallel_kinetic` |
| `integrate_rk4` | `pybspf.time_integration.integrate_rk4` |
| `integrate_schrodinger` | `bspf_models.waves.schrodinger.integrate_schrodinger` |
| `integrate_sine_gordon` | `bspf_models.waves.sine_gordon.integrate_sine_gordon` |
| `integrate_slab_gk` | `bspf_models.kinetic.gyrokinetic_slab.integrate_slab_gk` |
| `integrate_vlasov_poisson` | `bspf_models.kinetic.vlasov_poisson.integrate_vlasov_poisson` |
| `interpolate` | `pybspf.calculus.interpolate` |
| `interpolate_grid` | `pybspf.calculus.interpolate_grid` |
| `kh_initial_velocity` | `bspf_models.fluids.navier_stokes.kh_initial_velocity` |
| `laplacian` | `pybspf.operators.laplacian` |
| `linear_itg_diagnostics` | `bspf_models.kinetic.linear_itg.linear_itg_diagnostics` |
| `linear_itg_fields` | `bspf_models.kinetic.linear_itg.linear_itg_fields` |
| `linear_itg_initial` | `bspf_models.kinetic.linear_itg.linear_itg_initial` |
| `linear_itg_rhs` | `bspf_models.kinetic.linear_itg.linear_itg_rhs` |
| `log_drift_kinetic_diagnostics` | `bspf_models.kinetic.drift_kinetic.log_drift_kinetic_diagnostics` |
| `mixed_partial` | `pybspf.operators.mixed_partial` |
| `noise_diagnostics` | `pybspf.operators.noise_diagnostics` |
| `nonlinear_itg_bracket` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_bracket` |
| `nonlinear_itg_diagnostics` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_diagnostics` |
| `nonlinear_itg_drive_power` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_drive_power` |
| `nonlinear_itg_fields` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_fields` |
| `nonlinear_itg_initial` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_initial` |
| `nonlinear_itg_project` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_project` |
| `nonlinear_itg_rhs` | `bspf_models.kinetic.nonlinear_itg.nonlinear_itg_rhs` |
| `ns_divergence` | `bspf_models.fluids.navier_stokes.ns_divergence` |
| `ns_project_velocity` | `bspf_models.fluids.navier_stokes.ns_project_velocity` |
| `ns_raw_rhs` | `bspf_models.fluids.navier_stokes.ns_raw_rhs` |
| `ns_rhs` | `bspf_models.fluids.navier_stokes.ns_rhs` |
| `ns_rk4_step` | `bspf_models.fluids.navier_stokes.ns_rk4_step` |
| `ns_vorticity` | `bspf_models.fluids.navier_stokes.ns_vorticity` |
| `open_knots` | `pybspf.basis.open_knots` |
| `packet_diagnostics` | `bspf_models.kinetic.open_slab_packet.packet_diagnostics` |
| `packet_fields` | `bspf_models.kinetic.open_slab_packet.packet_fields` |
| `packet_initial` | `bspf_models.kinetic.open_slab_packet.packet_initial` |
| `packet_reference` | `bspf_models.kinetic.open_slab_packet.packet_reference` |
| `packet_reference_moments` | `bspf_models.kinetic.open_slab_packet.packet_reference_moments` |
| `packet_rhs` | `bspf_models.kinetic.open_slab_packet.packet_rhs` |
| `plan_1d` | `pybspf.plans.plan_1d` |
| `plan_2d` | `pybspf.plans.plan_2d` |
| `plan_3d` | `pybspf.plans.plan_3d` |
| `plan_alfven` | `bspf_models.plasma.alfven.plan_alfven` |
| `plan_collisional_itg` | `bspf_models.kinetic.collisional_itg.plan_collisional_itg` |
| `plan_drift_kinetic` | `bspf_models.kinetic.drift_kinetic.plan_drift_kinetic` |
| `plan_fast_axis` | `pybspf.fast_axis.plan_fast_axis` |
| `plan_itg_radial` | `bspf_models.kinetic.linear_itg.plan_itg_radial` |
| `plan_kdv` | `bspf_models.waves.kdv.plan_kdv` |
| `plan_linear_itg` | `bspf_models.kinetic.linear_itg.plan_linear_itg` |
| `plan_navier_stokes2d` | `bspf_models.fluids.navier_stokes.plan_navier_stokes2d` |
| `plan_nonlinear_itg` | `bspf_models.kinetic.nonlinear_itg.plan_nonlinear_itg` |
| `plan_open_slab_packet` | `bspf_models.kinetic.open_slab_packet.plan_open_slab_packet` |
| `plan_parallel_kinetic` | `bspf_models.kinetic.parallel_kinetic.plan_parallel_kinetic` |
| `plan_pressure_poisson2d` | `bspf_models.elliptic.pressure.plan_pressure_poisson2d` |
| `plan_pressure_poisson3d` | `bspf_models.elliptic.pressure3d.plan_pressure_poisson3d` |
| `plan_slab_gk` | `bspf_models.kinetic.gyrokinetic_slab.plan_slab_gk` |
| `plan_slab_mms` | `bspf_models.kinetic.gyrokinetic_mms.plan_slab_mms` |
| `plan_stream_navier_stokes2d` | `bspf_models.fluids.stream_navier_stokes.plan_stream_navier_stokes2d` |
| `plan_stream_sponge` | `bspf_models.fluids.stream_navier_stokes.plan_stream_sponge` |
| `plan_vlasov_poisson` | `bspf_models.kinetic.vlasov_poisson.plan_vlasov_poisson` |
| `plan_weak_navier_stokes2d` | `bspf_models.fluids.weak_navier_stokes.plan_weak_navier_stokes2d` |
| `poisson_dirichlet` | `bspf_models.kinetic.vlasov_poisson.poisson_dirichlet` |
| `pressure_action3d` | `bspf_models.elliptic.pressure3d.pressure_action3d` |
| `pressure_divergence` | `bspf_models.elliptic.pressure.pressure_divergence` |
| `pressure_gradient` | `bspf_models.elliptic.pressure.pressure_gradient` |
| `pressure_lift3d` | `bspf_models.elliptic.pressure3d.pressure_lift3d` |
| `pressure_remove_mean` | `bspf_models.elliptic.pressure.pressure_remove_mean` |
| `pressure_schur` | `bspf_models.elliptic.pressure.pressure_schur` |
| `pressure_schur3d` | `bspf_models.elliptic.pressure3d.pressure_schur3d` |
| `project_pressure2d` | `bspf_models.elliptic.pressure.project_pressure2d` |
| `rk4_step` | `pybspf.time_integration.rk4_step` |
| `sample_aligned_knots` | `pybspf.fast_axis.sample_aligned_knots` |
| `slab_gk_diagnostics` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_diagnostics` |
| `slab_gk_fields` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_fields` |
| `slab_gk_project` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_project` |
| `slab_gk_rhs` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_rhs` |
| `slab_gk_rhs_with_field_hat` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_rhs_with_field_hat` |
| `slab_gk_solve_charge` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_solve_charge` |
| `slab_gk_source_rates` | `bspf_models.kinetic.gyrokinetic_slab.slab_gk_source_rates` |
| `solve_pressure_poisson2d` | `bspf_models.elliptic.pressure.solve_pressure_poisson2d` |
| `solve_pressure_poisson3d` | `bspf_models.elliptic.pressure3d.solve_pressure_poisson3d` |
| `spline_primitive` | `pybspf.basis.spline_primitive` |
| `stream_evaluate_line` | `bspf_models.fluids.stream_navier_stokes.stream_evaluate_line` |
| `stream_kh_initial` | `bspf_models.fluids.stream_navier_stokes.stream_kh_initial` |
| `stream_ns_boundary_load` | `bspf_models.fluids.stream_navier_stokes.stream_ns_boundary_load` |
| `stream_ns_divergence` | `bspf_models.fluids.stream_navier_stokes.stream_ns_divergence` |
| `stream_ns_energy` | `bspf_models.fluids.stream_navier_stokes.stream_ns_energy` |
| `stream_ns_inertia_apply` | `bspf_models.fluids.stream_navier_stokes.stream_ns_inertia_apply` |
| `stream_ns_inertia_solve` | `bspf_models.fluids.stream_navier_stokes.stream_ns_inertia_solve` |
| `stream_ns_load` | `bspf_models.fluids.stream_navier_stokes.stream_ns_load` |
| `stream_ns_open_velocity` | `bspf_models.fluids.stream_navier_stokes.stream_ns_open_velocity` |
| `stream_ns_rhs` | `bspf_models.fluids.stream_navier_stokes.stream_ns_rhs` |
| `stream_ns_rk4_step` | `bspf_models.fluids.stream_navier_stokes.stream_ns_rk4_step` |
| `stream_ns_sponge_load` | `bspf_models.fluids.stream_navier_stokes.stream_ns_sponge_load` |
| `stream_ns_velocity` | `bspf_models.fluids.stream_navier_stokes.stream_ns_velocity` |
| `stream_ns_vorticity` | `bspf_models.fluids.stream_navier_stokes.stream_ns_vorticity` |
| `tensor_decompose` | `pybspf.operators.tensor_decompose` |
| `tensor_plan` | `pybspf.plans.tensor_plan` |
| `vlasov_poisson_fields` | `bspf_models.kinetic.vlasov_poisson.vlasov_poisson_fields` |
| `weak_kh_initial_velocity` | `bspf_models.fluids.weak_navier_stokes.weak_kh_initial_velocity` |
| `weak_ns_divergence` | `bspf_models.fluids.weak_navier_stokes.weak_ns_divergence` |
| `weak_ns_energy` | `bspf_models.fluids.weak_navier_stokes.weak_ns_energy` |
| `weak_ns_helmholtz` | `bspf_models.fluids.weak_navier_stokes.weak_ns_helmholtz` |
| `weak_ns_load` | `bspf_models.fluids.weak_navier_stokes.weak_ns_load` |
| `weak_ns_momentum_load` | `bspf_models.fluids.weak_navier_stokes.weak_ns_momentum_load` |
| `weak_ns_pointwise_divergence` | `bspf_models.fluids.weak_navier_stokes.weak_ns_pointwise_divergence` |
| `weak_ns_project` | `bspf_models.fluids.weak_navier_stokes.weak_ns_project` |
| `weak_ns_rhs` | `bspf_models.fluids.weak_navier_stokes.weak_ns_rhs` |
| `weak_ns_rk4_step` | `bspf_models.fluids.weak_navier_stokes.weak_ns_rk4_step` |
| `weak_ns_vorticity` | `bspf_models.fluids.weak_navier_stokes.weak_ns_vorticity` |
| `with_regularization` | `pybspf.plans.with_regularization` |
| `with_stream_dynamic_boundary` | `bspf_models.fluids.stream_navier_stokes.with_stream_dynamic_boundary` |

## Previous NumPy/CuPy API

| Previous API | Status |
| --- | --- |
| `BSPF1D`, `BSPF2D`, `PiecewiseBSPF1D`, `Grid1D` | Archived; use the JAX plan/operator API for new work; no drop-in object adapter |
| `Poisson*Solver`, `PressurePoisson2D`, Neumann precomputations | Archived; new model solvers have their own existing JAX contracts |
| `ClosedBSPFLine` | `pybspf.trial_spaces.ClosedBSPFLine`; host construction preserved |
| `use_gpu` / CuPy backend selection | Archived; select JAX devices instead |
| Old RK4 wrapper | Archived; use JAX `pybspf.time_integration` signatures |
| Other 0.1 modules | Preserved in `legacy/numpy_cupy/src/pybspf`, not shipped in new packages |

## Checkpoints and provenance

Configuration and checkpoint schema v1 are unchanged. Exact resume still checks
Python, numerical dependency versions, device and precision. A source-identity
migration accepts only the recorded pre-migration source fingerprint; unknown
sources remain rejected. Migrated runs record current package-qualified hashes
and versions, with original provenance retained under `migrated_from`.
This preserves state/ledger data while making the code transition explicit.

## References and archive

Pressure regression arrays and closed-space values were generated with the
source snapshot before migration. The archived `generate_reference.py --out DIR`
recreates them in a separate old-package environment. Their formulas, seed (5),
grids and parameters are in that script; existing pressure tolerances are retained.
Existing ISW 33x33 and 65x33 references are unchanged. The v1 checkpoint fixture
was generated using a 33-node, quadrature-20 air-sea run with a 30-second macro
step, 6-second inner step and checkpoint at 60 seconds.

Default tests never execute archived code. Frozen experiment sources and prior
results retain their original names and hashes. Historical NumPy/CuPy docs are
under the archive. See `migration-baseline.json` for verification evidence.
