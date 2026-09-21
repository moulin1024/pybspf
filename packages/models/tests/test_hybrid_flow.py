"""Unsteady nonlinear manufactured forcing exercises the rational mass terms."""

import sys
from pathlib import Path
import jax
import numpy as np
import pytest
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "pde"))
from hybrid_flow_mms import ContinuousHybridMMS  # noqa: E402

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", params=["host", "gpu", "gpu-float64", "gpu-float64-arnoldi"])
def hybrid(request):
    device = None
    if request.param != "host":
        try:
            device = jax.devices("gpu")[0]
        except RuntimeError:
            pytest.skip("GPU device unavailable")
    return ImmersedFlowPlan(
        assembly_device=device,
        basis_precision="float64" if request.param.startswith("gpu-float64") else "mpfr",
        basis_workers=2 if device is not None else 1,
        nx=33,
        ny=25,
        wall_method="rational",
        quadrature_factor=4,
        buffer_strength=3,
        rational_options=dict(degree=72, corner_poles=24, laurent=48, samples=600,
                              basis_construction="gpu" if request.param.endswith("arnoldi") else "cpu"),
    )


def test_unsteady_nonlinear_mms_and_rational_time_term(hybrid):
    p = hybrid
    exact = ContinuousHybridMMS(p.bounds, p.hole, waves=False, reference_degree=96)
    base, delta, _, load, missing = exact.prepare(p)
    c = la.cho_solve(p.mass_factor, p.force_load(np.column_stack(delta[1:3])))
    t = 0.31
    s, rate = exact.amplitude(t)
    residual = p.explicit(s * c) + load(t) - p.linear @ (s * c) - rate * (p.mass @ c)

    def dual(r):
        return np.sqrt(max(r @ la.cho_solve(p.mass_factor, r), 0))

    print("RESIDUAL", dual(residual), "OMITTED", dual(residual - rate * missing))
    assert dual(residual) < 1e-4
    assert dual(residual - rate * missing) > 100 * dual(residual)
    errors = []
    for dt in (0.08, 0.04, 0.02):
        a = np.zeros(p.dofs)
        step = p.stepper(dt)
        for k in range(round(0.4 / dt)):
            a = step.step(a, k * dt, load)
        target = [b + exact.amplitude(0.4)[0] * d for b, d in zip(base, delta)]
        du, dv = [
            o @ a + lift - f
            for o, lift, f in zip(
                p.operators_fluid[:2], p.lift_fields[1:3], target[1:3]
            )
        ]
        errors.append(np.sqrt(p.weights @ (du * du + dv * dv)))
    print("TIME ERRORS", errors)
    assert errors[0] / errors[1] > 3.2
    assert errors[1] / errors[2] > 3.2
    assert errors[-1] < 1e-5
    b, _ = p.arc.sample(190, offset=0.381)
    fields = p.evaluate(a, b)
    assert np.max(np.hypot(fields[1], fields[2])) < 1e-9


def test_plain_ns_boundary_flux_and_streamfunction(hybrid):
    p = hybrid
    a = p.stokes_state.copy()
    step = p.stepper(0.02)
    for k in range(10):
        a = step.step(a, k * 0.02)
    b, _ = p.arc.sample(222, offset=0.37)
    fields = p.evaluate(a, b)
    assert np.max(np.hypot(fields[1], fields[2])) < 1e-9
    np.testing.assert_allclose(
        p.out_weights @ (p.out_ops[0] @ a + p.out_lift[0]), 4 / 3, atol=1e-9
    )
    assert np.isfinite(p.diagnostics(a)["kinetic_energy"])
    # Single-valued streamfunction across the logarithm's branch cut.
    x = np.linspace(-0.8, -0.2, 11)
    eps = 1e-9
    cy = p.hole.center[1]
    plus = np.column_stack((x, np.full_like(x, cy + eps)))
    minus = np.column_stack((x, np.full_like(x, cy - eps)))
    fp, fm = p.evaluate(a, plus), p.evaluate(a, minus)
    np.testing.assert_allclose(fp[0] - fm[0], eps * (fp[1] + fm[1]), atol=1e-11)
    with pytest.raises(ValueError, match="only in the fluid"):
        p.evaluate(a, np.array([p.hole.center]))


@pytest.mark.parametrize("batch_size", [1, 2, 64])
def test_batched_grid_matches_individual_snapshots(hybrid, batch_size):
    p = hybrid
    rng = np.random.default_rng(73)
    states = np.stack([p.stokes_state, p.stokes_state + rng.normal(size=p.dofs)*1e-5,
                       p.stokes_state - rng.normal(size=p.dofs)*1e-5])
    x, y = np.linspace(-1,5,19), np.linspace(-1,1,15)
    expected = [p.grid(state,x,y) for state in states]
    actual = list(p.grid_many(states,x,y,batch_size=batch_size))
    assert len(actual) == len(expected)
    for got, want in zip(actual,expected):
        for key in want:
            np.testing.assert_allclose(got[key],want[key],rtol=2e-10,atol=2e-10,equal_nan=True)


def test_gpu_volume_operators_match_host_reference(hybrid):
    p = hybrid
    if p.assembly_device is None:
        pytest.skip("CPU reference fixture")
    points = p.points[::max(1, len(p.points)//31)][:31]
    # Include a partial final chunk and both vector/matrix coefficient inputs.
    for coefficients in (p.rational_lift,
                         np.column_stack((p.rational_modes[:, :3], p.rational_lift))):
        expected = p.rational.evaluate(points, coefficients, batch_size=17)
        actual = p.rational.evaluate(points, coefficients, batch_size=17,
                                     device=p.assembly_device, return_device=True)
        assert all(a.devices() == {p.assembly_device} for a in actual)
        for got, want in zip(jax.device_get(actual), expected):
            np.testing.assert_allclose(got, want, atol=2e-10, rtol=2e-10)
    expected_ops, expected_base = p.operators(points, with_base=True)
    ops, base = p.operators(points, with_base=True, device_output=True)
    assert all(a.devices() == {p.assembly_device} for a in (*ops, *base))
    for got, want in zip(jax.tree_util.tree_leaves(jax.device_get((ops, base))),
                         jax.tree_util.tree_leaves((expected_ops, expected_base))):
        np.testing.assert_allclose(got, want, atol=2e-9, rtol=2e-10)
