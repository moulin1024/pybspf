"""Unsteady nonlinear manufactured forcing exercises the rational mass terms."""

import sys
from pathlib import Path
import jax
import numpy as np
import pytest
import scipy.linalg as la
from bspf_jax.immersed_flow import ImmersedFlowPlan

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "pde"))
from hybrid_flow_mms import ContinuousHybridMMS  # noqa: E402

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def hybrid():
    return ImmersedFlowPlan(
        nx=33,
        ny=25,
        wall_method="rational",
        quadrature_factor=4,
        buffer_strength=3,
        rational_options=dict(degree=72, corner_poles=24, laurent=48, samples=600),
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
