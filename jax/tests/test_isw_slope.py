"""Numerical equivalence, invariants, failure handling and checkpoint regression."""

import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("pybspf", reason="Install the slope extra and root pybspf package")

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "examples/pde/isw_slope/source")
)
from slope_solver import BSPF, backend
import jax
import jax.numpy as jnp
from execution import parser, run


@pytest.fixture(scope="module", params=[(33, 33), (65, 33)])
def models(request):
    nx, nz = request.param
    with np.load(Path(__file__).parent / "reference/isw_slope" / f"{nx}x{nz}.npz") as f:
        reference = {key: f[key].copy() for key in f.files}
    return reference, BSPF(nx, nz)


def test_reference_rhs_rk4_and_invariants(models):
    ref, model = models
    rng = np.random.default_rng(214)
    a = rng.normal(size=model.shape) * 1e-7
    b = rng.normal(size=model.bshape) * 1e-6
    assert isinstance(model.mass(a), jax.Array)
    assert model.mass(a).dtype == jnp.float64
    np.testing.assert_allclose(model.mass(a), ref["mass"], rtol=2e-11, atol=1e-12)
    np.testing.assert_allclose(model.fields(a), ref["fields"], rtol=2e-10, atol=1e-12)
    u, w, ux, _, _, wz = model.fields(a)
    np.testing.assert_allclose(ux + wz, 0.0, atol=1e-13)
    np.testing.assert_allclose(
        jnp.sum(a * model.mass(a)), jnp.sum(model.W * (u * u + w * w)), rtol=1e-11
    )
    for actual, expected in zip(model.rhs(a, b, True), (ref["rhs_a"], ref["rhs_b"])):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-12)
    assert model.last_budget["mass_residual"] < 1.05e-11
    assert model.last_budget["kinetic_budget_rel"] < 1e-9
    before = len(model.iterations)
    for actual, expected in zip(model.rk4(a, b, 0.0001), (ref["rk4_a"], ref["rk4_b"])):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-12)
    assert len(model.iterations) == before + 4


def test_zero_and_failed_cg(models, monkeypatch):
    _, model = models
    np.testing.assert_array_equal(model.solve(np.zeros(model.shape)), 0.0)
    assert model.iterations[-1] == 0
    with pytest.raises(RuntimeError, match="CG failed"):
        model.solve(np.full(model.shape, np.nan))
    plan = dict(
        model.plan, mass_terms=tuple((-x, z) for x, z in model.plan["mass_terms"])
    )
    _, info = backend.rk4(
        plan, jnp.ones(model.shape) * 1e-7, jnp.zeros(model.bshape), 0.0001
    )
    with pytest.raises(RuntimeError, match="CG failed"):
        model._record(info)
    monkeypatch.setattr(model, "plan", plan)
    with pytest.raises(RuntimeError, match="CG failed"):
        model.rk4(jnp.ones(model.shape) * 1e-7, jnp.zeros(model.bshape), 0.0001)


def test_nowave_restart(tmp_path):
    def args(out, *extra):
        return parser().parse_args(
            [
                "--nx",
                "33",
                "--nz",
                "33",
                "--dt",
                ".01",
                "--save",
                ".01",
                "--out",
                str(out),
                *extra,
            ]
        )

    full, split, resumed = (tmp_path / k for k in ("full", "split", "resumed"))
    run(args(full, "--nowave", "--tfinal", ".02"))
    run(args(split, "--nowave", "--tfinal", ".01"))
    checkpoint = next(split.glob("state_*p01.npz"))
    result = run(args(resumed, "--resume", str(checkpoint), "--tfinal", ".02"))
    assert result["completed"] and result["backend"] == "bspf_jax.isw_slope"
    with (
        np.load(next(full.glob("state_*p02.npz"))) as f,
        np.load(next(resumed.glob("state_*p02.npz"))) as r,
    ):
        for key in ("a", "b"):
            np.testing.assert_allclose(r[key], f[key], rtol=1e-12, atol=1e-15)


def test_initial_projection(models, monkeypatch):
    # Analytic fixture exercises the initial projection without substituting
    # an invented file for the absent frozen physical wave.
    import slope_solver

    class Initial:
        def velocity(self, q, s):
            q, s = q[:, None], s[None, :]
            return 1e-4 * np.sin(np.pi * q) * np.sin(np.pi * s), np.zeros(
                (len(q), s.size)
            )

        def b(self, q, s):
            return 1e-5 * np.sin(np.pi * q[:, None]) * np.sin(np.pi * s[None, :])

    monkeypatch.setattr(slope_solver, "SharedInitial", Initial)
    ref, model = models
    for actual, expected in zip(model.initial(), (ref["initial_a"], ref["initial_b"])):
        assert isinstance(actual, jax.Array)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-12)


def test_rk4_only_downloads_small_report(models, monkeypatch):
    _, model = models
    a = jnp.zeros(model.shape)
    b = jnp.zeros(model.bshape)
    downloads = []
    original_get = jax.device_get

    def tracked_get(value):
        downloads.append(value)
        return original_get(value)

    monkeypatch.setattr(jax, "device_get", tracked_get)
    # Disallow implicit device-to-host conversions, while allowing the explicit
    # download of the validation report. Applicable on both CPU and GPU.
    with jax.transfer_guard_device_to_host("disallow"):
        state = model.rk4(a, b, 0.0001)
    assert all(isinstance(v, jax.Array) for v in state)
    assert len(downloads) == 1
    assert downloads[0].shape == (13,)
    assert downloads[0].dtype == jnp.float64


def test_checked_kernel_detects_nonfinite_final_state(models):
    _, model = models
    # Infinite step length makes the final update nonfinite even when the first
    # RHS has a converged mass solve. Check the flag directly, not on the host.
    _, report = backend.rk4_checked(
        model.plan, jnp.zeros(model.shape), jnp.zeros(model.bshape), float("inf")
    )
    assert not bool(report[-1])


def test_rejected_step_preserves_last_checkpoint(tmp_path, monkeypatch):
    import json

    original = backend.rk4_checked
    calls = 0

    def fail_second_step(*args):
        nonlocal calls
        calls += 1
        state, report = original(*args)
        if calls == 2:
            # Simulate a nonfinite final combination with all CG stages OK.
            state = (jnp.full_like(state[0], jnp.nan), state[1])
            report = report.at[8:12].set(1).at[12].set(0)
        return state, report

    monkeypatch.setattr(backend, "rk4_checked", fail_second_step)
    out = tmp_path / "rejected"
    args = parser().parse_args(
        [
            "--nx",
            "33",
            "--nz",
            "33",
            "--nowave",
            "--dt",
            ".01",
            "--save",
            ".01",
            "--tfinal",
            ".02",
            "--out",
            str(out),
        ]
    )
    with pytest.raises(FloatingPointError, match="Non-finite RK4"):
        run(args)
    status = json.loads((out / "status.json").read_text())
    assert not status["completed"]
    assert status["last_valid_s"] == 0.01
    assert status["steps_this_invocation"] == 1
    with np.load(out / "interrupted_checkpoint.npz") as checkpoint:
        assert float(checkpoint["time_s"]) == 0.01
        assert all(np.isfinite(checkpoint[k]).all() for k in ("a", "b"))
    assert not (out / "summary.json").exists()
