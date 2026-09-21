"""The FP64 basis option is explicit, runs on GPU, and never falls back to MPFR."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from scipy.interpolate import BSpline

from bspf_models._numerics._gpu_basis import _trial_values
from bspf_models._numerics._gpu_basis import gpu_trial_values
from bspf_models._numerics._weak_basis import mp_trial_values
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan


@pytest.fixture(scope="module")
def device():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("GPU device unavailable")


@pytest.mark.parametrize("n", [10, 11])
def test_fp64_basis_matches_mpfr_for_well_conditioned_projector(device, n):
    pytest.importorskip("gmpy2")
    rng = np.random.default_rng(24)
    x = np.linspace(-1, 1, n)
    knots = np.r_[np.repeat(-1., 4), np.linspace(-1, 1, 6)[1:-1], np.repeat(1., 4)]
    spline = BSpline(knots, np.eye(8), 3)
    line = SimpleNamespace(x=x, P=rng.normal(size=(8, n)))
    points = np.r_[x, np.linspace(-.97, .97, 23), x[:-1]+1e-14]
    transform = rng.normal(size=(n+2, 7))
    options = dict(second=True, transform=transform, layers=(.03,))
    expected = mp_trial_values(line, spline, points, **options)
    actual = gpu_trial_values(line, spline, points, device=device, **options)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=2e-10, atol=2e-9)
    arrays = jax.device_put((x, knots, line.P, points, transform, np.array([.03])), device)
    with jax.transfer_guard("disallow"):
        result = _trial_values(*arrays, degree=3, second=True, values_only=False)
        jax.block_until_ready(result)
    assert all(a.dtype == np.float64 and a.devices() == {device} for a in result)
    (values,) = gpu_trial_values(line, spline, points, device=device, values_only=True)
    reference = mp_trial_values(line, spline, points, values_only=True)[0]
    np.testing.assert_allclose(values, reference, atol=2e-12, rtol=2e-12)


def test_fp64_plan_and_postprocessing_never_use_mpfr(device, monkeypatch):
    import bspf_models._numerics._weak_basis as weak
    def forbidden(*args, **kwargs):
        raise AssertionError("FP64 option called MPFR")
    monkeypatch.setattr(weak, "evaluate_mp_chunks", forbidden)
    p = ImmersedFlowPlan(nx=17, ny=17, hole=None, assembly_device=device,
                         basis_precision="float64", basis_workers=4)
    assert p.basis_precision == "float64"
    assert p._basis_executor is None
    step = p.stepper(.002, device=device)
    state = step.step(step.initial_state)
    fields = p.grid(jax.device_get(state), np.linspace(-1, 5, 9), np.linspace(-1, 1, 7))
    assert all(np.all(np.isfinite(a)) for a in fields.values())


def test_basis_precision_validation():
    with pytest.raises(ValueError, match="basis_precision"):
        ImmersedFlowPlan(basis_precision="double")
    with pytest.raises(ValueError, match="requires assembly_device"):
        ImmersedFlowPlan(basis_precision="float64")
