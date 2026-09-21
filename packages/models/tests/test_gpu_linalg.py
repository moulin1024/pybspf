"""GPU setup decompositions preserve rank, reconstruction, and eigenspaces."""
import jax
import numpy as np
import pytest

from bspf_models._numerics._gpu_linalg import _svd
from bspf_models._numerics._gpu_linalg import _eigh
from bspf_models._numerics._gpu_linalg import _checked_host
from bspf_models._numerics._gpu_linalg import gpu_svd
from bspf_models._numerics._gpu_linalg import gpu_eigh


@pytest.fixture(scope="module")
def device():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("GPU device unavailable")


@pytest.mark.parametrize("shape", [(23, 8), (8, 23)])
@pytest.mark.parametrize("full", [False, True])
def test_gpu_svd_rank_and_orthogonality(device, shape, full):
    rng = np.random.default_rng(38)
    m, n = shape
    k = min(m, n)
    left = np.linalg.qr(rng.normal(size=(m, k)))[0]
    right = np.linalg.qr(rng.normal(size=(n, k)))[0]
    spectrum = np.array([1., .2, .01, 1e-6, 1e-10, 2e-13, 2e-14, 0.])
    a = (left*spectrum) @ right.T
    uploaded = jax.device_put(a, device)
    with jax.transfer_guard("disallow"):
        factors = _svd(uploaded, full_matrices=full)
        jax.block_until_ready(factors)
    assert all(x.devices() == {device} for x in factors)
    u, s, vh = jax.device_get(factors)
    np.testing.assert_allclose((u[:, :k]*s) @ vh[:k], a, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=2e-13)
    np.testing.assert_allclose(vh @ vh.T, np.eye(vh.shape[0]), atol=2e-13)
    np.testing.assert_allclose(s, spectrum, atol=2e-15, rtol=2e-13)
    assert np.count_nonzero(s > 1e-13*s[0]) == 6
    if full and n > m:
        np.testing.assert_allclose(a @ vh[k:].T, 0., atol=2e-14)
    host_factors = gpu_svd(a, device=device, full_matrices=full)
    assert all(isinstance(x, np.ndarray) for x in host_factors)


def test_gpu_energy_eigh_residual_and_mode_cutoff(device):
    rng = np.random.default_rng(81)
    q = np.linalg.qr(rng.normal(size=(19, 19)))[0]
    eigenvalues = np.r_[1e-13, 2e-12, 2e-10, np.geomspace(1e-8, 1., 16)]
    a = (q*eigenvalues) @ q.T
    uploaded = jax.device_put(a, device)
    with jax.transfer_guard("disallow"):
        factors = _eigh(uploaded)
        jax.block_until_ready(factors)
    assert all(x.devices() == {device} for x in factors)
    w, v = jax.device_get(factors)
    np.testing.assert_allclose(a @ v, v*w, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(v.T @ v, np.eye(len(w)), atol=2e-13)
    np.testing.assert_allclose(w, eigenvalues, atol=2e-15, rtol=2e-13)
    assert np.count_nonzero(w > 1e-11*w[-1]) == 17
    host_w, host_v = gpu_eigh(a, device=device)
    np.testing.assert_allclose((host_v*host_w) @ host_v.T, a, atol=2e-14)


def test_nonfinite_factors_raise_without_fallback():
    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        _checked_host((np.array([np.nan]),), "SVD")


def test_gpu_sampled_wall_svd_has_no_large_cpu_fallback(device, monkeypatch):
    import scipy.linalg as la
    from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
    original_svd, original_eigh = la.svd, la.eigh
    def small_svd(a, *args, **kwargs):
        assert min(a.shape) < 64, "Large setup SVD fell back to CPU"
        return original_svd(a, *args, **kwargs)
    def small_eigh(a, *args, **kwargs):
        assert len(a) < 64, "Volume eigensystem fell back to CPU"
        return original_eigh(a, *args, **kwargs)
    monkeypatch.setattr(la, "svd", small_svd)
    monkeypatch.setattr(la, "eigh", small_eigh)
    p = ImmersedFlowPlan(nx=33, ny=25, assembly_device=device, basis_workers=2)
    step = p.stepper(.02, device=device)
    state = step.initial_state
    for k in range(5):
        state = step.step(state, k*.02)
    boundary, _ = p.arc.sample(190, offset=.381)
    _, u, v, *_ = p.evaluate(jax.device_get(state), boundary)
    assert np.max(np.hypot(u, v)) < 2e-7
    flux = p.out_weights @ (p.out_ops[0] @ jax.device_get(state)+p.out_lift[0])
    np.testing.assert_allclose(flux, 4/3, atol=2e-9)
