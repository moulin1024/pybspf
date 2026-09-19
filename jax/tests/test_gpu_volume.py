"""Resident GPU volume assembly agrees with direct tensor-field evaluation."""
import jax
import numpy as np
import pytest

from bspf_jax._immersed_assembly import _tensor_operators, _prepare_volume, prepare_gpu_volume, gpu_tensor_operators


@pytest.mark.parametrize("consume", [False, True])
@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_resident_volume_fields_match_direct_tensor_contractions(rational, constrained, consume, monkeypatch):
    monkeypatch.setattr("bspf_jax._immersed_assembly._FUSED_OPERATOR_BYTES", 0)
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("GPU device unavailable")
    rng = np.random.default_rng(64)
    n, nx, ny, rank = 31, 5, 4, 7
    factors = tuple(tuple(rng.normal(size=(n, size)) for _ in range(3))
                    for size in (nx, ny))
    correction = tuple(rng.normal(size=(n, rank+1)) for _ in range(6)) if rational else None
    mapping = rng.normal(size=(rank, nx*ny)) if rational else None
    base = tuple(rng.normal(size=n) for _ in range(6))
    lift = rng.normal(size=nx*ny)
    scale = rng.uniform(.1, 2., nx*ny)
    constraints = rng.normal(size=(nx*ny, 9)) if constrained else None
    state = rng.normal(size=9 if constrained else nx*ny)
    coeff = lift+(constraints @ state if constrained else scale*state)
    c = coeff.reshape(nx, ny)
    (x, dx, xx), (y, dy, yy) = factors
    expected = [np.sum((a @ c)*b, axis=1) for a, b in
                ((x, y), (x, dy), (-dx, y), (dx, dy), (x, yy), (-xx, y))]
    expected = [value+b for value, b in zip(expected, base)]
    if rational:
        expected = [v+r[:, :-1] @ (mapping @ coeff)+r[:, -1]
                    for v, r in zip(expected, correction)]
    data = jax.device_put((factors, correction, mapping, base), device)
    prep = jax.device_put((lift, scale, constraints), device)
    with jax.transfer_guard("disallow"):
        operators, fields = _tensor_operators(*data)
        if not consume:
            lifts, raw = _prepare_volume(operators, fields, *prep)
            jax.block_until_ready((lifts, raw))
    if consume:
        # Exercise the sequential production assembly, including rational lift.
        operators, fields = gpu_tensor_operators(*data, device)
        lifts, raw = prepare_gpu_volume(operators, fields, *prep, device)
    assert all(a.devices() == {device} for a in raw)
    host_raw, host_lifts = jax.device_get((raw, lifts))
    for op, b, value in zip(host_raw, host_lifts[1:], expected[1:]):
        np.testing.assert_allclose(op @ state+b, value, atol=5e-12, rtol=5e-13)
