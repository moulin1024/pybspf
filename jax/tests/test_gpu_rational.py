"""GPU rational construction and derivatives against independent host recurrences."""
import jax
import numpy as np
import pytest

from bspf_jax.rational_stokes import RationalBasis, RationalStokesExtension
from bspf_jax._gpu_rational import evaluate_blocks, unpad_blocks, construct_block

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def device():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("GPU device unavailable")


def test_gpu_arnoldi_and_differentiated_recurrences(device):
    theta = np.arange(257)*2*np.pi/257
    z = 2*np.cos(theta)+1j*np.sin(theta)
    poles = [np.array([3+1j, 3-1j, -3+1j]), np.empty(0, complex)]
    cpu = RationalBasis(z, 32, poles, 24)
    gpu = RationalBasis(z, 32, poles, 24, device=device)
    for h, expected in zip(gpu.hessenberg, cpu.hessenberg):
        np.testing.assert_allclose(h, expected, atol=8e-14, rtol=8e-14)
    points = z[::3]*.98
    reference = cpu.evaluate(points)
    result = gpu.evaluate_gpu(points, device)
    for actual, expected in zip(result, reference):
        assert actual.devices() == {device}
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-12)
    data = gpu._gpu_data[device]
    zd = jax.device_put(points, device)
    sizes = tuple(n for _, n in gpu.blocks)
    pd = jax.device_put(np.zeros(32, complex), device)
    with jax.transfer_guard("disallow"):
        values = unpad_blocks(evaluate_blocks(zd, *data), sizes=sizes)
        h = construct_block(zd, pd, degree=32, polynomial=True)
        jax.block_until_ready((values, h))
    assert h.devices() == {device}
    assert all(v.devices() == {device} for v in values)


def test_gpu_rows_do_not_call_host_evaluator(device, monkeypatch):
    from bspf_jax.immersed_poisson import EllipticHole
    extension = RationalStokesExtension(
        (-2., 4., 1.), EllipticHole(center=(0., 0.), axes=(.3, .2)), degree=12,
        laurent=8, corner_poles=4, samples=64, assembly_device=device,
        basis_construction="gpu")
    points = np.array([[-1., -.4], [-.5, 0.], [.5, 0.], [1., .6], [2., -.8]])
    rng = np.random.default_rng(74)
    coefficients = rng.normal(size=(len(extension.columns), 2))
    expected = extension.evaluate(points, coefficients)
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU evaluation invoked host rational recurrences")
    monkeypatch.setattr(extension, "stream_rows", forbidden)
    monkeypatch.setattr(extension.basis, "evaluate", forbidden)
    for coeff, ref in ((coefficients, expected),
                       (coefficients[:, 0], tuple(a[:, 0] for a in expected))):
        actual = extension.evaluate(points, coeff, batch_size=3, device=device, return_device=True)
        for a, b in zip(actual, ref):
            assert a.devices() == {device}
            np.testing.assert_allclose(a, b, atol=2e-10, rtol=2e-12)
        empty = extension.evaluate(points[:0], coeff, device=device, return_device=True)
        assert all(a.shape == (0,)+coeff.shape[1:] for a in empty)
    with pytest.raises(ValueError, match="batch_size"):
        extension.evaluate(points, coefficients, batch_size=0, device=device)
