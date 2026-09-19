"""GPU parity and transfer guards for channel IMEX evolution."""
from types import SimpleNamespace
import jax
import numpy as np
import scipy.linalg as la
import pytest

from bspf_jax.immersed_flow import ImmersedFlowPlan
from bspf_jax.immersed_flow_gpu import GPUImmersedFlowStepper


def test_device_steps_match_host_and_do_not_transfer():
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA device unavailable")
    # A deterministic reduced Galerkin system exercises nonlinear volume and
    # backflow loads without expensive geometry assembly. Physical geometry
    # and wall constraints are separately covered by test_immersed_flow.
    rng = np.random.default_rng(25)
    n, m, out = 9, 31, 7
    a = rng.normal(size=(n, n))
    mass = a.T @ a + np.eye(n)
    b = rng.normal(size=(n, n))
    linear = b.T @ b * 0.01
    p = SimpleNamespace(
        mass=mass, linear=linear, linear_lift=rng.normal(size=n)*0.01,
        dofs=n, operators_fluid=tuple(rng.normal(size=(m,n))*0.1 for _ in range(5)),
        lift_fields=tuple(rng.normal(size=m)*0.1 for _ in range(6)),
        weights=np.ones(m)/m, out_ops=tuple(rng.normal(size=(out,n)) for _ in range(2)),
        out_lift=(np.linspace(-1,1,out),np.zeros(out)), out_weights=np.ones(out)/out,
        sigma=np.ones(m), nu=0.01, peak=1., bounds=(-1.,5.,1.),
        points=rng.uniform(-1,1,size=(m,2)), stokes_state=rng.normal(size=n)*0.01,
        mass_factor=la.cho_factor(mass),
    )
    p.explicit = lambda state, time=0.: ImmersedFlowPlan.explicit(p,state,time)
    p.rhs = lambda state: ImmersedFlowPlan.rhs(p,state)
    dt = 0.002
    host_step = ImmersedFlowPlan.stepper(p,dt)
    gpu = GPUImmersedFlowStepper(p,dt,device)
    host = p.stokes_state.copy()
    state = gpu.initial_state
    for k in range(5):
        host = host_step.step(host,k*dt)
        with jax.transfer_guard("disallow"):
            state = gpu.step(state,k*dt)
            state.block_until_ready()
        assert state.devices() == {device}
    np.testing.assert_allclose(jax.device_get(state),host,rtol=2e-12,atol=2e-13)
    with jax.transfer_guard("disallow"):
        values = gpu.diagnostics(state)
        jax.block_until_ready(values)
    expected = ImmersedFlowPlan.diagnostics(p,host)
    for key,value in jax.device_get(values).items():
        np.testing.assert_allclose(value,expected[key],rtol=2e-11,atol=2e-13)
    assert all(x.devices() == {device} for x in jax.tree_util.tree_leaves(gpu.data))
    with pytest.raises(ValueError,match="Upload"):
        gpu.step(host)


@pytest.mark.parametrize("workspace_limit", [0, 2*1024**3])
def test_gpu_volume_assembly_matches_independent_energy_forms(monkeypatch, workspace_limit):
    monkeypatch.setattr("bspf_jax._immersed_assembly._FUSED_OPERATOR_BYTES", workspace_limit)
    from bspf_jax._immersed_assembly import GPUVolumeAssembly
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(81)
    ops = tuple(rng.normal(size=(71,13)) for _ in range(5))
    weights = rng.uniform(0.1,1.,71)
    assembly = GPUVolumeAssembly(ops,weights,device)
    mass, stiffness = assembly.gram()
    a = rng.normal(size=13)
    u,v,xy,yy,minus_xx = (op @ a for op in ops)
    np.testing.assert_allclose(a @ mass @ a, weights @ (u*u+v*v),rtol=2e-14)
    np.testing.assert_allclose(a @ stiffness @ a,weights @ (2*xy*xy+yy*yy+minus_xx*minus_xx),rtol=2e-14)
    transform = rng.normal(size=(13,9))
    for got,op in zip(assembly.transform(transform),ops):
        np.testing.assert_allclose(got,op @ transform,rtol=2e-13,atol=2e-13)
    assert all(op.devices()=={device} for op in assembly.ops)

    sigma = rng.uniform(0., 3., len(weights))
    transformed, reduced_mass, reduced_stiffness, sponge = (
        assembly.transform_and_reduce(transform, mass, stiffness, sigma)
    )
    c = rng.normal(size=transform.shape[1])
    u, v, xy, yy, minus_xx = (op @ (transform @ c) for op in ops)
    for matrix, energy in (
        (reduced_mass, weights @ (u*u + v*v)),
        (reduced_stiffness, weights @ (2*xy*xy + yy*yy + minus_xx*minus_xx)),
        (sponge, (weights*sigma) @ (u*u + v*v)),
    ):
        np.testing.assert_allclose(c @ matrix @ c, energy, rtol=2e-13)
    for got, op in zip(transformed, ops):
        np.testing.assert_allclose(got @ c, op @ (transform @ c), rtol=2e-12, atol=2e-12)
