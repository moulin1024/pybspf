"""Implicit Q with a bounded dense rank core and protected polynomial modes."""

import numpy as np
import pytest
import scipy.sparse as sp

from bspf_models.fluids._embedded.sparse_qr import SparseNullSpace, _executable
from bspf_models.fluids.embedded_navier_stokes import plan_embedded_navier_stokes2d, geometry_prior_options


@pytest.fixture(scope="module", autouse=True)
def require_native_sparse_qr():
    try:
        _executable()
    except RuntimeError as error:
        pytest.skip(str(error))


def exact(p):
    x, y = np.asarray(p).T
    return np.column_stack(
        (
            0.02 * x * x,
            -0.04 * x * y,
            0.3 + x + y,
            0.04 * x,
            0 * x,
            -0.04 * y,
            -0.04 * x,
        )
    )


def boundary(p, tag):
    return exact(p)[:, :2]


def test_thin_ellipse_prior_sources_stay_inside_solid():
    from types import SimpleNamespace
    from bspf_models.fluids._embedded.prior_basis import geometry_dictionary
    from bspf_models.fluids._embedded.geometry import ObstacleGrid
    axes = np.array([1., .1])
    grid = SimpleNamespace(center=np.zeros(2), axes=axes,
                           edges=[np.array([-60., 60.]), np.array([-40., 40.])])
    grid.ellipse = lambda theta: ObstacleGrid.ellipse(grid, theta)
    dictionary = geometry_dictionary(np.array([-1.2, -.2]), np.array([2.4, .4]),
                                     grid, corners=False, surface_samples=32, levels=4)
    assert np.all(np.sum((dictionary.centers / axes)**2, axis=1) < 1)


def traction(p, n, nu=0.38):
    f = exact(p)
    return (
        nu
        * np.column_stack(
            (
                f[:, 3] * n[:, 0] + f[:, 4] * n[:, 1],
                f[:, 5] * n[:, 0] + f[:, 6] * n[:, 1],
            )
        )
        - f[:, 2, None] * n
    )


def test_implicit_qr_actions_with_redundant_constraints():
    rng = np.random.default_rng(21)
    a = rng.normal(size=(6, 10))
    a = np.vstack((a, a[0], 2 * a[3]))
    q = SparseNullSpace(sp.csr_matrix(a), 1e-10)
    try:
        u = rng.normal(size=10)
        zero = q.lift(q.restrict(u))
        affine = q.affine(a @ u)
        assert q.rank == 6
        block = rng.normal(size=(10, 3))
        np.testing.assert_allclose(q.full_q_block(block), np.column_stack([q.full_q(x) for x in block.T]), atol=1e-13)
        # The exported range core and implicit Q must come from the SAME QR.
        transformed = q.full_qt(a.T @ np.arange(len(a)))
        ordered = np.arange(len(a))[q.row_order][q.column_order]
        np.testing.assert_allclose(transformed[:q.rank], q.R @ ordered, atol=1e-12)
        target = rng.normal(size=len(a))
        np.testing.assert_allclose(target @ q.affine_transpose(u), u @ q.affine(target), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a @ zero, 0, atol=1e-12)
        np.testing.assert_allclose(a @ affine, a @ u, atol=1e-12)
        np.testing.assert_allclose(zero @ affine, 0, atol=1e-12)
    finally:
        q.close()


def test_array_projector_matches_implicit_and_jax_actions(monkeypatch):
    from bspf_models.fluids._embedded.sparse_qr import ArrayNullSpace
    p = plan_embedded_navier_stokes2d(boundary, dt=.001, viscosity=.38,
        cells=3, degree=3, order=40, convection_order=40, outflow=True,
        constraint_backend="implicit_qr", linear_backend="host_sparse",
        constraint_rank_tolerance=1e-9, wall_enforcement="constraint", projector_backend="array")
    try:
        implicit, array = p.spatial.projector, p.projector
        rng = np.random.default_rng(92)
        x, z = rng.normal(size=array.dimension), rng.normal(size=array.free)
        np.testing.assert_allclose(array.restrict(x), implicit.restrict(x), atol=1e-12)
        np.testing.assert_allclose(array.lift(z), implicit.lift(z), atol=1e-12)
        restrict, lift = array.jax_actions()
        np.testing.assert_allclose(restrict(x), array.restrict(x), atol=1e-12)
        np.testing.assert_allclose(lift(z), array.lift(z), atol=1e-12)
        with pytest.raises(ValueError, match="exceeding cap"):
            ArrayNullSpace(implicit, max_bytes=1)
        def forbid_worker(*args, **kwargs):
            raise AssertionError("Array time stepping must not call the QR worker")
        monkeypatch.setattr(implicit.inner.qr, "_action", forbid_worker)
        state = p.stokes_initial_state(np.tile([1-.04*.38, 1.], (len(p.force_points), 1)),
                                      traction(p.traction_points, p.traction_normals))
        x, y = p.force_points.T
        forcing = np.column_stack((1-.04*.38+2*.02**2*x**3, 1+2*.02**2*x*x*y))
        load = p.load(forcing, traction(p.traction_points, p.traction_normals))
        state = p.advance(state, 3, load=lambda t: load)
        points = np.array([[-.8,.4],[.7,.6]])
        np.testing.assert_allclose(p.evaluate(points,state), exact(points), atol=2e-8)
        assert p.wall_slip_error(state.coefficients) < 1e-10
    finally:
        p.close()


def test_enriched_polynomial_velocity_survives_local_and_global_elimination():
    p = plan_embedded_navier_stokes2d(
        boundary, dt=.001, viscosity=.38, cells=3, degree=5,
        order=56, convection_order=40, outflow=True,
        linear_backend="host_sparse", constraint_backend="implicit_qr",
        constraint_rank_tolerance=1e-9, **geometry_prior_options(5),
    )
    try:
        s = p.spatial
        coefficients = np.zeros(2 * p.nv)
        for root, (x, w) in s.volume.items():
            v = s.vbasis[root].evaluate(x, 0)[0]
            c = np.linalg.lstsq(np.sqrt(w)[:, None] * v,
                                np.sqrt(w)[:, None] * boundary(x, "outer"), rcond=1e-13)[0]
            for d in range(2):
                coefficients[s.vi[root] + d * p.nv] = c[:, d]
        q = s.projector
        reconstructed = s.affine_velocity + q.lift(q.restrict(coefficients - s.affine_velocity))
        np.testing.assert_allclose(reconstructed, coefficients, atol=2e-8, rtol=0)
        state = p.stokes_initial_state(np.tile([1-.04*.38, 1.], (len(p.force_points), 1)),
                                       traction(p.traction_points, p.traction_normals))
        points = np.array([[-.8, .4], [.7, .6], [.6, -.7]])
        np.testing.assert_allclose(p.evaluate(points, state), exact(points), atol=2e-7, rtol=0)
    finally:
        p.close()


@pytest.mark.parametrize("outflow", [True, False])
@pytest.mark.parametrize("rank_tolerance", [1e-5, 1e-9, 1e-11])
@pytest.mark.parametrize("wall_penalty_factor,wall_enforcement", [(1., "nitsche"), (64., "nitsche"), (1., "constraint")])
def test_manufactured_flow_with_sparse_pressure_recovery(outflow, rank_tolerance, wall_penalty_factor, wall_enforcement, monkeypatch):
    def forbid_dense(*args, **kwargs):
        raise AssertionError("Global sparse matrices must not be materialized as dense")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbid_dense)
    monkeypatch.setattr(sp.csc_matrix, "toarray", forbid_dense)
    p = plan_embedded_navier_stokes2d(
        boundary,
        dt=0.001,
        viscosity=0.38,
        cells=3,
        degree=3,
        order=40,
        convection_order=40,
        outflow=outflow,
        linear_backend="host_sparse",
        constraint_backend="implicit_qr",
        constraint_rank_tolerance=rank_tolerance,
        wall_penalty_factor=wall_penalty_factor,
        wall_enforcement=wall_enforcement,
    )
    try:
        initial = p.stokes_initial_state(
            np.tile([1 - 0.04 * 0.38, 1.0], (len(p.force_points), 1)),
            traction(p.traction_points, p.traction_normals),
        )
        x, y = p.force_points.T
        force = np.column_stack(
            (1 - 0.04 * 0.38 + 2 * 0.02**2 * x**3, 1 + 2 * 0.02**2 * x * x * y)
        )
        load = p.load(force, traction(p.traction_points, p.traction_normals))
        state = p.advance(initial, 3, load=lambda t: load)
        points = np.array([[-0.8, 0.4], [0.7, 0.6], [0.6, -0.7]])
        want = exact(points)
        if not outflow:
            want[:, 2] -= 0.3 - np.pi * 0.27 * 0.19 * (0.13 - 0.07) / (
                4 - np.pi * 0.27 * 0.19
            )
        np.testing.assert_allclose(p.evaluate(points, state), want, atol=2e-8)
        div, jump, normal, valid = p.incompressibility_errors(state.coefficients)
        assert valid and max(div, jump, normal) < 1e-10
        if wall_enforcement == "constraint":
            assert p.wall_slip_error(state.coefficients) < 1e-10
        assert p.spatial.A is None
        assert sp.issparse(p.spatial.raw_constraints)
        bad = state.coefficients.copy()
        bad[5] += 0.1
        with pytest.raises(ValueError, match="incompressibility"):
            p.initialize(bad)
    finally:
        p.close()
