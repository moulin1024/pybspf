"""Independent geometry, source and layer-potential checks for the annulus model."""
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.special import hankel1, k0

from bspf_models.elliptic.embedded_poisson import SplineDomain
from bspf_models.elliptic.spline_annulus import (
    SplineBoundary, SplineAnnulus, FourierSourcePlan, FourierParticular, AnnulusPanelPlan,
)


@pytest.fixture(scope='module')
def domain():
    t = np.arange(8)*2*np.pi/8
    outer = SplineDomain(np.column_stack((np.cos(t), .85*np.sin(t))))
    inner = SplineDomain(np.array([.12, -.06])+np.column_stack((.25*np.cos(t), .2*np.sin(t))))
    return SplineAnnulus(outer.curve, inner.curve)


def homogeneous(sigma):
    center = np.array([.12, -.06])
    if sigma == 0:
        return lambda x: np.log(np.linalg.norm(x-center, axis=1))+.3*x[:, 0]
    if sigma > 0:
        return lambda x: k0(np.sqrt(sigma)*np.linalg.norm(x-center, axis=1))+.2*np.exp(np.sqrt(sigma)*x[:, 0])
    return lambda x: hankel1(0, np.sqrt(-sigma)*np.linalg.norm(x-center, axis=1))+.2*np.exp(1j*np.sqrt(-sigma)*x[:, 0])


@pytest.mark.parametrize('sigma', [0., 4., -4.])
def test_two_boundaries_near_targets_and_independent_quadrature(domain, sigma):
    p = AnnulusPanelPlan(domain, sigma, order=10)
    u = homogeneous(sigma)
    solution = p.solve((u, u))
    probes = [[.6, .1], [-.4, .2]]
    for component, b in enumerate(domain.boundaries):
        t = b.a+.237*(b.b-b.a)
        probes.append(b.curve(t)-1e-6*b.normal(t, 1 if component == 0 else -1))
        ts = b.a+(np.arange(11)+.371)/11*(b.b-b.a)
        expected = u(b.curve(ts))
        np.testing.assert_allclose(solution.boundary(component, ts), expected, atol=2e-6, rtol=2e-7)
    probes = np.array(probes)
    assert np.all(domain.contains(probes))
    actual = solution.interior(probes)
    np.testing.assert_allclose(actual, u(probes), atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(actual, solution.interior(probes, quadrature_order=56), atol=2e-9, rtol=2e-9)
    assert solution.training_residual < 2e-13
    if sigma == 0:
        assert abs(p.weights@solution.density) < 1e-11


def test_constant_and_log_hole_mode(domain):
    p = AnnulusPanelPlan(domain, order=10)
    u = lambda x: np.full(len(x), 2.3)
    sol = p.solve((u, u))
    np.testing.assert_allclose(sol.density, 0, atol=1e-10)
    np.testing.assert_allclose(sol.constant, 2.3, atol=1e-12)
    # The logarithmic singularity is in the excluded hole, hence harmonic on Omega.
    u = homogeneous(0.)
    sol = p.solve((u, u))
    np.testing.assert_allclose(sol.interior(np.array([[.5, .1], [-.5, 0.]])),
                               u(np.array([[.5, .1], [-.5, 0.]])), atol=1e-7)


def test_geometry_orientation_and_nonuniform_parameter_scale(domain):
    boundaries = []
    for b in domain.boundaries:
        curve = b.curve
        # Reversing all active coefficients and reflected knots reverses the
        # same geometry. Affine reparameterization removes integer-knot assumptions.
        knots = (b.a+b.b-curve.t[::-1])*1.7+2.4
        boundaries.append(BSpline(knots, curve.c[::-1], curve.k, extrapolate='periodic'))
    reversed_domain = SplineAnnulus(*boundaries)
    p = AnnulusPanelPlan(reversed_domain, -4., order=10)
    u = homogeneous(-4.)
    x = np.array([[.5, .1], [-.5, 0.]])
    np.testing.assert_allclose(p.solve((u, u)).interior(x), u(x), atol=2e-7, rtol=2e-7)


@pytest.fixture(scope='module')
def source_plan(domain):
    return FourierSourcePlan(domain, modes=8, samples=40)


@pytest.mark.parametrize('sigma', [0., 4., -4.])
def test_general_source_requires_only_domain_samples(domain, source_plan, sigma):
    # Construct a frequency from the source dictionary; exact solution is NOT
    # supplied as a particular. Boundary correction is nonzero.
    freq = source_plan.frequencies[len(source_plan.frequencies)//2+1]
    def u(x):
        return np.exp(1j*(x-source_plan.center)@freq)
    def f(x):
        assert np.all(domain.contains(x)), 'source was requested outside the PDE domain'
        return (np.dot(freq, freq)+sigma)*u(x)
    p = AnnulusPanelPlan(domain, sigma, order=10)
    sol = p.solve((u, u), source=f, source_plan=source_plan, source_tolerance=1e-8)
    x = np.array([[.5, .1], [-.5, 0.]])
    np.testing.assert_allclose(sol.interior(x), u(x), atol=2e-7, rtol=2e-7)
    assert sol.particular.stats['validation_relative_max'] < 1e-8


def test_unresolved_source_is_rejected(domain):
    plan = FourierSourcePlan(domain, modes=2, samples=20)
    with pytest.raises(ValueError, match='source validation failed'):
        plan.fit(lambda x: np.sin(100*x[:, 0]), 0., tolerance=1e-6)


def test_off_dictionary_high_frequency_source_and_independent_wall_probes(domain):
    plan = FourierSourcePlan(domain, modes=16, samples=72, padding=1.5)
    waves = np.array([[23.1, 4.7], [-7.2, 21.9], [3.7, -5.1]])
    phases = np.array([.31, 1.73, -.83])
    amplitudes = np.array([.7, -.4, 1.])
    points = [domain.sample(83, shift=.731)]
    for component, boundary in enumerate(domain.boundaries):
        t = boundary.a+(np.arange(311)+.619)/311*(boundary.b-boundary.a)
        for distance in (1e-7, 3e-3):
            points.append(boundary.curve(t)-distance*boundary.normal(t, 1 if component == 0 else -1))
    points = np.vstack(points)
    assert np.all(domain.contains(points))
    basis = plan.basis(points)
    for sigma in (0., 64., -64.):
        def source(x):
            assert np.all(domain.contains(x)), 'extension requested exterior data'
            return np.cos(x@waves.T+phases)@(amplitudes*(np.sum(waves**2, axis=1)+sigma))
        particular = plan.fit(source, sigma, tolerance=1e-7)
        expected = source(points)
        assert np.max(abs(basis@particular.coefficients-expected))/np.max(abs(expected)) < 1e-7
    assert plan.stats['collar_samples'] > 0


@pytest.mark.parametrize('mode', ['zero', 'resonant'])
def test_particular_pde_including_box_resonance(source_plan, mode):
    norm = np.sum(source_plan.frequencies**2, axis=1)
    j = int(np.argmin(norm)) if mode == 'zero' else int(np.flatnonzero(norm > 0)[len(norm)//3])
    sigma = -norm[j]
    c = np.zeros(len(norm), dtype=complex)
    c[j] = 1.
    particular = FourierParticular(source_plan, c, sigma, {})
    x = np.array([[.4, .2], [-.3, .2]])
    h = 2e-4
    lap = np.zeros(len(x), dtype=complex)
    for d in (0, 1):
        step = np.eye(2)[d]*h
        lap += (-particular(x+2*step)+16*particular(x+step)-30*particular(x)
                +16*particular(x-step)-particular(x-2*step))/(12*h*h)
    expected = source_plan.basis(x)[:, j]
    np.testing.assert_allclose(-lap+sigma*particular(x), expected, atol=2e-8, rtol=2e-8)


def test_invalid_geometry_is_rejected(domain):
    with pytest.raises(ValueError, match='inside'):
        SplineAnnulus(domain.boundaries[1], domain.boundaries[0])
    b = domain.boundaries[0].curve
    c = b.c.copy()
    c[-1, 0] += .1
    with pytest.raises(ValueError, match='close'):
        SplineBoundary(BSpline(b.t, c, b.k))


@pytest.mark.parametrize('sigma', [0., 4., -4.])
def test_close_boundaries(domain, sigma):
    outer = domain.boundaries[0].curve
    inner = BSpline(outer.t, .94*outer.c, outer.k, extrapolate='periodic')
    thin = SplineAnnulus(outer, inner)
    u = lambda x: (np.exp(np.sqrt(sigma)*x[:, 0]) if sigma > 0 else
                   np.exp(1j*np.sqrt(-sigma)*x[:, 0]) if sigma < 0 else x[:, 0]+.3*x[:, 1])
    plan = AnnulusPanelPlan(thin, sigma, order=10)
    solution = plan.solve((u, u))
    b = thin.boundaries[0]
    t = b.a+(np.arange(5)+.37)/5*(b.b-b.a)
    points = .97*b.curve(t)
    assert np.all(thin.contains(points))
    actual = solution.interior(points)
    np.testing.assert_allclose(actual, u(points), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual, solution.interior(points, quadrature_order=56), atol=2e-9, rtol=2e-9)


def test_non_cubic_geometry_and_knot_insertion():
    t = np.arange(8)*2*np.pi/8
    c = np.column_stack((np.cos(t), .9*np.sin(t)))
    outer = BSpline(np.arange(-5, 14), np.vstack((c, c[:5])), 5, extrapolate='periodic')
    inner = BSpline(outer.t, .3*outer.c, 5, extrapolate='periodic')
    outer = outer.insert_knot(.37)
    inner = inner.insert_knot(1.23)
    domain = SplineAnnulus(outer, inner)
    plan = AnnulusPanelPlan(domain, 4., order=10)
    u = lambda x: np.exp(2*x[:, 0])
    x = np.array([[.4, 0.], [-.4, .1]])
    np.testing.assert_allclose(plan.solve((u, u)).interior(x), u(x), atol=1e-7, rtol=1e-7)


def test_adaptive_reports_exhaustion_and_reduces_boundary_residual(domain):
    from bspf_models.elliptic.spline_annulus import adaptive_dirichlet
    u = homogeneous(0.)
    _, history = adaptive_dirichlet(domain, 0., (u, u), order=6,
                                    tolerance=1e-14, max_refinements=1)
    assert not history[-1]['converged']
    assert history[-1]['unknowns'] > history[0]['unknowns']
    assert history[-1]['boundary_indicator'] < history[0]['boundary_indicator']
    with pytest.raises(ValueError, match='every geometry knot'):
        AnnulusPanelPlan(domain, breaks=[[0., 8.], [0., 8.]])


def test_independent_inner_and_outer_dirichlet_data(domain):
    plan = AnnulusPanelPlan(domain, 0., order=10)
    zero = lambda x: np.zeros(len(x))
    one = lambda x: np.ones(len(x))
    sol = plan.solve((zero, one))
    near = []
    for component, boundary in enumerate(domain.boundaries):
        t = boundary.a+.231*(boundary.b-boundary.a)
        near.append(boundary.curve(t)-1e-5*boundary.normal(t, 1 if component == 0 else -1))
    values = sol.interior(np.array(near))
    assert abs(values[0]) < 1e-3
    assert abs(values[1]-1) < 1e-3
    inside = sol.interior(np.array([[.5, .1], [-.5, 0.]]))
    assert np.all((inside.real > 0) & (inside.real < 1))
    assert np.max(abs(inside.imag)) < 1e-12


@pytest.mark.parametrize('sigma', [0., 4., -4.])
def test_batched_quadrature_matches_scalar_self_near_far(domain, sigma):
    p = AnnulusPanelPlan(domain, sigma, order=8)
    points = [np.array([[.6,.1],[-.5,.15],[4.,3.]])]
    for component, b in enumerate(domain.boundaries):
        t = b.a+.237*(b.b-b.a)
        points.append(np.array([b.curve(t)-1e-6*b.normal(t, 1 if component==0 else -1)]))
    points = np.vstack(points)
    np.testing.assert_allclose(p.potential_matrix(points), p.potential_matrix(points,batched=False),
                               atol=3e-14,rtol=3e-13)
    np.testing.assert_allclose(p.matrix,p.potential_matrix(p.points,p.components,p.parameters,batched=False),
                               atol=3e-14,rtol=3e-13)


@pytest.mark.parametrize('sigma', [0., 4., -4.])
def test_streamed_layer_matches_dense_and_bounds_kernel_blocks(domain, sigma, monkeypatch):
    plan = AnnulusPanelPlan(domain, sigma, order=8)
    rng = np.random.default_rng(214)
    density = rng.normal(size=plan.count)+1j*rng.normal(size=plan.count)
    targets = [domain.sample(9, .237), np.array([[4., 3.]])]
    for component, b in enumerate(domain.boundaries):
        t = b.a+.237*(b.b-b.a)
        targets.append(np.array([b.curve(t)-1e-6*b.normal(t, 1 if component == 0 else -1)]))
    targets = np.vstack(targets)
    expected = plan.potential_matrix(targets)@density
    # Include self interactions and shared internal panel endpoints/jump averaging.
    params = np.r_[plan.parameters, domain.boundaries[0].knots[1:-1]]
    comps = np.r_[plan.components, np.zeros(len(domain.boundaries[0].knots)-2, dtype=int)]
    boundary = np.vstack((plan.points, domain.boundaries[0].curve(params[len(plan.points):])))
    expected_boundary = plan.potential_matrix(boundary, comps, params)@density
    def forbidden(*args, **kwargs):
        raise AssertionError('streamed apply must not assemble the dense operator')
    monkeypatch.setattr(plan, 'potential_matrix', forbidden)
    original_kernel = plan._kernel
    for size in (1, 7, 512):
        def checked_kernel(distance, dot):
            if distance.ndim == 2:
                assert distance.shape[0] <= size
            return original_kernel(distance, dot)
        monkeypatch.setattr(plan, '_kernel', checked_kernel)
        np.testing.assert_allclose(plan.apply_layer(targets, density, block_size=size), expected,
                                   atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(plan.apply_layer(boundary, density, comps, params, block_size=size),
                                   expected_boundary, atol=2e-13, rtol=2e-12)
    assert plan.apply_layer(np.empty((0, 2)), density).shape == (0,)
    with pytest.raises(ValueError, match='block_size'):
        plan.apply_layer(targets, density, block_size=0)
    with pytest.raises(ValueError, match='density'):
        plan.apply_layer(targets, density[:-1])


def test_solution_evaluation_avoids_dense_operator(domain, monkeypatch):
    plan = AnnulusPanelPlan(domain, order=10)
    exact = homogeneous(0.)
    solution = plan.solve((exact, exact))
    def forbidden(*args, **kwargs):
        raise AssertionError('evaluation must not build a dense target matrix')
    monkeypatch.setattr(plan, 'potential_matrix', forbidden)
    points = np.array([[.6, .1], [-.5, .15]])
    np.testing.assert_allclose(solution.interior(points, block_size=1), exact(points), atol=1e-7)
    b = domain.boundaries[0]
    t = b.a+np.array([.213, .427])*(b.b-b.a)
    np.testing.assert_allclose(solution.boundary(0, t, block_size=1), exact(b.curve(t)), atol=1e-7)
