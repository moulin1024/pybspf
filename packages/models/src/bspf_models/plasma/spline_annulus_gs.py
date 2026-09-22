"""Fixed-boundary Grad--Shafranov on a spline section/annulus, physical R>0.

psi=sqrt(R)*u transforms -Delta*psi=S(R,Z,psi) into
-Delta u = S/sqrt(R) - 3*u/(4*R**2). This is a Poisson Picard backend,
not a free-boundary equilibrium or a guaranteed convergent nonlinear solver.
Geometry uses local (R-major_radius,Z); all data callbacks use physical (R,Z).
"""
from dataclasses import dataclass

import numpy as np

from bspf_models.elliptic.spline_annulus import AnnulusPanelPlan, FourierSourcePlan


def _values(data, points):
    values = np.asarray(data(points) if callable(data) else data)
    if values.ndim == 0:
        values = np.full(len(points), values)
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError('data must return one finite scalar per point')
    if np.iscomplexobj(values) and np.max(abs(values.imag)) > 1e-10*max(1., np.max(abs(values))):
        raise ValueError('Grad--Shafranov data must be real')
    return values.real


class SplineAnnulusGSPlan:
    """Reusable geometry and source factorizations; all boundaries are Dirichlet.

    The source is sampled only in the physical annulus. Boundary resolution is
    fixed for each plan; independent boundary checks can reject an inadequate
    order/subdivision. Picard convergence is checked by the transformed PDE
    residual on source validation points, not just changes between iterates.
    """
    def __init__(self, domain, *, major_radius=3., modes=12, samples=56,
                 padding=2., order=12, subdivisions=1, source_plan=None, layer_block_size=512):
        if not np.isfinite(major_radius) or domain.bounds[0, 0]+major_radius <= 0:
            raise ValueError('physical annulus must stay at R>0')
        if not isinstance(layer_block_size, (int, np.integer)) or layer_block_size <= 0:
            raise ValueError('layer_block_size must be a positive integer')
        self.layer_block_size = int(layer_block_size)
        self.domain, self.major_radius = domain, float(major_radius)
        if source_plan is not None and source_plan.domain is not domain:
            raise ValueError("source plan belongs to a different domain")
        self.source_plan = source_plan if source_plan is not None else FourierSourcePlan(
            domain, modes=modes, samples=samples, padding=padding)
        self.poisson = AnnulusPanelPlan(domain, order=order, subdivisions=subdivisions)
        # Retain source grids, never their dense target-by-boundary operators.
        self._grids = (self.source_plan.points, self.source_plan.validation)

    def physical(self, local):
        return np.asarray(local)+np.array([self.major_radius, 0.])

    def _u(self, solution, local):
        return solution.interior(local, block_size=self.layer_block_size)

    def solve(self, source, boundary_flux=None, *, tolerance=1e-7,
              source_tolerance=1e-8, boundary_tolerance=1e-7,
              max_iterations=30, relaxation=1., initial_flux=0., nonlinear=False, callback=None):
        """Solve -Delta*psi=source. Raise on source/boundary/Picard failure.

        With nonlinear=False, source is scalar or source(physical_points).
        With nonlinear=True it is source(physical_points, psi_values).
        Boundary data are the pair (outer, inner), each scalar or callable.
        """
        if boundary_flux is None:
            boundary_flux = (0.,)*len(self.domain.boundaries)
        if (not 0 < relaxation <= 1 or not np.isfinite(tolerance) or tolerance <= 0
                or not np.isfinite(source_tolerance) or source_tolerance <= 0
                or not np.isfinite(boundary_tolerance) or boundary_tolerance <= 0
                or max_iterations < 1 or len(boundary_flux) != len(self.domain.boundaries)):
            raise ValueError('invalid iteration/boundary parameters')
        def boundary(data):
            def evaluate(local):
                physical = self.physical(local)
                return _values(data, physical)/np.sqrt(physical[:, 0])
            return evaluate
        boundary_data = tuple(boundary(data) for data in boundary_flux)
        grids = self._grids
        physical = [self.physical(x) for x in grids]
        roots = [np.sqrt(x[:, 0]) for x in physical]
        potential = [3/(4*x[:, 0]**2) for x in physical]
        states = [_values(initial_flux, x)/root for x, root in zip(physical, roots)]
        history = []
        for iteration in range(max_iterations):
            forcing = []
            for x, root, v, u in zip(physical, roots, potential, states):
                s = _values(source(x, root*u) if nonlinear else source, x)
                forcing.append(s/root-v*u)
            def effective(local):
                for grid, values in zip(grids, forcing):
                    if local is grid:
                        return values
                raise ValueError('unexpected source evaluation grid')
            # Inexact Picard: intermediate effective sources need not meet the
            # final fit gate. BOTH fit error and full PDE residual gate return.
            try:
                solution = self.poisson.solve(boundary_data, source=effective,
                                              source_plan=self.source_plan, source_tolerance=np.inf)
            except ValueError as exc:
                raise ValueError(f"GS iteration {iteration+1}: {exc}") from exc
            updated_complex = [self._u(solution, x) for x in grids]
            imaginary = max(float(np.max(abs(x.imag))) for x in updated_complex)
            updated = [x.real for x in updated_complex]
            x, root, v = physical[1], roots[1], potential[1]
            rhs = _values(source(x, root*updated[1]) if nonlinear else source, x)/root
            # Laplace of the layer correction is zero; the particular's Fourier
            # source gives -Delta u analytically, with no finite differences.
            represented = solution.particular.source_values(grids[1])
            residual = represented+v*updated_complex[1]-rhs
            scale = max(float(np.max(abs(rhs))), float(np.max(abs(v*updated_complex[1]))), 1e-14)
            error = float(np.max(abs(residual))/scale)
            row = dict(iteration=iteration+1, relative_pde_residual=error,
                       imaginary_u_max=imaginary,
                       source_relative_max=solution.particular.stats['validation_relative_max'])
            history.append(row)
            if callback is not None:
                callback(row)
            if not np.isfinite(error) or not np.isfinite(row["source_relative_max"]):
                raise RuntimeError("nonfinite GS iteration")
            if error <= tolerance and row["source_relative_max"] <= source_tolerance:
                solution.particular.stats["tolerance"] = source_tolerance
                result = SplineAnnulusGSSolution(self, solution, history)
                boundary_errors, boundary_scales = [], []
                # Different Gauss order and near-endpoint probes in every panel.
                probes = np.r_[np.polynomial.legendre.leggauss(self.poisson.order+3)[0], -.999, .999]
                for panel in self.poisson.panels:
                    t = panel.mid+panel.half*probes
                    local = panel.boundary.curve(t)
                    p = self.physical(local)
                    expected = _values(boundary_flux[panel.component], p)
                    actual = np.sqrt(p[:, 0])*solution.boundary(panel.component, t,
                                                              quadrature_order=self.poisson.qorder+8,
                                                              block_size=self.layer_block_size)
                    boundary_errors.extend(abs(actual-expected))
                    boundary_scales.extend(abs(expected))
                result.boundary_error = float(max(boundary_errors)/max(1., max(boundary_scales)))
                if result.boundary_error > boundary_tolerance:
                    raise ValueError(f'GS boundary validation failed: {result.boundary_error:.3e}; increase order/subdivisions')
                return result
            states = [(1-relaxation)*old+relaxation*new for old, new in zip(states, updated)]
        raise RuntimeError(f'GS Picard iteration failed after {max_iterations} steps: residual={error:.3e}, source_error={row["source_relative_max"]:.3e}')

    def solve_profiles(self, p_prime, ff_prime, *, mu0=1., boundary_flux=None, **kwargs):
        """Profiles are scalars or functions of psi, using -Delta*psi=mu0 R²p'+FF'."""
        if not np.isfinite(mu0) or mu0 <= 0:
            raise ValueError('mu0 must be finite and positive')
        def source(points, psi):
            p = p_prime(psi) if callable(p_prime) else p_prime
            f = ff_prime(psi) if callable(ff_prime) else ff_prime
            return mu0*points[:, 0]**2*p+f
        return self.solve(source, boundary_flux, nonlinear=True, **kwargs)


@dataclass
class SplineAnnulusGSSolution:
    plan: SplineAnnulusGSPlan
    transformed: object
    history: list
    boundary_error: float = float('nan')

    def flux(self, physical_points):
        """Evaluate flux at physical (R,Z) interior points; preserve numerical imaginary part."""
        physical = np.asarray(physical_points)
        local = physical-np.array([self.plan.major_radius, 0.])
        if not np.all(self.plan.domain.contains(local)):
            raise ValueError('flux evaluation requires physical interior points')
        return np.sqrt(physical[:, 0])*self.transformed.interior(local, block_size=self.plan.layer_block_size)


    def jets(self, physical_points, *, step=4e-4):
        """Fourth-order axial finite differences; mixed Hessian is second order.

        Diagnostic derivatives, with an explicit physical stencil size. Every
        stencil must remain inside the computational section. Not used in PDE
        residual acceptance, which retains its analytic source identity.
        """
        x = np.asarray(physical_points, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(step) or step <= 0:
            raise ValueError('invalid diagnostic points or step')
        offsets = np.array([[0,0],[1,0],[-1,0],[2,0],[-2,0],
                            [0,1],[0,-1],[0,2],[0,-2],[1,1],[1,-1],[-1,1],[-1,-1]])
        probes = (x[:,None,:]+step*offsets).reshape(-1,2)
        f = self.flux(probes).real.reshape(len(x),len(offsets))
        g = np.column_stack(((f[:,4]-8*f[:,2]+8*f[:,1]-f[:,3])/(12*step),
                             (f[:,8]-8*f[:,6]+8*f[:,5]-f[:,7])/(12*step)))
        h = np.empty((len(x),2,2))
        h[:,0,0] = (-f[:,3]+16*f[:,1]-30*f[:,0]+16*f[:,2]-f[:,4])/(12*step**2)
        h[:,1,1] = (-f[:,7]+16*f[:,5]-30*f[:,0]+16*f[:,6]-f[:,8])/(12*step**2)
        h[:,0,1] = h[:,1,0] = (f[:,9]-f[:,10]-f[:,11]+f[:,12])/(4*step**2)
        return f[:,0], g, h

    def magnetic_axis(self, guess=None, *, sign=1, step=4e-4, tolerance=1e-6):
        """Locate one nondegenerate O point; sign=1 selects a flux maximum.

        Multiple axes/separatrix classification is outside this local diagnostic.
        The geometric toroidal axis R=0 remains excluded.
        """
        from scipy.optimize import root
        if sign not in (-1,1) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('invalid axis sign or tolerance')
        if guess is None:
            points = self.plan.physical(self.plan.domain.sample(17,.371))
            guess = points[np.argmax(sign*self.flux(points).real)]
        def gradient(x):
            return self.jets(np.asarray(x)[None,:],step=step)[1][0]
        answer = root(gradient, np.asarray(guess,dtype=float), tol=1e-9)
        value, grad, hess = self.jets(answer.x[None,:],step=step)
        residual = float(np.linalg.norm(grad[0]))
        eigenvalues = np.linalg.eigvalsh(hess[0])
        if (not answer.success or residual > tolerance or np.any(sign*eigenvalues >= 0)):
            raise ValueError('magnetic axis search did not find a resolved nondegenerate O point')
        return dict(position=answer.x.tolist(), flux=float(value[0]),
                    gradient_norm=residual, hessian_eigenvalues=eigenvalues.tolist(),
                    derivative_step=step)
