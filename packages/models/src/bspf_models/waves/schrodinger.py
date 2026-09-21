"""Schrödinger and nonlinear Schrödinger propagation."""
from numbers import Integral
import jax
import jax.numpy as jnp
from pybspf.time_integration import _integrate_modal_rk4
def integrate_schrodinger(mass, hamiltonian, initial, times):
    """Evolve i M y_t = H y for constant Hermitian H and positive-definite M.

    Pure JAX generalized Hermitian eigendecomposition followed by exact modal
    phases removes time-discretization error. This dense O(N^3) method is for
    small autonomous linear systems, not nonlinear or time-dependent models.
    ``initial`` is a vector at times[0]; output shape is (len(times), N).
    Callers ensure Hermitian matrices, positive-definite mass, and finite real
    times. Spatial discretization and floating-point errors remain. Quadratic
    norm y* M y is preserved to roundoff. The function can be jitted.
    """
    from jax.scipy.linalg import solve_triangular
    mass, hamiltonian = jnp.asarray(mass), jnp.asarray(hamiltonian)
    initial, times = jnp.asarray(initial), jnp.asarray(times)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or hamiltonian.shape != mass.shape:
        raise ValueError('mass and hamiltonian must be matching square matrices')
    if initial.ndim != 1 or initial.shape[0] != mass.shape[0]:
        raise ValueError('initial must be a vector matching the matrices')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    lower = jnp.linalg.cholesky(mass)
    left = solve_triangular(lower, hamiltonian, lower=True)
    transformed = solve_triangular(lower, left.conj().T, lower=True).conj().T
    energies, modes = jnp.linalg.eigh((transformed+transformed.conj().T)/2)
    coefficients = modes.conj().T@(lower.conj().T@initial)
    physical_modes = solve_triangular(lower.conj().T, modes, lower=False)
    phases = jnp.exp(-1j*(times-times[0])[:, None]*energies)
    return (phases*coefficients)@physical_modes.T


def integrate_nlse(weak, initial, times, *, coupling=2.0, substeps=1):
    """Cubic NLSE i M q_t = K q - g Q* W (|Qq|² Qq), with g=coupling.

    ``weak`` is a Galerkin1D assembled with resolved quadrature; Q=weak.values
    evaluates its trial functions at quadrature points. Positive g is focusing.
    Initial data are free nodal coefficients; physical samples are E @ q.
    The return shape is (len(times), Nfree), including initial exactly.

    A fourth-order interaction-picture (Lawson RK4) scheme treats the constant
    linear part with exact modal phases and evaluates the cubic projection at
    four stages. This avoids the linear explicit-RK4 stability limit, but is
    not unconditionally stable or exactly norm/energy conserving. Refine the
    step and monitor both invariants; highly oscillatory modes can require
    smaller steps for accuracy. Dense quadrature costs O(N * Nquad) per stage.
    Caller supplies increasing finite real times and a finite real coupling.
    substeps is a positive static integer. Supports outer JIT.
    """
    from jax.scipy.linalg import solve_triangular
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError('substeps must be a positive static integer')
    initial, times = jnp.asarray(initial), jnp.asarray(times)
    if initial.ndim != 1 or initial.shape[0] != weak.mass.shape[0]:
        raise ValueError('initial must be a vector of free nodal coefficients')
    if times.ndim != 1 or times.size < 1 or jnp.iscomplexobj(times):
        raise ValueError('times must be a nonempty real 1D array')
    coupling = jnp.asarray(coupling)
    if coupling.ndim != 0 or jnp.iscomplexobj(coupling):
        raise ValueError('coupling must be a real scalar')
    initial = initial.astype(jnp.result_type(initial, 1j))
    lower = jnp.linalg.cholesky(weak.mass)
    left = solve_triangular(lower, weak.stiffness, lower=True)
    transformed = solve_triangular(lower, left.conj().T, lower=True).conj().T
    energies, modes = jnp.linalg.eigh((transformed+transformed.conj().T)/2)
    physical_modes = solve_triangular(lower.conj().T, modes, lower=False)
    quadrature_modes = weak.values@physical_modes
    projection = quadrature_modes.conj().T*weak.quadrature_weights
    modal = modes.conj().T@(lower.conj().T@initial)

    def nonlinear(c):
        field = quadrature_modes@c
        return 1j*coupling*(projection@(jnp.abs(field)**2*field))

    history = _integrate_modal_rk4(energies, nonlinear, modal, times, substeps)
    return jnp.concatenate((initial[None], history[1:]@physical_modes.T), axis=0)
