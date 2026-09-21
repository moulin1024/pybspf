"""B-spline/Fourier operators and compatible problem-specific solver exports."""

# Re-export the canonical grid type and the current operator wrappers so users
# can import from ``pybspf`` directly instead of depending on file layout.
from .basis import BSplineValues, make_bspline_basis_values
from .bspf_split import bspf_kkt_1d_decompose_precompute, split2d_kkt_directional
from .grid import Grid1D
from .galerkin import ClosedBSPFLine
from .ops.differentiation import DerivativeResult
from .operators import BSPF1D, BSPF2D, PiecewiseBSPF1D, bspf1d, bspf2d
from .time_integration import integrate_rk4

__all__ = [
    "ClosedBSPFLine",
    "BSPF1D",
    "BSPF2D",
    "BSplineValues",
    "DerivativeResult",
    "Grid1D",
    "PiecewiseBSPF1D",
    "Poisson1DDirichletSolver",
    "Poisson2DDirichletSolver",
    "PressurePoisson2D",
    "PressurePoisson2DResult",
    "bspf1d",
    "bspf2d",
    "bspf_kkt_1d_decompose_precompute",
    "bspf_kkt_poisson_neumann_apply",
    "bspf_kkt_poisson_neumann_precompute",
    "integrate_rk4",
    "make_bspline_basis_values",
    "split2d_kkt_directional",
]


# Solver imports are deferred so core operators do not load research workflows.
_SOLVER_EXPORTS = {
    "Poisson1DDirichletSolver": ".solvers.poisson1d",
    "Poisson2DDirichletSolver": ".solvers.poisson2d",
    "PressurePoisson2D": ".solvers.pressure_poisson2d",
    "PressurePoisson2DResult": ".solvers.pressure_poisson2d",
    "bspf_kkt_poisson_neumann_apply": ".solvers.poisson_neumann_bspf",
    "bspf_kkt_poisson_neumann_precompute": ".solvers.poisson_neumann_bspf",
}


def __getattr__(name):
    if name not in _SOLVER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(_SOLVER_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
