"""NGSolve-based high-order FEM baseline for 2D Poisson problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ngsolve import BND, BilinearForm, CGSolver, GridFunction, H1, LinearForm, Preconditioner, dx, grad
from ngsolve.meshes import MakeStructured2DMesh


@dataclass
class NGSolvePoisson2DBaselineSolver:
    """Structured quadrilateral H1 FEM baseline on a rectangular domain."""

    mesh: object
    order: int
    n_elem_x: int
    n_elem_y: int
    domain_x: tuple[float, float]
    domain_y: tuple[float, float]

    @classmethod
    def from_domain(
        cls,
        *,
        domain_x: tuple[float, float],
        domain_y: tuple[float, float],
        n_elem_x: int,
        n_elem_y: int,
        order: int = 4,
    ) -> "NGSolvePoisson2DBaselineSolver":
        if n_elem_x < 1 or n_elem_y < 1:
            raise ValueError("n_elem_x and n_elem_y must both be positive.")
        if order < 1:
            raise ValueError("order must be positive.")

        x0, x1 = map(float, domain_x)
        y0, y1 = map(float, domain_y)
        mesh = MakeStructured2DMesh(
            quads=True,
            nx=int(n_elem_x),
            ny=int(n_elem_y),
            mapping=lambda x, y: (x0 + (x1 - x0) * x, y0 + (y1 - y0) * y),
        )
        return cls(
            mesh=mesh,
            order=int(order),
            n_elem_x=int(n_elem_x),
            n_elem_y=int(n_elem_y),
            domain_x=(x0, x1),
            domain_y=(y0, y1),
        )

    def solve(
        self,
        *,
        rhs_cf,
        dirichlet_cf,
        initial_guess_cf=None,
        preconditioner: str = "multigrid",
        tol: float = 1.0e-10,
        maxsteps: int = 500,
        printrates: bool = False,
    ) -> tuple[GridFunction, dict[str, object]]:
        """Solve ``Delta u = rhs_cf`` with Dirichlet data ``dirichlet_cf``."""
        fes = H1(self.mesh, order=self.order, dirichlet="left|right|bottom|top")
        u, v = fes.TnT()

        bilinear = BilinearForm(fes)
        bilinear += grad(u) * grad(v) * dx

        linear = LinearForm(fes)
        linear += (-rhs_cf) * v * dx

        solution = GridFunction(fes)
        if initial_guess_cf is not None:
            solution.Set(initial_guess_cf)
        solution.Set(dirichlet_cf, BND)

        pre = Preconditioner(bilinear, preconditioner)

        bilinear.Assemble()
        linear.Assemble()
        pre.Update()

        residual = linear.vec.CreateVector()
        residual.data = linear.vec - bilinear.mat * solution.vec
        initial_residual_norm = float(residual.Norm())
        cg = CGSolver(
            bilinear.mat,
            pre.mat,
            maxsteps=maxsteps,
            precision=tol,
            printrates=printrates,
        )
        solution.vec.data += cg * residual
        residual.data = linear.vec - bilinear.mat * solution.vec

        metadata = {
            "order": self.order,
            "n_elem_x": self.n_elem_x,
            "n_elem_y": self.n_elem_y,
            "ndof": int(fes.ndof),
            "preconditioner": preconditioner,
            "tol": float(tol),
            "maxsteps": int(maxsteps),
            "cg_steps": int(cg.GetSteps()),
            "initial_residual_norm": initial_residual_norm,
            "final_residual_norm": float(residual.Norm()),
            "used_initial_guess": bool(initial_guess_cf is not None),
        }
        return solution, metadata

    def sample_on_grid(self, solution: GridFunction, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate one NGSolve solution on a tensor-product grid."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        values = np.empty((y.size, x.size), dtype=np.float64)
        for j, yj in enumerate(y):
            for i, xi in enumerate(x):
                values[j, i] = solution(self.mesh(float(xi), float(yj)))
        return values


__all__ = ["NGSolvePoisson2DBaselineSolver"]
