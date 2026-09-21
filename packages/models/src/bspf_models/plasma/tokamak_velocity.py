"""Linear free-boundary MHD in the shared stream NS velocity-stage framework.

Independent state: displacement q, velocity v, and magnetic coefficients b.
q'=v, b'=C v, M v'=-C.T b+P q-vacuum_traction(q).
The magnetic derivative is assembled from curl(u cross B0), using the shared
BSPF differential kernels. The old displacement stiffness is only a reference.
This is NOT nonlinear Navier--Stokes/MHD or a Cartesian pressure projection.
"""

from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
from pybspf.time_integration import rk4_stages
from bspf_models.fluids.stream_navier_stokes import stream_ns_inertia_solve
from bspf_models.plasma.tokamak_vacuum import magnetic_displacement


@dataclass
class TokamakVelocityPlan:
    spatial: object
    induction: np.ndarray
    magnetic_basis_weighted: np.ndarray
    equilibrium_drive: np.ndarray
    inertia: object
    stiffness_reference_error: float
    maximum_frequency: float

    @property
    def size(self):
        return self.induction.shape[1]

    def initial_state(self, displacement, velocity=None):
        velocity = np.zeros_like(displacement) if velocity is None else velocity
        return np.concatenate((displacement, velocity, self.induction @ displacement))

    def split(self, state):
        if state.shape != (3 * self.size,):
            raise ValueError(
                "State must contain displacement, velocity and magnetic coefficients"
            )
        return np.split(state, 3)

    def vacuum_force(self, q):
        trace = self.spatial.trace @ q
        return self.spatial.trace.T @ (self.spatial.vacuum.boundary_energy @ trace)

    def rhs(self, state):
        q, v, b = self.split(state)
        lorentz = (
            -self.induction.T @ b + self.equilibrium_drive @ q - self.vacuum_force(q)
        )
        acceleration = stream_ns_inertia_solve(self.inertia, lorentz[:, None])[:, 0]
        return np.concatenate((v, acceleration, self.induction @ v))

    def step(self, state, dt):
        if not np.isfinite(dt) or dt <= 0 or dt * self.maximum_frequency > 2.7:
            raise ValueError("RK4 step must resolve the fastest linear Alfven mode")
        return rk4_stages(state, dt, lambda x: (self.rhs(x), None))[0]

    def energy(self, state):
        q, v, b = self.split(state)
        return float(
            (v @ v + b @ b - q @ self.equilibrium_drive @ q + q @ self.vacuum_force(q))
            / 2
        )

    def magnetic_fields(self, magnetic_coefficients):
        weights = self.spatial.quadrature_weights
        weighted = self.magnetic_basis_weighted @ magnetic_coefficients
        return np.column_stack(np.split(weighted, 3)) / np.sqrt(weights[:, None])


def plan_tokamak_velocity(model):
    points, weights = model.quadrature_points, model.quadrature_weights
    raw = model.basis.evaluate(points)
    eq = model.evaluator.evaluate(points, hessian=True)
    _, qr, qz, qp = magnetic_displacement(raw, eq, points, model.toroidal_f)
    transform = model.transform
    f = {key: value @ transform for key, value in raw.items()}
    qr, qz, qp = [a @ transform for a in (qr, qz, qp)]
    weighted = np.vstack([np.sqrt(weights[:, None]) * a for a in (qr, qz, qp)])
    # Orthogonal magnetic coordinates preserve the full resolved induction range.
    # No eigenmode selection or prescribed exponential evolution is performed.
    magnetic_basis, induction = la.qr(weighted, mode="economic")
    j = model.evaluator.alpha * points[:, 0] * np.maximum(eq[0], 0) ** 2
    source = f["xr"].T @ ((weights * j)[:, None] * qz) - f["xz"].T @ (
        (weights * j)[:, None] * qr
    )
    drive = (source + source.T) / 2
    vacuum = model.trace.T @ model.vacuum.boundary_energy @ model.trace
    stiffness = induction.T @ induction - drive + vacuum
    error = la.norm(stiffness - model.stiffness) / la.norm(model.stiffness)
    maximum = np.sqrt(max(la.eigvalsh((stiffness + stiffness.T) / 2)[-1], 0))
    # Plasma restricted-mass SVD gives identity inertia in retained coordinates.
    inertia = SimpleNamespace(
        denominator=np.ones((len(stiffness), 1)), inertia_transform=None
    )
    return TokamakVelocityPlan(
        model, induction, magnetic_basis, drive, inertia, float(error), float(maximum)
    )
