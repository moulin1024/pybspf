"""Independent MMS checks for the BSPF rectangle-minus-hole prototype."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from bspf_models.elliptic.immersed_poisson import ImmersedPoissonPlan
from bspf_models.elliptic.random_wave_mms import RandomWaveMMS


class HoleMMS:
    """Zero outer trace; the pole case has NO analytic whole-box continuation."""

    def __init__(self, kind, hole):
        self.kind, self.hole = kind, hole
        self.wave = (
            RandomWaveMMS.create(kmax=int(kind.removeprefix("random")) * np.pi)
            if kind.startswith("random")
            else None
        )

    def evaluate(self, points):
        x, y = np.asarray(points).T
        bubble = (1 - x * x) * (1 - y * y)
        db = np.column_stack((-2 * x * (1 - y * y), -2 * y * (1 - x * x)))
        lapb = -4 + 2 * (x * x + y * y)
        if self.wave is not None:
            w, dw, fw = self.wave.evaluate(points)
        elif self.kind == "pole":
            d = points - self.hole.center
            r2 = np.sum(d * d, axis=1)
            w, dw, fw = 0.5 * np.log(r2), d / r2[:, None], np.zeros(len(x))
        elif self.kind == "polynomial":
            w, dw, fw = np.ones(len(x)), np.zeros((len(x), 2)), np.zeros(len(x))
        else:
            raise ValueError(self.kind)
        return (
            bubble * w,
            bubble[:, None] * dw + w[:, None] * db,
            bubble * fw - 2 * np.sum(db * dw, axis=1) - w * lapb,
        )

    def forcing(self, points):
        if np.any(self.hole.level(points) <= 1):
            raise AssertionError("The solver requested forcing inside the hole")
        return self.evaluate(points)[2]

    def wall(self, points):
        if np.max(abs(self.hole.level(points) - 1)) > 1e-10:
            raise AssertionError(
                "The solver requested solution values away from the wall"
            )
        return self.evaluate(points)[0]


def relative(error, reference):
    return float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-300))


def validate(solution, mms, grid_count=257, boundary_count=256):
    """All checkpoints differ from the PDE fit and boundary enforcement nodes."""
    plan = solution.plan
    # Offset independent Cartesian samples, no dependence on the training grid.
    axis = -1 + 2 * (np.arange(grid_count) + 0.371) / grid_count
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    fluid = plan.hole.level(points) > 1
    u, grad, laplace = solution.grid(axis, axis)
    exact, dexact, f = mms.evaluate(points[fluid])
    error = u.ravel()[fluid] - exact
    gradient_error = grad.reshape(-1, 2)[fluid] - dexact
    pde_error = -laplace.ravel()[fluid] - f
    boundary, t = plan.arc.sample(boundary_count, offset=0.371)
    bu, _, _ = solution.evaluate(boundary)
    wall_exact = mms.evaluate(boundary)[0]
    normal = plan.hole.normal(t)
    offsets = np.array([1e-2, 1e-3, 1e-4, 1e-6])
    collar = (boundary[None, :, :] + offsets[:, None, None] * normal).reshape(-1, 2)
    cu, cg, cl = solution.evaluate(collar)
    eu, eg, ef = mms.evaluate(collar)
    outer = np.concatenate(
        [np.column_stack((axis, np.full_like(axis, s))) for s in (-1, 1)]
        + [np.column_stack((np.full_like(axis, s), axis)) for s in (-1, 1)]
    )
    ou, og, ol = solution.evaluate(outer)
    _, oeg, oef = mms.evaluate(outer)
    metrics = dict(
        grid_relative_u=relative(error, exact),
        grid_max_u=float(np.max(abs(error))),
        grid_relative_gradient=relative(gradient_error, dexact),
        grid_max_gradient=float(np.max(np.linalg.norm(gradient_error, axis=1))),
        grid_relative_pde=relative(pde_error, f),
        grid_max_pde=float(np.max(abs(pde_error))),
        boundary_max_u=float(np.max(abs(bu - wall_exact))),
        collar_max_u=float(np.max(abs(cu - eu))),
        collar_max_gradient=float(np.max(np.linalg.norm(cg - eg, axis=1))),
        collar_relative_pde=relative(-cl - ef, ef),
        collar_max_pde=float(np.max(abs(cl + ef))),
        outer_max_u=float(np.max(abs(ou))),
        outer_max_gradient=float(np.max(np.linalg.norm(og - oeg, axis=1))),
        outer_max_pde=float(np.max(abs(ol + oef))),
        hole_auxiliary_max_u=float(np.max(abs(u.ravel()[~fluid]))),
        grid_count=grid_count,
        collar_offsets=offsets.tolist(),
        independent_boundary_count=boundary_count,
    )
    plot_exact, plot_error = np.full(len(points), np.nan), np.full(len(points), np.nan)
    plot_exact[fluid], plot_error[fluid] = exact, error
    fields = dict(
        x=axis,
        y=axis,
        numerical=u,
        exact=plot_exact.reshape(xx.shape),
        error=plot_error.reshape(xx.shape),
        physical=fluid.reshape(xx.shape),
        boundary=boundary,
        center=plan.hole.center,
        axes=plan.hole.axes,
    )
    return metrics, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[17, 25, 33, 49, 65])
    parser.add_argument("--cases", nargs="+", default=["random4", "random8", "pole"])
    parser.add_argument("--oversampling", type=float, default=1.5)
    parser.add_argument("--grid-count", type=int, default=257)
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--endpoint-points", type=int)
    parser.add_argument("--chebyshev-modes", type=int)
    parser.add_argument("--out", type=Path, default=Path("build/immersed_poisson"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for nodes in args.nodes:
        plan = ImmersedPoissonPlan(
            nodes=nodes,
            oversampling=args.oversampling,
            rcond=args.rcond,
            endpoint_points=args.endpoint_points,
            chebyshev_modes=args.chebyshev_modes,
        )
        print(
            f"N={nodes}, setup={plan.setup_seconds:.2f}s, rank={plan.rank}/{plan.ndofs}",
            flush=True,
        )
        for kind in args.cases:
            mms = HoleMMS(kind, plan.hole)
            solution = plan.solve(mms.forcing, mms.wall)
            metrics, fields = validate(solution, mms, args.grid_count)
            record = dict(case=kind, **solution.diagnostics, **metrics)
            records.append(record)
            print(json.dumps(record), flush=True)
            np.savez(args.out / f"{kind}_n{nodes}.npz", **fields)
            (args.out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        del solution, plan


if __name__ == "__main__":
    main()
