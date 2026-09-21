"""Fixed 73x33 BSPF spaces versus an independently converged rational Stokes solve."""

import gc
import json
from pathlib import Path
import jax
import numpy as np
import scipy.linalg as la
from bspf_models.fluids.immersed_flow import ImmersedFlowPlan
from bspf_models.fluids.immersed_flow import channel_lift
from lightning_stokes_reference import LightningStokes


def main():
    jax.config.update("jax_enable_x64", True)
    out = Path("build/immersed_flow/stokes_comparison")
    out.mkdir(parents=True, exist_ok=True)
    ref = LightningStokes(96, 32, 48, 800)
    print("REFERENCE", ref.verify(), flush=True)
    x, y = np.linspace(-1, 5, 401), np.linspace(-1, 1, 161)
    xx, yy = np.meshgrid(x, y)
    zz = xx + 1j * yy
    fluid = ((xx - 0.19) / 0.31) ** 2 + ((yy + 0.13) / 0.23) ** 2 > 1
    fluid[[0, 0, -1, -1], [0, -1, 0, -1]] = False
    rf = np.full((6,) + zz.shape, np.nan)
    rf[:, fluid] = ref.evaluate(zz[fluid])
    np.savez(out / "reference.npz", x=x, y=y, fields=rf)
    records = []
    for method in ("svd", "factor"):
        p = ImmersedFlowPlan(
            nx=73, ny=33, wall_method=method, wall_rcond=1e-5, buffer_strength=0
        )
        rv = ref.evaluate(p.points[:, 0] + 1j * p.points[:, 1])
        targets = (rv[0], rv[1], rv[4], rv[5] - rv[3], rv[5])
        differences = [v - lift for v, lift in zip(targets, p.lift_fields[1:])]
        rhs = sum(
            o.T @ (p.weights * d * w)
            for o, d, w in zip(p.operators_fluid, differences, (1, 1, 2, 1, 1))
        )
        gram = p.mass + p.stiffness
        best = la.solve(gram, rhs, assume_a="pos")
        states = [(method, p.stokes_state, best, p.constraint_rank)]
        if method == "svd":
            b, _ = p.arc.sample(p.boundary_count)
            op = p.operators(b)
            c = np.vstack(op[1:3]) * p.scale
            target = -np.concatenate(channel_lift(b)[1:3])
            u, s, vh = la.svd(c, full_matrices=False)
            rank = int(np.sum(s > 1e-10 * s[0]))
            v = vh[p.constraint_rank : rank]
            constraint = v @ (p.transform / p.scale[:, None])
            target = (u[:, p.constraint_rank : rank].T @ target) / s[
                p.constraint_rank : rank
            ] - v @ (p.lift_coefficients / p.scale)
            q, r = la.qr(constraint.T, mode="full")
            a0 = q[:, : len(v)] @ la.solve_triangular(r[: len(v)].T, target, lower=True)
            z = q[:, len(v) :]
            strict = a0 + z @ la.solve(
                z.T @ p.linear @ z,
                -z.T @ (p.linear @ a0 + p.linear_lift),
                assume_a="pos",
            )
            strictbest = a0 + z @ la.solve(
                z.T @ gram @ z, z.T @ (rhs - gram @ a0), assume_a="pos"
            )
            states.append(("svd_strict", strict, strictbest, rank))
        for label, state, approximation, rank in states:
            row = dict(
                method=label,
                background_grid=[p.nx, p.ny],
                rank=rank,
                dofs=p.dofs - (rank - p.constraint_rank),
                setup_seconds=p.setup_seconds,
            )
            for name, a in [("pde", state), ("best_h1_velocity", approximation)]:
                f = p.grid(a, x, y)
                du, dv, dw = f["u"] - rf[0], f["v"] - rf[1], f["vorticity"] - rf[3]
                values = [
                    o @ a + lift
                    for o, lift in zip(p.operators_fluid, p.lift_fields[1:])
                ]
                errors = [v - t for v, t in zip(values, targets)]

                def norm(es, weights=p.weights):
                    return np.sqrt(
                        sum(weights @ (e * e * w) for e, w in zip(es, (1, 1, 2, 1, 1)))
                    )

                b, _ = p.arc.sample(512, offset=0.371)
                bf = p.evaluate(a, b)
                row[name] = dict(
                    velocity_relative_l2=float(
                        np.sqrt(
                            np.nansum(du * du + dv * dv)
                            / np.nansum(rf[0] ** 2 + rf[1] ** 2)
                        )
                    ),
                    velocity_max=float(np.nanmax(np.hypot(du, dv))),
                    vorticity_relative_l2=float(
                        np.sqrt(np.nansum(dw * dw) / np.nansum(rf[3] ** 2))
                    ),
                    vorticity_max=float(np.nanmax(abs(dw))),
                    velocity_relative_h1=float(norm(errors) / norm(targets)),
                    wall_max=float(np.max(np.hypot(bf[1], bf[2]))),
                )
                np.savez(out / f"{label}_{name}.npz", x=x, y=y, **f)
            records.append(row)
            print(json.dumps(row), flush=True)
        del p, states
        gc.collect()
    (out / "comparison.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
