"""Stage-by-stage diagnostics of the zero-refinement pressure direct solver."""

import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from bspf_jax import (
    plan_pressure_poisson2d,
    pressure_gradient,
    pressure_schur,
    pressure_remove_mean,
)
from bspf_jax.pressure import _differentiate, _tensor_solve, _lift, _null_field, _wall

jax.config.update("jax_enable_x64", True)


def norm(x):
    return float(jnp.linalg.norm(x))


def linf(x):
    return float(jnp.max(abs(x)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument(
        "--out", type=Path, default=Path("build/pressure_accuracy_diagnosis")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for n in args.sizes:
        start = time.perf_counter()
        print(f"N={n} building plan", flush=True)
        grid = np.linspace(0, 1, n)
        plan = plan_pressure_poisson2d(
            grid,
            grid,
            endpoint_method="chebyshev",
            baseline_points=16,
            chebyshev_modes=12,
        )
        line = plan.x
        differentiate = jax.jit(_differentiate, static_argnums=2)
        D = differentiate(line, jnp.eye(n), 0)
        Dy = differentiate(plan.y, jnp.eye(n), 0)
        H = D[:, 1:-1] @ D[1:-1, :]
        A = H[1:-1, 1:-1] - line.coupling @ line.hei
        V, Vi, L = map(
            np.asarray, (line.vectors, line.inverse_vectors, line.eigenvalues)
        )
        row = dict(N=n, line={}, cases={})
        r = row["line"]
        r["Dx_Dy_relative_difference"] = norm(D - Dy) / norm(D)
        r["projector_xy_relative_difference"] = norm(
            line.projector - plan.y.projector
        ) / norm(line.projector)
        r.update(
            cond_V=float(np.linalg.cond(V)),
            cond_endpoint=float(
                np.linalg.cond(
                    np.asarray(H[jnp.ix_(jnp.array([0, n - 1]), jnp.array([0, n - 1]))])
                )
            ),
            eigen_relative_residual=norm(
                A @ line.vectors - line.vectors * line.eigenvalues
            )
            / norm(A),
            low_norm=norm(line.low),
            projector_norm=norm(line.projector),
            D_norm=norm(D),
            factor_product_norm_ratio=norm(line.low) * norm(line.projector) / norm(D),
            eigenvalues_first8=[[float(z.real), float(z.imag)] for z in L[:8]],
            wall_R_cond=float(np.linalg.cond(np.asarray(plan.wall_r))),
        )
        vectors = {
            "constant": jnp.ones(n),
            "linear": line.x,
            "smooth": jnp.exp(line.x),
            "random": jnp.asarray(np.random.default_rng(17).normal(size=n)),
        }
        for name, v in vectors.items():
            applied = _differentiate(line, v, 0)
            dense = D @ v
            r[name] = dict(
                apply_norm=norm(applied),
                dense_norm=norm(dense),
                difference_linf=linf(applied - dense),
                difference_relative=norm(applied - dense) / max(norm(dense), 1e-300),
            )
            if name != "random":
                expected = {
                    "constant": jnp.zeros(n),
                    "linear": jnp.ones(n),
                    "smooth": jnp.exp(line.x),
                }[name]
                r[name]["apply_error_linf"] = linf(applied - expected)
                r[name]["dense_error_linf"] = linf(dense - expected)
        r["null_gradient_linf"] = linf(D[1:-1] @ line.null_basis)
        r["null_gradient_factored_linf"] = linf(
            _differentiate(line, line.null_basis, 0)[1:-1]
        )
        np.savez_compressed(
            args.out / f"line_{n}.npz",
            D=np.asarray(D),
            P=np.asarray(line.projector),
            low=np.asarray(line.low),
            A=np.asarray(A),
            V=V,
            Vi=Vi,
            lam=L,
            Z=np.asarray(line.null_basis),
            C=np.asarray(line.coupling),
            hei=np.asarray(line.hei),
            Ei=np.asarray(line.endpoint_inverse),
        )
        print("LINE " + json.dumps(r), flush=True)

        def dense_schur(p):
            return D @ (plan.mask * (D @ p)) + (plan.mask * (p @ Dy.T)) @ Dy.T

        schur_fn = jax.jit(pressure_schur)

        def schur(p):
            return schur_fn(plan, p)

        dense_schur = jax.jit(dense_schur)
        tensor_fn = jax.jit(_tensor_solve)

        def tensor(b):
            return tensor_fn(plan, b)

        lift_fn = jax.jit(_lift)

        def lift(p):
            return lift_fn(plan, p)

        grad_fn = jax.jit(pressure_gradient)

        def grad(p):
            return grad_fn(plan, p)

        xx, yy = jnp.meshgrid(line.x, line.x, indexing="ij")
        fields = {
            "smooth": jnp.exp(xx + 0.5 * yy)
            + jnp.sin(3 * jnp.pi * xx) * jnp.cos(2 * jnp.pi * yy),
            "random": jnp.asarray(np.random.default_rng(71).normal(size=(n, n))),
        }
        for name, p in fields.items():
            print(f"N={n} case={name}", flush=True)
            expected = pressure_remove_mean(plan, p)
            b = schur(p)
            target = grad(p)
            entry = dict(
                rhs_norm=norm(b),
                threshold=1e-9 + 1e-10 * norm(b),
                rhs_dense_difference_linf=linf(b - dense_schur(p)),
                rhs_dense_difference_relative=norm(b - dense_schur(p)) / norm(b),
                stages={},
            )
            p0 = tensor(b)
            wallres = _wall(plan, target - grad(p0)).ravel()
            coefficients = (
                jax.scipy.linalg.solve_triangular(plan.wall_r, plan.wall_q.T @ wallres)
                / plan.wall_scales
            )
            completion = sum(coefficients[j] * _null_field(plan, j) for j in range(7))
            p1 = p0 + completion
            p2 = pressure_remove_mean(plan, p1)
            entry["completion_coefficients"] = np.asarray(coefficients).tolist()
            entry["completion_schur_linf"] = linf(schur(completion))
            entry["mean_shift"] = float(
                jnp.sum(plan.weights * p1) / jnp.sum(plan.weights)
            )
            for label, field in [
                ("tensor", p0),
                ("completed", p1),
                ("mean_removed", p2),
            ]:
                residual = schur(field) - b
                entry["stages"][label] = dict(
                    schur_linf=linf(residual),
                    schur_l2=norm(residual),
                    lifted_relative_residual=norm(residual + lift(field)) / norm(b),
                    dense_lifted_relative_residual=norm(
                        dense_schur(field) + lift(field) - b
                    )
                    / norm(b),
                    interior_gradient_relative_error=norm(
                        (grad(field) - target)[1:-1, 1:-1]
                    )
                    / norm(target[1:-1, 1:-1]),
                    pressure_relative_error=norm(
                        pressure_remove_mean(plan, field) - expected
                    )
                    / norm(expected),
                )
            # Manufactured lifted equation removes completion/gauge ambiguity.
            lifted_rhs = dense_schur(p) + lift(p)
            recovered = tensor(lifted_rhs)
            entry["dense_manufactured_lifted"] = dict(
                relative_solution_error=norm(recovered - p) / norm(p),
                relative_residual=norm(
                    dense_schur(recovered) + lift(recovered) - lifted_rhs
                )
                / norm(lifted_rhs),
            )
            entry["constant_schur_linf"] = linf(schur(jnp.ones_like(p)))
            row["cases"][name] = entry
            print("CASE " + json.dumps(entry), flush=True)
        row["seconds"] = time.perf_counter() - start
        records.append(row)
        (args.out / "results.json").write_text(json.dumps(records, indent=2))
        print(f"N={n} done in {row['seconds']:.1f}s", flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
