"""Regularized normal continuation, global seam control, and Poisson coupling."""

import json
import pickle
from pathlib import Path

import jax
import numpy as np
import scipy.linalg as la

from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.normal_continuation import chart
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.regularized_normal import collar_matrices, normal_weights
from bspf_jax.smooth_extension import factors, evaluate_factors, replace_boundary_rule
from embedded_poisson_approximation import interior

jax.config.update("jax_enable_x64", True)


def solve(a, rhs):
    scale = np.maximum(la.norm(a, axis=0), 1e-30)
    c, _, rank, _ = la.lstsq(a / scale, rhs, cond=1e-14)
    return c / scale[:, None], int(rank)


def errors(basis, c, exact):
    u, g, lap = evaluate_factors(basis, c)
    eu, eg, ef = exact
    return dict(
        value=float(la.norm(u - eu) / la.norm(eu)),
        gradient=float(la.norm(g - eg) / la.norm(eg)),
        laplacian=float(la.norm(lap + ef) / la.norm(ef)),
    )


def main():
    out = Path("build/regularized_normal")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain in benchmark_domains():
        with Path(
            f"build/embedded_poisson_smooth_extension_n49/{domain.name}_geometry.pkl"
        ).open("rb") as f:
            geom = pickle.load(f)
        geom = replace_boundary_rule(geom, 12)
        mms = RandomWaveMMS.create(kmax=12 * np.pi)
        print(domain.name, "collar", flush=True)
        cache = out / f"{domain.name}_collar.pkl"
        if cache.exists():
            with cache.open("rb") as handle:
                collar = pickle.load(handle)
        else:
            collar = collar_matrices(geom, retained=6)
            with cache.open("wb") as handle:
                pickle.dump(collar, handle)
        tt = (
            np.arange(domain.period)[:, None] + (np.arange(17)[None, :] + 0.371) / 17
        ).ravel()
        p = chart(domain, tt[:, None], 0.01 * np.array([0.113, 0.287, 0.479])[None, :])[
            0
        ].reshape(-1, 2)
        cb = factors(geom.line, p)
        exact = mms.evaluate(p)
        # Direct normal value-only noise sensitivity before Cartesian projection.
        ri, _ = normal_weights([0.113, 0.287, 0.479], retained=14)
        ip = chart(domain, tt[:, None], 0.01 * ri[None, :])[0]
        data = mms.evaluate(ip.reshape(-1, 2))[0].reshape(len(tt), -1)
        rng = np.random.default_rng(90210)
        noise = rng.normal(size=data.shape) * 1e-12
        phase = p.reshape(len(tt), 3, 2) @ mms.wavevectors.T + mms.phases
        kn = domain.normal(tt) @ mms.wavevectors.T
        normal_truth = [
            -np.einsum("tek,tk,k->te", np.sin(phase), kn, mms.amplitudes),
            -np.einsum("tek,tk,k->te", np.cos(phase), kn**2, mms.amplitudes),
        ]
        for retained in (6, 8, 14):
            _, w = normal_weights([0.113, 0.287, 0.479], retained=retained)
            for amplitude in (0.0, 1.0):
                pred = ((data + amplitude * noise) @ w.T).ravel()
                row = dict(
                    stage="normal_values",
                    domain=domain.name,
                    retained=retained,
                    input_noise=amplitude * 1e-12,
                    value=float(la.norm(pred - exact[0]) / la.norm(exact[0])),
                    gain=float(np.max(np.sum(abs(w), axis=1))),
                )
                for j in (1, 2):
                    _, dw = normal_weights(
                        [0.113, 0.287, 0.479], retained=retained, derivative=j
                    )
                    prediction = (data + amplitude * noise) @ dw.T / 0.01**j
                    row[f"normal_derivative_{j}"] = float(
                        la.norm(prediction - normal_truth[j - 1])
                        / la.norm(normal_truth[j - 1])
                    )
                rows.append(row)
        # Standalone projection: exact INTERIOR values permitted only in this diagnostic.
        root = np.sqrt(geom.weights)
        ext_target = (
            mms.evaluate(collar["inside"].reshape(-1, 2))[0].reshape(
                collar["inside"].shape[:2]
            )
            @ collar["weights"].T
        ).ravel()
        n_ext = len(ext_target)
        a = np.vstack(
            (root[:, None] * geom.value, collar["exterior_basis"] / np.sqrt(n_ext))
        )
        rhs = np.r_[root * mms.evaluate(geom.points)[0], ext_target / np.sqrt(n_ext)][
            :, None
        ]
        input_noise = rng.normal(size=collar["inside"].shape[:2]) * 1e-12
        exterior_noise = (input_noise @ collar["weights"].T).ravel()
        perturbation = np.r_[
            root * rng.normal(size=len(root)) * 1e-12, exterior_noise / np.sqrt(n_ext)
        ]
        rhs = np.column_stack((rhs[:, 0], rhs[:, 0] + perturbation))
        coef, rank = solve(a, rhs)
        for column in (0, 1):
            row = dict(
                stage="global_projection",
                domain=domain.name,
                rank=rank,
                input_noise=column * 1e-12,
                **errors(cb, coef[:, column], exact),
            )
            rows.append(row)
            print(row, flush=True)
        knots = np.arange(domain.period, dtype=float)
        left = factors(geom.line, chart(domain, knots - 1e-10, 0.00479)[0])
        right = factors(geom.line, chart(domain, knots + 1e-10, 0.00479)[0])
        seam = [
            float(np.max(abs(x - y)))
            for x, y in zip(
                evaluate_factors(left, coef[:, 0]), evaluate_factors(right, coef[:, 0])
            )
        ]
        rows.append(
            dict(stage="seam", domain=domain.name, value_gradient_laplacian=seam)
        )
        del a
        # PDE solve: only f in Omega and g on Gamma, all collar samples UNKNOWN.
        check = interior(domain, 101, 103, 0.61)
        basis = factors(geom.line, check)
        truth = mms.evaluate(check)
        rb = np.sqrt(geom.boundary_weights)
        physical = np.vstack(
            (root[:, None] * geom.laplace, rb[:, None] * geom.traces[0])
        )
        rhs0 = np.r_[
            root * mms.evaluate(geom.points)[2], rb * mms.evaluate(geom.boundary)[0]
        ]
        for weight in (0.0, 1e3, 1e5):
            a = np.vstack((physical, weight * collar["relation"] / np.sqrt(n_ext)))
            rhs = np.r_[rhs0, np.zeros(n_ext)][:, None]
            coef, rank = solve(a, rhs)
            row = dict(
                stage="poisson",
                domain=domain.name,
                weight=weight,
                rank=rank,
                physical_data_only=True,
                **errors(basis, coef[:, 0], truth),
                collar_relation_rms=float(
                    la.norm(collar["relation"] @ coef[:, 0]) / np.sqrt(n_ext)
                ),
                boundary_rms=float(
                    la.norm(
                        geom.traces[0] @ coef[:, 0] - mms.evaluate(geom.boundary)[0]
                    )
                    / np.sqrt(len(geom.boundary))
                ),
            )
            rows.append(row)
            print(row, flush=True)
            np.savez(
                out / f"{domain.name}_weight{weight}.npz",
                coefficient=coef[:, 0],
                points=check,
                value=evaluate_factors(basis, coef[:, 0])[0],
                exact=truth[0],
            )
        (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
