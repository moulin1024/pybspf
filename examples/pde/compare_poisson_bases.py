"""Matched-space comparison: BSPF, real Fourier, and degree-13 B-splines.

All cases share geometry, quadrature, arclength trace, box-H2 scaling, TSVD
cutoffs and independent validation points. H2 fitting uses exact interior jets
ONLY in a separately labelled approximation diagnostic, never in the PDE solve.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import scipy.linalg as la
from scipy.interpolate import BSpline
from scipy.special import roots_legendre

from bspf_jax.convex_poisson import ArcLengthBoundary, box_h2_root, trace_transform
from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.random_wave_mms import RandomWaveMMS
from bspf_jax.stream_navier_stokes import _stream_line, stream_evaluate_line
from embedded_poisson_approximation import interior


class TrialBasis:
    def __init__(self, family, n, half_width=1.2, degree=13):
        self.family, self.n = family, n
        self.half_width, self.degree = half_width, degree
        length = 2 * half_width
        if family == "bspf":
            self.line = _stream_line(
                np.linspace(-half_width, half_width, n),
                clamped=False,
                dirichlet=False,
                endpoint_points=12,
                chebyshev_modes=12,
            )
            self.lam = np.asarray(self.line.lam)
            self.bending = np.asarray(self.line.bending)
        elif family == "fourier":
            if n % 2 != 1:
                raise ValueError("Use an odd dimension for paired real Fourier modes")
            self.frequency = 2 * np.pi * np.arange(1, (n + 1) // 2) / length
            self.lam = np.r_[0, np.repeat(self.frequency**2, 2)]
            self.bending = np.diag(self.lam**2)
        elif family == "bspline":
            if n <= degree:
                raise ValueError("Require more B-splines than polynomial degree")
            breaks = np.linspace(-half_width, half_width, n - degree + 1)
            knots = np.r_[
                np.repeat(breaks[0], degree + 1),
                breaks[1:-1],
                np.repeat(breaks[-1], degree + 1),
            ]
            self.spline = BSpline(knots, np.eye(n), degree, extrapolate=False)
            q, w = roots_legendre(max(24, degree + 2))
            points = np.concatenate(
                [(a + b) / 2 + (b - a) / 2 * q for a, b in zip(breaks[:-1], breaks[1:])]
            )
            weights = np.concatenate(
                [(b - a) / 2 * w for a, b in zip(breaks[:-1], breaks[1:])]
            )
            b, g, h = [self.spline(points, nu=d) for d in range(3)]
            _, r = la.qr(np.sqrt(weights[:, None]) * b, mode="economic")
            t = la.solve_triangular(r, np.eye(n))
            g = g @ t
            k = g.T @ (weights[:, None] * g)
            self.lam, v = la.eigh((k + k.T) / 2)
            self.transform = t @ v
            h = h @ self.transform
            self.bending = h.T @ (weights[:, None] * h)
        else:
            raise ValueError("Unknown basis family")

    def evaluate(self, points):
        x = np.asarray(points)
        if self.family == "bspf":
            return stream_evaluate_line(self.line, x)
        if self.family == "bspline":
            return tuple(self.spline(x, nu=d) @ self.transform for d in range(3))
        phase = (x[:, None] + self.half_width) * self.frequency
        cosine, sine = np.cos(phase), np.sin(phase)
        factor = np.sqrt(1 / self.half_width)
        result = [np.empty((len(x), self.n)) for _ in range(3)]
        result[0][:, 0] = 1 / np.sqrt(2 * self.half_width)
        result[1][:, 0] = result[2][:, 0] = 0
        result[0][:, 1::2], result[0][:, 2::2] = factor * cosine, factor * sine
        result[1][:, 1::2] = -factor * sine * self.frequency
        result[1][:, 2::2] = factor * cosine * self.frequency
        result[2][:, 1::2] = -factor * cosine * self.frequency**2
        result[2][:, 2::2] = -factor * sine * self.frequency**2
        return tuple(result)

    def factors(self, points):
        output = []
        for axis in range(2):
            unique, index = np.unique(points[:, axis], return_inverse=True)
            output.append(tuple(v[index] for v in self.evaluate(unique)))
        return output


def exact_jets(mms, points):
    if isinstance(mms, ProfileMMS):
        return mms.jets(points)
    u, g, _ = mms.evaluate(points)
    cosine = np.cos(points @ mms.wavevectors.T + mms.phases) * mms.amplitudes
    k = mms.wavevectors
    return np.column_stack(
        (
            u,
            g,
            -cosine @ k[:, 0] ** 2,
            -np.sqrt(2) * cosine @ (k[:, 0] * k[:, 1]),
            -cosine @ k[:, 1] ** 2,
        )
    )


class ProfileMMS:
    """Analytic test functions not defined as a finite Fourier sum."""

    def __init__(self, name):
        self.name = name

    def jets(self, points):
        x, y = np.asarray(points).T
        if self.name == "polynomial":
            z = 1 + 0.3 * x - 0.2 * y
            u = z**8 + 0.2 * x * y
            ux, uy = 2.4 * z**7 + 0.2 * y, -1.6 * z**7 + 0.2 * x
            xx, xy, yy = 5.04 * z**6, -3.36 * z**6 + 0.2, 2.24 * z**6
        else:
            if self.name == "gaussian":
                q = np.array([[100.0, 25.0], [25.0, 60.0]])
                delta = points - [0.27, -0.13]
            elif self.name == "rational":
                q = np.array([[9.0, 2.0], [2.0, 16.0]])
                delta = points - [0.13, -0.09]
            else:
                raise ValueError(self.name)
            v = delta @ q
            radius = np.sum(delta * v, axis=1)
            if self.name == "gaussian":
                u = np.exp(-radius / 2)
                ux, uy = (-v * u[:, None]).T
                xx = (v[:, 0] ** 2 - q[0, 0]) * u
                xy = (v[:, 0] * v[:, 1] - q[0, 1]) * u
                yy = (v[:, 1] ** 2 - q[1, 1]) * u
            else:
                u = 1 / (1 + radius)
                ux, uy = (-2 * v * u[:, None] ** 2).T
                xx = 8 * v[:, 0] ** 2 * u**3 - 2 * q[0, 0] * u**2
                xy = 8 * v[:, 0] * v[:, 1] * u**3 - 2 * q[0, 1] * u**2
                yy = 8 * v[:, 1] ** 2 * u**3 - 2 * q[1, 1] * u**2
        return np.column_stack((u, ux, uy, xx, np.sqrt(2) * xy, yy))

    def evaluate(self, points):
        jets = self.jets(points)
        return jets[:, 0], jets[:, 1:3], -(jets[:, 3] + jets[:, 5])


def field_jets(basis, coefficients):
    (x, dx, xx), (y, dy, yy) = basis
    c = np.asarray(coefficients).reshape(x.shape[1], y.shape[1])
    v, g, h = x @ c, dx @ c, xx @ c
    return np.column_stack(
        (
            np.sum(v * y, axis=1),
            np.sum(g * y, axis=1),
            np.sum(v * dy, axis=1),
            np.sum(h * y, axis=1),
            np.sqrt(2) * np.sum(g * dy, axis=1),
            np.sum(v * yy, axis=1),
        )
    )


def pair(x, y):
    return (x[:, :, None] * y[:, None, :]).reshape(len(x), -1)


def whiten(matrix, root):
    # Right triangular BLAS solve, keeping the large matrix in Fortran storage
    # so both whitening and SVD can overwrite it without an extra full copy.
    return la.blas.dtrsm(1.0, root, matrix, side=1, lower=0, overwrite_b=1)


def fit(matrix, rhs, root, cutoffs):
    start = perf_counter()
    matrix = whiten(matrix, root)
    whiten_seconds = perf_counter() - start
    start = perf_counter()
    u, s, vh = la.svd(matrix, full_matrices=False, overwrite_a=True)
    svd_seconds = perf_counter() - start
    del matrix
    projected = u.T @ rhs
    results = []
    for cutoff in cutoffs:
        keep = s > cutoff * s[0]

        def solve_one(data):
            a = vh[keep].T @ ((u.T @ data)[keep] / s[keep])
            return la.solve_triangular(root, a)

        # A repeated single RHS, excluding data construction and field evaluation.
        solve_one(rhs[:, 0])
        timings = []
        for _ in range(5):
            t = perf_counter()
            solve_one(rhs[:, 0])
            timings.append(perf_counter() - t)
        coefficient = la.solve_triangular(
            root, vh[keep].T @ (projected[keep] / s[keep, None])
        )
        results.append(
            dict(
                cutoff=cutoff,
                rank=int(keep.sum()),
                coefficient=coefficient,
                repeated_rhs_seconds=float(np.median(timings)),
            )
        )
    return (
        results,
        dict(
            whiten_seconds=whiten_seconds,
            svd_seconds=svd_seconds,
            smallest_relative_singular=float(s[-1] / s[0]),
        ),
        s,
    )


def error_metrics(predicted, exact):
    norm = np.linalg.norm
    return dict(
        value=float(norm(predicted[:, 0] - exact[:, 0]) / norm(exact[:, 0])),
        gradient=float(norm(predicted[:, 1:3] - exact[:, 1:3]) / norm(exact[:, 1:3])),
        h2=float(norm(predicted - exact) / norm(exact)),
        laplacian=float(
            norm(predicted[:, 3] + predicted[:, 5] - exact[:, 3] - exact[:, 5])
            / norm(exact[:, 3] + exact[:, 5])
        ),
        value_linf=float(np.max(abs(predicted[:, 0] - exact[:, 0]))),
    )


def run_case(args, n):
    out = args.out / f"{args.family}_n{n}"
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(
        family=args.family,
        n=n,
        degree=args.degree,
        half_width=1.2,
        volume_order=args.volume_order,
        boundary_count=args.boundary_count,
        bands=args.bands,
        profiles=args.profiles,
        cutoffs=args.cutoffs,
        seed=20260918,
    )
    if args.resume and (out / "results.json").exists():
        saved = json.loads((out / "results.json").read_text())
        if saved["settings"] == settings and saved.get("complete"):
            print("REUSE", out, flush=True)
            return
    domain = benchmark_domains()[0]
    points, weights, _ = domain.volume_rule(args.volume_order)
    arc = ArcLengthBoundary(domain)
    boundary, _ = arc.sample(args.boundary_count)
    check = interior(domain, 127, 131, 0.613)
    edge, _ = arc.sample(2 * args.boundary_count, offset=0.371)
    names = [f"{band}pi" for band in args.bands] + args.profiles
    bands = args.bands + [None] * len(args.profiles)
    cases = [RandomWaveMMS.create(kmax=band * np.pi) for band in args.bands]
    cases += [ProfileMMS(name) for name in args.profiles]
    truth = [exact_jets(m, points) for m in cases]
    check_truth = [exact_jets(m, check) for m in cases]
    boundary_truth = [m.evaluate(boundary)[0] for m in cases]
    edge_truth = [m.evaluate(edge)[0] for m in cases]
    t = perf_counter()
    basis = TrialBasis(args.family, n, degree=args.degree)
    line_seconds = perf_counter() - t
    t = perf_counter()
    volume = basis.factors(points)
    trace = basis.factors(boundary)
    factor_seconds = perf_counter() - t
    t = perf_counter()
    root = box_h2_root(basis)
    root_seconds = perf_counter() - t
    # Validation factors are excluded from solver construction timing.
    check_basis = basis.factors(check)
    edge_basis = basis.factors(edge)
    rows = []
    (x, dx, xx), (y, dy, yy) = volume
    print(
        "BASIS",
        args.family,
        n,
        "line",
        line_seconds,
        "factors",
        factor_seconds,
        flush=True,
    )
    for task in ("pde", "h2_fit"):
        print("ASSEMBLE", args.family, n, task, flush=True)
        start = perf_counter()
        if task == "pde":
            a = np.empty((len(points) + args.boundary_count + 2, n * n), order="F")
            a[: len(points)] = -pair(xx, y) - pair(x, yy)
            a[: len(points)] *= np.sqrt(weights[:, None])
            a[len(points) :] = trace_transform(
                pair(trace[0][0], trace[1][0]), arc.length
            )
            rhs = np.column_stack(
                [
                    np.r_[
                        np.sqrt(weights) * (-(v[:, 3] + v[:, 5])),
                        trace_transform(g, arc.length),
                    ]
                    for v, g in zip(truth, boundary_truth)
                ]
            )
        else:
            a = np.empty((6 * len(points), n * n), order="F")
            for block, (left, right) in enumerate(
                ((x, y), (dx, y), (x, dy), (xx, y), (np.sqrt(2) * dx, dy), (x, yy))
            ):
                a[block * len(points) : (block + 1) * len(points)] = pair(left, right)
                a[block * len(points) : (block + 1) * len(points)] *= np.sqrt(
                    weights[:, None]
                )
            rhs = np.column_stack(
                [(np.sqrt(weights[:, None]) * v).T.ravel() for v in truth]
            )
        assemble_seconds = perf_counter() - start
        shape = a.shape
        matrix_bytes = a.nbytes
        print("SVD", args.family, n, task, shape, flush=True)
        fits, timing, singular = fit(a, rhs, root, args.cutoffs)
        del a
        np.save(out / f"{task}_singular.npy", singular)
        for fit_result in fits:
            coefficients = fit_result.pop("coefficient")
            for col, (name, band) in enumerate(zip(names, bands)):
                c = coefficients[:, col]
                train = field_jets(volume, c)
                predicted = field_jets(check_basis, c)
                berror = field_jets(edge_basis, c)[:, 0] - edge_truth[col]
                if task == "h2_fit":
                    train_residual = np.linalg.norm(
                        np.sqrt(weights[:, None]) * (train - truth[col])
                    ) / np.linalg.norm(np.sqrt(weights[:, None]) * truth[col])
                else:
                    verr = (train[:, 3] + train[:, 5]) - (
                        truth[col][:, 3] + truth[col][:, 5]
                    )
                    berr = field_jets(trace, c)[:, 0] - boundary_truth[col]
                    train_residual = np.sqrt(
                        np.sum(weights * verr**2)
                        + np.linalg.norm(trace_transform(berr, arc.length)) ** 2
                    ) / np.linalg.norm(rhs[:, col])
                row = dict(
                    task=task,
                    case=name,
                    band=band,
                    **fit_result,
                    **timing,
                    line_seconds=line_seconds,
                    factor_seconds=factor_seconds,
                    h2_scaling_seconds=root_seconds,
                    assemble_seconds=assemble_seconds,
                    setup_seconds=line_seconds
                    + factor_seconds
                    + root_seconds
                    + assemble_seconds
                    + timing["whiten_seconds"]
                    + timing["svd_seconds"],
                    matrix_shape=shape,
                    matrix_bytes=matrix_bytes,
                    coefficient_norm=float(np.linalg.norm(c)),
                    box_h2_norm=float(np.linalg.norm(root @ c)),
                    relative_training_residual=float(train_residual),
                    boundary_h32=float(
                        np.linalg.norm(trace_transform(berror, arc.length))
                    ),
                    boundary_linf=float(np.max(abs(berror))),
                    **error_metrics(predicted, check_truth[col]),
                )
                rows.append(row)
                print("RESULT", args.family, n, json.dumps(row), flush=True)
                if fit_result["cutoff"] == 1e-13:
                    np.savez(
                        out / f"{task}_{name}.npz",
                        coefficient=c,
                        points=check,
                        value=predicted[:, 0],
                        exact=check_truth[col][:, 0],
                    )
        (out / "results.json").write_text(
            json.dumps(
                dict(
                    settings=settings,
                    rows=rows,
                    volume_samples=len(points),
                    validation_samples=len(check),
                    validation_boundary_count=len(edge),
                    complete=task == "h2_fit",
                ),
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=["bspf", "fourier", "bspline"], required=True
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=[17, 33, 65])
    parser.add_argument("--degree", type=int, default=13)
    parser.add_argument("--volume-order", type=int, default=16)
    parser.add_argument("--boundary-count", type=int, default=1024)
    parser.add_argument("--bands", type=int, nargs="+", default=[4, 12])
    parser.add_argument(
        "--profiles",
        nargs="*",
        choices=["polynomial", "gaussian", "rational"],
        default=["polynomial", "gaussian", "rational"],
    )
    parser.add_argument(
        "--cutoffs", type=float, nargs="+", default=[1e-11, 1e-13, 1e-14]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("build/poisson_basis_comparison")
    )
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    for n in args.nodes:
        run_case(args, n)


if __name__ == "__main__":
    main()
