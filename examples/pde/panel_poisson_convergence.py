"""Compare global FFT, knot-aligned panels, and endpoint-driven adaptation."""

import json
from pathlib import Path

import numpy as np

from boundary_fft_core import BoundaryFFTPlan, harmonic
from boundary_fft_random_mms import boundary_values
from bspf_jax.convex_poisson import ArcLengthBoundary
from bspf_jax.embedded_poisson import benchmark_domains
from bspf_jax.panel_poisson import PanelPoissonPlan, adaptive_solve


def particular(points):
    return -np.sum(points**2, axis=1) / 4  # -Delta v=1


def exact(points):
    return particular(points) + harmonic(points)


def main():
    out = Path("build/panel_poisson")
    out.mkdir(parents=True, exist_ok=True)
    domain = benchmark_domains()[0]
    arc = ArcLengthBoundary(domain)
    boundary, t = arc.sample(1024, offset=0.371)
    near = np.mod(
        (
            np.arange(domain.period)[:, None]
            + np.r_[0, 10.0 ** -np.arange(1, 8), -(10.0 ** -np.arange(1, 8))]
        ),
        domain.period,
    ).ravel()
    t_all = np.r_[t, near]
    points_all = domain.curve(t_all)
    expected = exact(points_all)
    scale = np.max(abs(expected))
    theta = 2 * np.pi * np.arange(64) / 64
    interior = np.vstack(
        [
            r * np.column_stack((np.cos(theta), np.sin(theta)))
            for r in (0.0, 0.15, 0.3, 0.45)
        ]
    )
    exact_interior = exact(interior)
    rows = []

    def record(solution, method, level=None):
        numerical = solution.boundary(t_all)
        values = solution.interior(interior)
        row = dict(
            method=method,
            level=level,
            order=solution.plan.order,
            panels=solution.plan.panels,
            unknowns=solution.plan.count,
            boundary_relative=float(
                np.linalg.norm(numerical[:1024] - expected[:1024])
                / np.linalg.norm(expected[:1024])
            ),
            boundary_max_scaled=float(np.max(abs(numerical - expected)) / scale),
            knot_near_max_scaled=float(
                np.max(abs(numerical[1024:] - expected[1024:])) / scale
            ),
            interior_relative=float(
                np.linalg.norm(values - exact_interior) / np.linalg.norm(exact_interior)
            ),
            setup_seconds=solution.plan.setup_seconds,
            solve_seconds=solution.solve_seconds,
            training_residual=solution.training_residual,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    for order in (4, 6, 8, 12, 16, 24, 32):
        plan = PanelPoissonPlan(domain, order)
        record(plan.solve(exact, particular), "panel_p")
    for refinement in (2, 4, 8):
        breaks = np.arange(domain.period * refinement + 1) / refinement
        plan = PanelPoissonPlan(domain, 12, breaks)
        record(plan.solve(exact, particular), "panel_uniform_h", refinement)

    def progress(solution, entry):
        record(solution, "panel_adaptive", entry["level"])
        print(
            json.dumps(
                dict(
                    adaptive_indicator=entry["max_indicator"],
                    converged=entry["converged"],
                )
            ),
            flush=True,
        )

    solution, history = adaptive_solve(
        domain,
        exact,
        particular=particular,
        order=12,
        tolerance=1e-12,
        max_refinements=8,
        callback=progress,
    )
    check_parameters = np.r_[near, t[::16]]
    coarse = solution.boundary(check_parameters)
    fine = solution.boundary(check_parameters, 96)
    quadrature_change = float(np.max(abs(fine - coarse)) / scale)
    print(
        json.dumps(dict(quadrature_relative_max_change=quadrature_change)), flush=True
    )
    assert quadrature_change < 2e-12
    assert history[-1]["converged"]
    np.savez(
        out / "adaptive_solution.npz",
        breaks=solution.plan.breaks,
        parameters=solution.plan.parameters,
        density=solution.density,
        constant=solution.constant,
        controls=domain.controls,
    )
    np.savez(
        out / "validation.npz",
        parameters=t_all,
        points=points_all,
        exact=expected,
        numerical=solution.boundary(t_all),
    )

    # Same geometry, MMS and independent validation points as the panel solver.
    angles = []
    for parameter in t_all:
        span = min(int(parameter), domain.period - 1)
        angles.append(
            2
            * np.pi
            / arc.length
            * (arc.offsets[span] + arc.integral(span, parameter - span))
        )
    for count in (64, 128, 256, 512, 1024):
        plan = BoundaryFFTPlan(domain, count)
        density, constant, diagnostics = plan.solve(
            harmonic(plan.points), tolerance=2e-14
        )
        numerical = boundary_values(
            plan, density[:, None], np.array([constant]), points_all, np.array(angles)
        )[:, 0] + particular(points_all)
        values = plan.interior_evaluate(density, constant, interior) + particular(
            interior
        )
        row = dict(
            method="global_fft",
            unknowns=count,
            boundary_relative=float(
                np.linalg.norm(numerical[:1024] - expected[:1024])
                / np.linalg.norm(expected[:1024])
            ),
            boundary_max_scaled=float(np.max(abs(numerical - expected)) / scale),
            knot_near_max_scaled=float(
                np.max(abs(numerical[1024:] - expected[1024:])) / scale
            ),
            interior_relative=float(
                np.linalg.norm(values - exact_interior) / np.linalg.norm(exact_interior)
            ),
            setup_seconds=plan.setup_seconds,
            **diagnostics,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(
        geometry="unchanged convex periodic cubic B-spline, 12 original spans",
        source="f=1; analytic particular -(x^2+y^2)/4; nonzero harmonic correction",
        independent_arclength_points=1024,
        knot_and_near_knot_probes=len(near),
        quadrature_relative_max_change=quadrature_change,
        rows=rows,
        adaptation=history,
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
