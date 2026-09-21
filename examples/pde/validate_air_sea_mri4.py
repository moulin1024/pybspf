"""Fourth-order time-refinement experiment on the actual coupled spatial ODE.

Same n=33 spatial model and physics at every refinement, fixed H/h_fast=10.
Compare MRI-GARK4 with the retained first-order lagged scheme. Changes to an
output grouping window alone are intentionally NOT called time refinement.
"""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import jax
import numpy as np

from bspf_jax.air_sea import plan_air_sea, plan_air_sea_stepper
from validate_air_sea_double_gyre import integrate


def modal_distance(plan, a, b):
    """Physical RMS differences, evaluated in orthonormal coefficient spaces.

    Avoid subtracting 290 K reconstructed temperatures when time errors approach
    float64 roundoff. The harmonic mean zonal wind contributes to air kinetic mass.
    """
    d = [np.asarray(x) - np.asarray(y) for x, y in zip(a, b)]
    result = dict(
        ocean_velocity=float(
            np.sqrt(np.sum(np.asarray(plan.flow.denominator) * d[0] ** 2))
        ),
        air_velocity=float(
            np.sqrt(
                np.sum(np.asarray(plan.air_flow.denominator) * d[1] ** 2) + d[5] ** 2
            )
        ),
        sst=float(np.linalg.norm(d[2])),
        air_temperature=float(np.linalg.norm(d[3])),
        humidity=float(np.linalg.norm(d[4])),
    )
    scales = dict(
        ocean_velocity=0.1,
        air_velocity=8.0,
        sst=1.0,
        air_temperature=1.0,
        humidity=0.01,
    )
    result["scaled_combined"] = float(
        np.sqrt(sum((result[k] / scales[k]) ** 2 for k in scales))
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("build/air_sea_mri4_validation")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_enable_x64", True)
    plan = plan_air_sea(n=33)
    plan = replace(
        plan,
        config=replace(
            plan.config, ocean_depth=50.0, mixed_layer_depth=5.0, atmosphere_depth=300.0
        ),
    )
    sizes = (2400.0, 1200.0, 600.0, 300.0, 150.0)
    duration = 4800.0
    states, runs = {}, {}
    for method in ("mri-gark4", "lagged"):
        states[method], runs[method] = {}, {}
        for h in sizes:
            stepper = plan_air_sea_stepper(
                dt_air=h / 10, dt_ocean=h, window=h, method=method
            )
            state, budget = integrate(plan, stepper, duration)
            states[method][h] = state
            runs[method][str(int(h))] = dict(
                stepper=stepper._asdict(),
                effective_dt_air=stepper.effective_dt_air,
                **budget,
            )
            print(
                f"{method} H={h:g}, h={stepper.effective_dt_air:g}: {budget}",
                flush=True,
            )
    differences, orders, reference_errors = {}, {}, {}
    reference = states["mri-gark4"][sizes[-1]]
    for method in states:
        differences[method] = [
            modal_distance(plan, states[method][a], states[method][b])
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        values = [d["scaled_combined"] for d in differences[method]]
        orders[method] = np.log2(np.array(values[:-1]) / values[1:]).tolist()
        reference_errors[method] = [
            modal_distance(plan, states[method][h], reference) for h in sizes[:-1]
        ]
    if not all(3.6 < order < 4.5 for order in orders["mri-gark4"]):
        raise RuntimeError(f"Actual coupled PDE did not exhibit fourth order: {orders}")
    if not all(0.7 < order < 1.4 for order in orders["lagged"]):
        raise RuntimeError(f"Legacy control did not exhibit first order: {orders}")
    report = dict(
        config=asdict(plan.config),
        n=33,
        duration_seconds=duration,
        macro_steps=sizes,
        refinement="H and actual fast step H/10 halved together, identical spatial discretization",
        runs=runs,
        successive_differences=differences,
        observed_orders=orders,
        errors_vs_finest_mri=reference_errors,
        reference="MRI H=150 s; finite-step reference, not analytic truth; orders use successive differences",
        scope="Temporal convergence of this semi-discrete transient, not spatial convergence or ocean equilibrium",
    )
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    h = np.array(sizes[:-1])
    for method, label in (
        ("mri-gark4", "MRI-GARK4 + RK4"),
        ("lagged", "Legacy lagged coupling"),
    ):
        values = np.array([d["scaled_combined"] for d in differences[method]])
        ax.loglog(h, values, "o-", label=label)
        order = 4 if method == "mri-gark4" else 1
        ax.loglog(
            h,
            values[0] * (h / h[0]) ** order,
            "--",
            alpha=0.45,
            label=f"H^{order} reference slope",
        )
    ax.set(
        xlabel="Slow macro step H (seconds), fast h = H/10",
        ylabel="Scaled RMS difference: solution(H) − solution(H/2)",
        title="Actual 2D air–sea model: temporal refinement",
    )
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.savefig(args.out / "convergence.png", dpi=160)
    plt.close(fig)
    print(f"Passed: observed orders {orders}; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
